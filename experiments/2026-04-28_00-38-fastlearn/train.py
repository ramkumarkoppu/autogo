"""learngo-local-teacher training: policy loss masked by is_teacher flag.

Policy loss is computed per-sample (cross-entropy vs dense MCTS target) and
multiplied by is_teacher; value loss is applied to all samples. wd=0.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alpha_go.dataset import GoDataset
from alpha_go.model import SizeInvariantGoResNet, count_parameters

sys.stdout.reconfigure(line_buffering=True)

MODEL_CHANNELS = 192
MODEL_N_BLOCKS = 10
MODEL_VALUE_HIDDEN = 64
MODEL_NAME = "SizeInvariantGoResNet-192ch-10b"
MIN_STEPS = 300
MAX_EPOCHS = 30
BATCH_SIZE = 128
BOARD_SIZE = 9
LEARNING_RATE = 1e-3
WARMUP_STEPS = 200
WEIGHT_DECAY = 5e-3
TIME_BUDGET_SECONDS = 10 * 60
# Steps per second on a single rtx6000_ada (empirically ~30 for 128ch x 10b
# bs=128 with bf16). Used to size the cosine schedule so LR fully decays
# within the time budget instead of stalling at ~44% peak.
ASSUMED_STEP_RATE = 30.0
TARGET_TRAIN_ACC = 0.95  # stop when train policy acc >= 0.95
NUM_WORKERS = 10

GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()
EXP_NAME = Path(__file__).resolve().parent.name

# Held-out validation: a frozen iter-40 selfplay set from the v7 experiment.
# Same model architecture / 9x9 / NPZ schema as our data, so policy and value
# heads can be evaluated directly without retraining or a format adapter.
HOLDOUT_DATA_DIR = Path(
    "/nfs/game_data_root/experiments/"
    "2026-04-15_08-00-learngo-decen-collect-v7/selfplay-it40"
)
HOLDOUT_MAX_BATCHES = 400




def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_paths_from_txt(txt_path: Path, base_dir: Path) -> list[Path]:
    paths = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "*" in line:
                    matches = glob.glob(str(base_dir / line))
                    paths.extend([Path(m) for m in sorted(matches)])
                else:
                    path = base_dir / line
                    if path.exists():
                        paths.append(path)
    valid = []
    validated_dirs: set[Path] = set()
    for p in paths:
        if p in validated_dirs:
            valid.append(p)
            continue
        for npz_file in p.rglob("*.npz"):
            npz_dir = npz_file.parent
            if npz_dir not in validated_dirs:
                try:
                    data = dict(np.load(npz_file))
                    if "num_moves" in data.keys():
                        validated_dirs.add(npz_dir)
                        valid.append(npz_dir)
                except Exception:
                    pass
    print(f"  Resolved {len(valid)} data dirs from {len(paths)} base paths")
    return valid


def _find_dataset(txt_rel: str) -> list[Path]:
    for check_dir in [Path.cwd(), Path(os.environ.get("ALPHAGO_BASE_DIR", ".")).resolve(), GAME_DATA_DIR]:
        candidate = check_dir / txt_rel
        if candidate.exists():
            paths = _load_paths_from_txt(candidate, GAME_DATA_DIR)
            if paths:
                return paths
    raise FileNotFoundError(f"Dataset txt not found: {txt_rel}")


def augment_batch_dense(board, mcts_policy, board_size):
    B = board.shape[0]
    N = board_size
    transforms = torch.randint(0, 8, (B,), device=board.device)
    spatial_BHW = mcts_policy[:, :N*N].view(B, N, N)
    pass_B = mcts_policy[:, N*N:]
    flip_mask = transforms >= 4
    if flip_mask.any():
        board[flip_mask] = board[flip_mask].flip(-1)
        spatial_BHW[flip_mask] = spatial_BHW[flip_mask].flip(-1)
    rot = transforms % 4
    for k in [1, 2, 3]:
        mask = rot == k
        if mask.any():
            board[mask] = torch.rot90(board[mask], k, [-2, -1])
            spatial_BHW[mask] = torch.rot90(spatial_BHW[mask], k, [-2, -1])
    return board, torch.cat([spatial_BHW.view(B, N*N), pass_B], dim=-1)


def masked_loss(model, board, mcts_policy, winner, is_teacher):
    """Policy CE per-sample * is_teacher + value BCE on all samples."""
    policy_BC, value_B = model(board)
    # Per-sample CE against dense target
    logp = F.log_softmax(policy_BC, dim=-1)
    policy_ce_B = -(mcts_policy * logp).sum(dim=-1)  # (B,)
    denom = is_teacher.sum().clamp_min(1.0)
    policy_loss = (policy_ce_B * is_teacher).sum() / denom
    value_loss = F.binary_cross_entropy_with_logits(value_B, winner.float())
    return policy_loss + value_loss, policy_loss, value_loss


@torch.no_grad()
def evaluate_holdout(model, device, max_games=HOLDOUT_MAX_BATCHES):
    """Policy/value accuracy on the frozen held-out selfplay-it40 set,
    split by color to move (we played black vs we played white).

    For each NPZ position we know the side to move from `local_idx % 2`
    (even == black, odd == white). The board is canonicalised so the
    side-to-move's stones are 1 (matching how train.py feeds boards in).
    Policy target is MCTS argmax (recomputed from `mcts_visits`); value
    target is `1` iff the side-to-move ended up winning the game.
    """
    if not HOLDOUT_DATA_DIR.exists():
        print(f"WARN: holdout dir not found: {HOLDOUT_DATA_DIR}")
        return None
    npz_files = sorted(HOLDOUT_DATA_DIR.rglob("*.npz"))[:max_games]
    if not npz_files:
        return None

    boards: list[np.ndarray] = []
    targets: list[int] = []
    winners_cp: list[int] = []
    is_black: list[bool] = []
    for f in npz_files:
        d = dict(np.load(f, allow_pickle=True))
        if "boards" not in d or "winner" not in d or "mcts_visits" not in d:
            continue
        n_moves = int(d["num_moves"])
        winner = int(d["winner"])           # 1=black, 2=white, 0=draw
        boards_arr = d["boards"]            # (n_moves+1, H, W) int8
        visits = d["mcts_visits"].astype(np.float32)
        for idx in range(n_moves):
            v = visits[idx]
            if v.sum() == 0:
                continue
            stm_is_black = (idx % 2 == 0)   # side-to-move is black on even ply
            current_player = 1 if stm_is_black else 2
            board = boards_arr[idx].copy()
            if not stm_is_black:
                # Swap stones so STM is always 1.
                board = np.where(board == 1, 2, np.where(board == 2, 1, board))
            boards.append(board)
            targets.append(int(np.argmax(v)))
            winners_cp.append(1 if winner == current_player else 0)
            is_black.append(stm_is_black)

    if not boards:
        return None

    boards_t = torch.from_numpy(np.stack(boards)).float()
    target_t = torch.tensor(targets, dtype=torch.long)
    winner_t = torch.tensor(winners_cp, dtype=torch.long)

    model.eval()
    pol_pred_chunks: list[torch.Tensor] = []
    val_pred_chunks: list[torch.Tensor] = []
    for i in range(0, len(boards_t), BATCH_SIZE):
        b = boards_t[i:i + BATCH_SIZE].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            policy_logits, value_logits = model(b)
        pol_pred_chunks.append(policy_logits.argmax(dim=-1).cpu())
        val_pred_chunks.append((value_logits > 0).long().cpu())
    model.train()

    pol_pred = torch.cat(pol_pred_chunks)
    val_pred = torch.cat(val_pred_chunks)
    pol_correct = (pol_pred == target_t)
    val_correct = (val_pred == winner_t)
    is_black_t = torch.tensor(is_black, dtype=torch.bool)

    def _safe(num: torch.Tensor, den: torch.Tensor) -> float:
        d = int(den.sum().item())
        return float(num.sum().item()) / d if d else float("nan")

    n_b = int(is_black_t.sum().item())
    return {
        "policy_acc":   _safe(pol_correct, torch.ones_like(pol_correct)),
        "value_acc":    _safe(val_correct, torch.ones_like(val_correct)),
        "policy_acc_b": _safe(pol_correct & is_black_t, is_black_t),
        "policy_acc_w": _safe(pol_correct & ~is_black_t, ~is_black_t),
        "value_acc_b":  _safe(val_correct & is_black_t, is_black_t),
        "value_acc_w":  _safe(val_correct & ~is_black_t, ~is_black_t),
        "n_samples":    int(pol_correct.numel()),
        "n_samples_b":  n_b,
        "n_samples_w":  int(pol_correct.numel()) - n_b,
    }


def _write_holdout_result(iteration: int, metrics: dict) -> Path:
    out_dir = Path(__file__).resolve().parent / "holdout_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"it{iteration}.json"

    def _r(k: str) -> float | None:
        v = metrics.get(k)
        return round(v, 4) if v is not None and np.isfinite(v) else None

    payload = {
        "iteration": iteration,
        "policy_acc":   _r("policy_acc"),
        "value_acc":    _r("value_acc"),
        "policy_acc_b": _r("policy_acc_b"),
        "policy_acc_w": _r("policy_acc_w"),
        "value_acc_b":  _r("value_acc_b"),
        "value_acc_w":  _r("value_acc_w"),
        "n_samples":    int(metrics["n_samples"]),
        "n_samples_b":  int(metrics["n_samples_b"]),
        "n_samples_w":  int(metrics["n_samples_w"]),
        "holdout_dir":  str(HOLDOUT_DATA_DIR),
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out)
    return out


@torch.no_grad()
def evaluate_model(model, dataloader, device, board_size, max_batches=50):
    model.eval()
    total_loss, policy_correct, value_correct = 0.0, 0, 0
    n_samples, n_value_samples, n_batches = 0, 0, 0
    for batch in dataloader:
        if n_batches >= max_batches:
            break
        board = batch["board"].to(device)
        winner = batch["winner"].to(device)
        mcts_policy = batch["mcts_policy"].to(device)
        # For val we evaluate unmasked (full) policy CE so val_loss is comparable across iters
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _, _ = model.compute_dense_loss(board, None, mcts_policy, winner)
            policy_logits, value_logits = model(board)
        total_loss += loss.item()
        target_idx = mcts_policy.argmax(dim=-1)
        pred_idx = policy_logits.argmax(dim=-1)
        # Mirror training: teacher positions are those where the current player won.
        teacher_mask = winner.bool()
        policy_correct += ((pred_idx == target_idx) & teacher_mask).sum().item()
        pred_winner = (value_logits > 0).long()
        value_correct += (pred_winner == winner).sum().item()
        n_samples += int(teacher_mask.sum().item())
        n_value_samples += board.shape[0]
        n_batches += 1
    model.train()
    return {
        "loss": total_loss / max(1, n_batches),
        "policy_acc": policy_correct / max(1, n_samples),
        "value_acc": value_correct / max(1, n_value_samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-txt", default=None)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--time-budget", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load the iter checkpoint and run "
                             "holdout eval, writing holdout_eval/it{N}.json.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Eval-only: checkpoint path. Defaults to "
                             "/nfs/checkpoints/{EXP}/iter{N}_best.pt.")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    if args.eval_only:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else \
            Path(f"/nfs/checkpoints/{EXP_NAME}/iter{args.iteration}_best.pt")
        if not ckpt_path.exists():
            print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        model = SizeInvariantGoResNet(channels=MODEL_CHANNELS, n_blocks=MODEL_N_BLOCKS,
                                      value_hidden=MODEL_VALUE_HIDDEN).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {ckpt_path} for eval-only.")
        metrics = evaluate_holdout(model, device)
        if metrics is None:
            print("ERROR: holdout dir missing; nothing to evaluate.", file=sys.stderr)
            sys.exit(1)
        out = _write_holdout_result(args.iteration, metrics)
        print(f"holdout iter{args.iteration}: "
              f"policy_acc={metrics['policy_acc']:.4f} "
              f"(black={metrics['policy_acc_b']:.4f}, white={metrics['policy_acc_w']:.4f})  "
              f"value_acc={metrics['value_acc']:.4f} "
              f"(black={metrics['value_acc_b']:.4f}, white={metrics['value_acc_w']:.4f})  "
              f"n={metrics['n_samples']} (b={metrics['n_samples_b']}, w={metrics['n_samples_w']})")
        print(f"Wrote {out}")
        return

    if args.dataset_txt is None:
        print("ERROR: --dataset-txt required when not in --eval-only mode",
              file=sys.stderr)
        sys.exit(2)

    train_paths = _find_dataset(args.dataset_txt)
    print(f"Train data: {len(train_paths)} dirs")

    t0 = time.time()
    train_dataset = GoDataset(train_paths, load_mcts_policy=True, load_is_teacher=True, in_memory=True)
    print(f"Loaded dataset into RAM in {time.time()-t0:.1f}s")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)
    print(f"Train: {len(train_dataset):,}")
    steps_per_epoch = len(train_loader)
    epoch_cap = MAX_EPOCHS * steps_per_epoch
    # Size cosine schedule to actual training duration so LR fully decays.
    # Capped at epoch_cap to avoid running past data exhaustion semantically
    # (the train loop also bounded by time_budget).
    time_budget_for_sched = args.time_budget if args.time_budget else TIME_BUDGET_SECONDS
    sched_steps = int(time_budget_for_sched * ASSUMED_STEP_RATE)
    max_steps = min(epoch_cap, sched_steps)
    print(f"Epochs={MAX_EPOCHS} cap={epoch_cap}, sched_steps={sched_steps}, "
          f"max_steps={max_steps}, steps_per_epoch={steps_per_epoch}")

    model = SizeInvariantGoResNet(channels=MODEL_CHANNELS, n_blocks=MODEL_N_BLOCKS,
                                  value_hidden=MODEL_VALUE_HIDDEN).to(device)
    n_params = count_parameters(model)
    print(f"Model: {MODEL_NAME}, params: {n_params:,}")

    if args.resume_from:
        print(f"Loading checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    time_budget = args.time_budget if args.time_budget else TIME_BUDGET_SECONDS
    lr = args.lr if args.lr else LEARNING_RATE
    print(f"LR={lr}, WD={WEIGHT_DECAY}, time_budget={time_budget}s")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler()
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=WARMUP_STEPS, total_steps=max_steps)

    model.train()
    step = 0
    stop_training = False
    best_val_seen = False
    train_start = time.time()
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps or time.time() - train_start > time_budget or stop_training:
                break
            board = batch["board"].to(device)
            winner = batch["winner"].to(device)
            mcts_policy = batch["mcts_policy"].to(device)
            # Train policy on all MCTS positions (is_teacher=True) and on
            # winner's moves in random-playout data (no MCTS labels). Parent
            # was winner-only, throwing away ~half of MCTS-data positions.
            ds_is_teacher = batch["is_teacher"].to(device)
            is_teacher = (ds_is_teacher.bool() | winner.bool()).float()
            board, mcts_policy = augment_batch_dense(board, mcts_policy, BOARD_SIZE)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, policy_loss, value_loss = masked_loss(model, board, mcts_policy, winner, is_teacher)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % 200 == 0:
                teacher_frac = is_teacher.mean().item()
                elapsed = time.time() - train_start
                train_check = evaluate_model(model, train_loader, device, BOARD_SIZE, max_batches=20)
                print(f"  Step {step}: loss={loss.item():.4f} policy={policy_loss.item():.4f} "
                      f"value={value_loss.item():.4f} teacher_frac={teacher_frac:.2f} "
                      f"train_acc={train_check['policy_acc']:.2%} ({elapsed:.0f}s)")
                if step > MIN_STEPS and train_check["policy_acc"] >= TARGET_TRAIN_ACC:
                    print(f"    Train policy acc {train_check['policy_acc']:.2%} >= {TARGET_TRAIN_ACC:.0%}, stopping")
                    stop_training = True
        if time.time() - train_start > time_budget or stop_training:
            break

    elapsed = time.time() - train_start
    print(f"Training done: {step} steps in {elapsed:.0f}s")

    train_eval = evaluate_model(model, train_loader, device, BOARD_SIZE)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Train loss={train_eval['loss']:.4f}, policy_acc={train_eval['policy_acc']:.2%}")

    ckpt_dir = Path(f"/nfs/checkpoints/{EXP_NAME}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"iter{args.iteration}_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "step": step, "n_params": n_params,
                "config": MODEL_NAME, "iteration": args.iteration}, ckpt_path)
    print(f"Saved: {ckpt_path}")

    holdout = evaluate_holdout(model, device)
    if holdout is not None:
        _write_holdout_result(args.iteration, holdout)
        print(f"Holdout: policy_acc={holdout['policy_acc']:.4f} "
              f"(B={holdout['policy_acc_b']:.4f}, W={holdout['policy_acc_w']:.4f})  "
              f"value_acc={holdout['value_acc']:.4f} "
              f"(B={holdout['value_acc_b']:.4f}, W={holdout['value_acc_w']:.4f})  "
              f"n={holdout['n_samples']}")

    def _hr(k: str) -> float | None:
        if not holdout:
            return None
        v = holdout.get(k)
        return round(v, 4) if v is not None and np.isfinite(v) else None

    print("\n===RESULT===")
    print(json.dumps({
        "iteration": args.iteration,
        "train_loss": round(train_eval["loss"], 6),
        "train_policy_acc": round(train_eval["policy_acc"], 4),
        "train_value_acc": round(train_eval["value_acc"], 4),
        "holdout_policy_acc":   _hr("policy_acc"),
        "holdout_value_acc":    _hr("value_acc"),
        "holdout_policy_acc_b": _hr("policy_acc_b"),
        "holdout_policy_acc_w": _hr("policy_acc_w"),
        "holdout_value_acc_b":  _hr("value_acc_b"),
        "holdout_value_acc_w":  _hr("value_acc_w"),
        "holdout_n_samples":    int(holdout["n_samples"]) if holdout else 0,
        "holdout_n_samples_b":  int(holdout["n_samples_b"]) if holdout else 0,
        "holdout_n_samples_w":  int(holdout["n_samples_w"]) if holdout else 0,
        "steps_completed": step,
        "elapsed_seconds": round(elapsed, 0),
        "peak_vram_mb": round(peak_vram_mb, 0),
        "checkpoint": str(ckpt_path),
        "resumed_from": args.resume_from or "",
    }))


if __name__ == "__main__":
    main()
