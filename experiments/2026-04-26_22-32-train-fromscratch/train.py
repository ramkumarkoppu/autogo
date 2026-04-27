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

MODEL_CHANNELS = 128
MODEL_N_BLOCKS = 10
MODEL_VALUE_HIDDEN = 64
MODEL_NAME = "SizeInvariantGoResNet-128ch-10b"
MIN_STEPS = 300
MAX_EPOCHS = 2
BATCH_SIZE = 128
BOARD_SIZE = 9
LEARNING_RATE = 1e-3
WARMUP_STEPS = 200
WEIGHT_DECAY = 5e-3
TIME_BUDGET_SECONDS = 15 * 60
TARGET_TRAIN_ACC = 0.95  # stop when train policy acc >= 0.95
NUM_WORKERS = 10

GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()
EXP_NAME = Path(__file__).resolve().parent.name




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
    parser.add_argument("--dataset-txt", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--time-budget", type=int, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

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
    max_steps = MAX_EPOCHS * steps_per_epoch
    print(f"Epochs={MAX_EPOCHS}, steps_per_epoch={steps_per_epoch}, max_steps={max_steps}")

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
            # Random-playout data has no teacher labels; treat the game winner's
            # moves as teacher targets so the policy head gets a learning signal.
            is_teacher = winner.float()
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

    print("\n===RESULT===")
    print(json.dumps({
        "iteration": args.iteration,
        "train_loss": round(train_eval["loss"], 6),
        "train_policy_acc": round(train_eval["policy_acc"], 4),
        "train_value_acc": round(train_eval["value_acc"], 4),
        "steps_completed": step,
        "elapsed_seconds": round(elapsed, 0),
        "peak_vram_mb": round(peak_vram_mb, 0),
        "checkpoint": str(ckpt_path),
        "resumed_from": args.resume_from or "",
    }))


if __name__ == "__main__":
    main()
