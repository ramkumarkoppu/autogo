"""Analysis for learngo-local-teacher.

Plots:
- KL(MCTS || Policy) boxplots across iterations (computed from NPZ mcts_visits)
- Argmax disagreement % across iterations
- Win rate vs katago (black/white) across iterations

KL is computed on-the-fly: for each iteration's data dir we load the matching
checkpoint and score sampled positions. Results cached to policy_kl.json.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_NAME = Path(__file__).resolve().parent.name
DEFAULT_EXP = Path(f"experiments/{EXP_NAME}")
DEFAULT_DATA = Path(f"/data/eric/game_data_root/experiments/{EXP_NAME}")


def parse_games(data_dir: Path) -> pd.DataFrame:
    rows = []
    for npz in sorted(data_dir.rglob("*.npz")):
        d = dict(np.load(npz, allow_pickle=True))
        if "winner" not in d:
            continue
        ba, wa = str(d.get("black_agent", "")), str(d.get("white_agent", ""))
        is_katago = "Kata" in ba or "Kata" in wa
        our_is_black = "Kata" not in ba if is_katago else True
        winner = int(d["winner"])
        our_wins = (winner == 1 and our_is_black) or (winner == 2 and not our_is_black)
        delta = np.nan
        res = str(d.get("result", ""))
        if is_katago and "+" in res:
            try:
                pts = float(res.split("+")[1])
                winner_char = res.split("+")[0]
                delta = pts if (winner_char.startswith("B") and our_is_black) or \
                               (winner_char.startswith("W") and not our_is_black) else -pts
            except Exception:
                pass
        rows.append({
            "is_katago": is_katago, "our_is_black": our_is_black,
            "our_wins": our_wins, "delta": delta,
            "num_moves": int(d.get("num_moves", 0)),
        })
    return pd.DataFrame(rows)


def compute_kl(data_dir: Path, max_games: int = 200, positions_per_game: int = 20,
               force: bool = False) -> dict | None:
    """Compute KL(MCTS || Policy) stats from `mcts_visits` + `mcts_policy_priors` in NPZs.

    No checkpoint reload — the priors saved during MCTS are the network's policy
    at the acting checkpoint. Caches to policy_kl.json.
    """
    cache = data_dir / "policy_kl.json"
    if cache.exists() and not force:
        with open(cache) as f:
            cached = json.load(f)
        # Recompute if the cache predates the per-color split.
        if "by_color" in cached:
            return cached

    npz_files = sorted(data_dir.glob("*.npz"))[:max_games]
    if not npz_files:
        return None

    # Per-color buckets: key = "black" if position is black-to-move (idx % 2 == 0) else "white"
    buckets: dict[str, dict[str, list]] = {
        c: {"kl_divs": [], "disagree": [], "pol_ents": [], "mcts_ents": []}
        for c in ("black", "white")
    }
    for f in npz_files:
        d = dict(np.load(f, allow_pickle=True))
        if "mcts_visits" not in d or "mcts_policy_priors" not in d:
            continue
        visits = d["mcts_visits"].astype(np.float32)
        priors = d["mcts_policy_priors"].astype(np.float32)
        n_moves = int(d["num_moves"])
        idxs = np.random.choice(n_moves, min(positions_per_game, n_moves), replace=False)
        for idx in idxs:
            v = visits[idx]; s = v.sum()
            if s == 0:
                continue
            mp = v / s
            pp = priors[idx]
            mask = mp > 0
            kl = float(np.sum(mp[mask] * np.log(mp[mask] / np.clip(pp[mask], 1e-8, None))))
            m2 = pp > 0
            color = "black" if int(idx) % 2 == 0 else "white"
            b = buckets[color]
            b["kl_divs"].append(kl)
            b["disagree"].append(int(np.argmax(pp) != np.argmax(mp)))
            b["pol_ents"].append(float(-np.sum(pp[m2] * np.log(np.clip(pp[m2], 1e-8, None)))))
            b["mcts_ents"].append(float(-np.sum(mp[mask] * np.log(mp[mask]))))

    total = sum(len(b["kl_divs"]) for b in buckets.values())
    if total == 0:
        return None

    def _summarise(b):
        kl = b["kl_divs"]
        return {
            "kl_mean": float(np.mean(kl)) if kl else float("nan"),
            "kl_median": float(np.median(kl)) if kl else float("nan"),
            "kl_divs": [round(x, 5) for x in kl],
            "disagreement_pct": float(np.mean(b["disagree"]) * 100) if b["disagree"] else float("nan"),
            "policy_entropies": [round(x, 5) for x in b["pol_ents"]],
            "mcts_entropies": [round(x, 5) for x in b["mcts_ents"]],
            "n_positions": len(kl),
        }

    by_color = {c: _summarise(b) for c, b in buckets.items()}
    all_kl = [x for b in buckets.values() for x in b["kl_divs"]]
    all_dis = [x for b in buckets.values() for x in b["disagree"]]
    all_pe = [x for b in buckets.values() for x in b["pol_ents"]]
    all_me = [x for b in buckets.values() for x in b["mcts_ents"]]
    result = {
        "kl_mean": float(np.mean(all_kl)),
        "kl_median": float(np.median(all_kl)),
        "kl_divs": [round(x, 5) for x in all_kl],
        "disagreement_pct": float(np.mean(all_dis) * 100),
        "policy_entropies": [round(x, 5) for x in all_pe],
        "mcts_entropies": [round(x, 5) for x in all_me],
        "n_positions": total,
        "by_color": by_color,
    }
    with open(cache, "w") as f:
        json.dump(result, f)
    return result


def _iter_source_from_name(name: str) -> tuple[int, str] | None:
    """Return (iteration, source) for a data dir name.

    Current league naming:
    - as-black-it{N}                 -> (N, "as-black")  (iter N is black)
    - as-white-it{N}                 -> (N, "as-white")  (iter N is white)

    Legacy (older runs of this experiment, still parsed so historical KL
    cache files don't get re-derived from scratch):
    - selfplay-it{N}                 -> (N, "selfplay")
    - eval-katago-human-it{N}        -> (N, "katago_human")
    - eval-katago-<YYYY-MM-DD>-it{N} -> (N, "katago_<YYYY-MM-DD>")
    - precollect-katago              -> (-1, "katago_gtp")
    - precollect-selfplay            -> (-1, "selfplay")
    - collect-it{N}                  -> (N, "katago_gtp")
    - collect-human-it{N}            -> (N, "katago_human")
    """
    m = re.match(r"as-black-it(\d+)$", name)
    if m:
        return (int(m.group(1)), "as-black")
    m = re.match(r"as-white-it(\d+)$", name)
    if m:
        return (int(m.group(1)), "as-white")
    if name == "precollect-katago":
        return (-1, "katago_gtp")
    if name == "precollect-selfplay":
        return (-1, "selfplay")
    m = re.match(r"selfplay-it(\d+)$", name)
    if m:
        return (int(m.group(1)), "selfplay")
    m = re.match(r"eval-katago-human-it(\d+)$", name)
    if m:
        return (int(m.group(1)), "katago_human")
    m = re.match(r"eval-katago-(\d{4}-\d{2}-\d{2})-it(\d+)$", name)
    if m:
        return (int(m.group(2)), f"katago_{m.group(1)}")
    m = re.match(r"collect-human-it(\d+)$", name)
    if m:
        return (int(m.group(1)), "katago_human")
    m = re.match(r"collect-it(\d+)$", name)
    if m:
        return (int(m.group(1)), "katago_gtp")
    return None


_EMPTY_OPP = {"games": 0, "wr": np.nan, "wr_b": np.nan, "wr_w": np.nan,
              "delta": np.nan, "deltas": np.array([]),
              "deltas_b": np.array([]), "deltas_w": np.array([]),
              "games_b": 0, "games_w": 0}


def _opp_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(_EMPTY_OPP)
    kata = df[df["is_katago"]]
    kb = kata[kata["our_is_black"]]
    kw = kata[~kata["our_is_black"]]
    return {
        "games": len(kata),
        "wr": kata["our_wins"].mean() if len(kata) else np.nan,
        "wr_b": kb["our_wins"].mean() if len(kb) else np.nan,
        "wr_w": kw["our_wins"].mean() if len(kw) else np.nan,
        "delta": kata["delta"].dropna().mean() if len(kata) else np.nan,
        "deltas": kata["delta"].dropna().to_numpy() if len(kata) else np.array([]),
        "deltas_b": kb["delta"].dropna().to_numpy() if len(kb) else np.array([]),
        "deltas_w": kw["delta"].dropna().to_numpy() if len(kw) else np.array([]),
        "games_b": len(kb), "games_w": len(kw),
    }


def _finalise(agg: dict) -> dict:
    """Collapse per-source frame lists into per-opponent stat dicts."""
    frames_by_opp = agg.pop("_frames", {})
    return {
        "opponents": {
            opp: _opp_stats(pd.concat(frames, ignore_index=True))
            for opp, frames in frames_by_opp.items()
        },
        "kl": agg.get("kl", {}),
    }


def collect_iters(data_dir: Path) -> dict[int, dict]:
    """Aggregate per-iteration stats dynamically across all discovered opponents.

    Per iteration:
      - `opponents[<source>]`: dict with games/wr/wr_b/wr_w/delta for each katago variant
        (source names like `katago_human`, `katago_2026-04-06`, legacy `katago_gtp`).
      - `kl[<source>]`: policy_kl.json-style dict, one per source (each katago_* and selfplay).
    """
    by_iter: dict[int, dict] = {}
    for d in sorted(data_dir.iterdir()) if data_dir.exists() else []:
        if not d.is_dir():
            continue
        match = _iter_source_from_name(d.name)
        if match is None:
            continue
        it, source = match
        df = parse_games(d)
        if df.empty:
            continue
        agg = by_iter.setdefault(it, {})
        if source.startswith("katago_"):
            agg.setdefault("_frames", {}).setdefault(source, []).append(df)
        # KL stored per source so we can compare policy behaviour against each opponent.
        if source.startswith("katago_") or source == "selfplay":
            kl_bucket = agg.setdefault("kl", {})
            if source not in kl_bucket:
                kl = compute_kl(d)
                if kl is not None:
                    kl_bucket[source] = kl
    return {k: _finalise(v) for k, v in by_iter.items()}


def _all_opponents(stats: dict[int, dict]) -> list[str]:
    """Every katago opponent that showed up in any iteration, sorted for stable colors."""
    return sorted({o for s in stats.values() for o in s.get("opponents", {})})


def _all_kl_sources(stats: dict[int, dict]) -> list[str]:
    """Every KL source (opponents + selfplay) that showed up, sorted."""
    return sorted({k for s in stats.values() for k in s.get("kl", {})})


def _label(source: str) -> str:
    """Human-readable label for legend entries (katago_2026-04-06 → katago-2026-04-06)."""
    return source.replace("_", "-") if source != "selfplay" else "self-play"


def _color_map(keys: list[str]) -> dict[str, tuple]:
    """Stable color per key via tab20. Sorted keys → deterministic assignment."""
    cmap = plt.get_cmap("tab20")
    return {k: cmap(i % 20) for i, k in enumerate(keys)}


def plot_kl(exp_dir: Path, stats: dict[int, dict]):
    sources = _all_kl_sources(stats)
    items = [(it, s) for it, s in sorted(stats.items()) if s.get("kl")]
    if not items or not sources:
        print("No KL data — no NPZs with mcts_visits or no matching checkpoints")
        return
    labels = [str(it) for it, _ in items]
    xs = list(range(len(items)))
    colors = _color_map(sources)
    n = len(sources)
    slot = 0.85 / n
    offsets = [(-((n - 1) / 2) + i) * slot for i in range(n)]

    def _color_kl(s: dict, source: str, color: str) -> dict | None:
        """Per-color sub-dict for a given source at iteration state `s`, or None."""
        kl = s.get("kl", {}).get(source)
        return (kl.get("by_color") or {}).get(color) if kl else None

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    fig.suptitle(f"{EXP_NAME}: Policy vs MCTS (by source, split by color to move)",
                 fontweight="bold")

    def _grouped_boxplot(ax, color, field, ylabel, title):
        handles = []
        for source, off in zip(sources, offsets):
            vals = [(_color_kl(s, source, color) or {}).get(field) or [np.nan]
                    for _, s in items]
            bp = ax.boxplot(vals, positions=[x + off for x in xs],
                            widths=slot * 0.9, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor=colors[source], alpha=0.6),
                            medianprops=dict(color="black"))
            handles.append(bp["boxes"][0])
        ax.legend(handles, [_label(s) for s in sources], fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")

    def _disagree_bars(ax, color, title):
        for source, off in zip(sources, offsets):
            vals = [(_color_kl(s, source, color) or {}).get("disagreement_pct", np.nan)
                    for _, s in items]
            ax.bar([x + off for x in xs], vals, width=slot * 0.9,
                   color=colors[source], alpha=0.7, label=_label(source))
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Disagreement (%)")
        ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    for row, color in enumerate(("black", "white")):
        _grouped_boxplot(axes[row, 0], color, "kl_divs", "KL (nats)",
                         f"KL(MCTS || Policy) — {color} to move")
        _grouped_boxplot(axes[row, 1], color, "mcts_entropies", "Entropy (nats)",
                         f"MCTS entropy — {color} to move")
        _disagree_bars(axes[row, 2], color, f"Argmax disagreement — {color} to move")

    for col in range(3):
        y_lo = min(axes[0, col].get_ylim()[0], axes[1, col].get_ylim()[0])
        y_hi = max(axes[0, col].get_ylim()[1], axes[1, col].get_ylim()[1])
        axes[0, col].set_ylim(y_lo, y_hi); axes[1, col].set_ylim(y_lo, y_hi)

    plt.tight_layout()
    out = exp_dir / "kl_divergence.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def _boards_per_checkpoint(exp_dir: Path, base_dir: Path,
                           iters: list[int]) -> dict[int, int]:
    """Training-board count for each iteration's checkpoint.

    Parses `dataset-it{N}.txt` (the exact manifest fed to train.py), resolves
    each referenced directory under `base_dir` (the shared game_data_root),
    and sums `num_moves` across every NPZ. Handles cross-experiment bootstrap
    data correctly because the manifest paths already include those dirs.

    Per-dir results cached in `boards_cache.json`. Cache is invalidated for
    any dir whose NPZ count has grown (frozen iters stay cached; in-progress
    iters get re-walked).
    """
    cache_path = exp_dir / "boards_cache.json"
    try:
        cache: dict[str, dict] = json.loads(cache_path.read_text()) \
            if cache_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        cache = {}
    out: dict[int, int] = {}
    dirty = False
    for it in iters:
        txt = exp_dir / f"dataset-it{it}.txt"
        if not txt.exists():
            out[it] = 0
            continue
        total = 0
        for raw in txt.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            d = base_dir / line
            npzs = sorted(d.rglob("*.npz")) if d.exists() else []
            cur = len(npzs)
            entry = cache.get(line)
            if entry is None or entry.get("npz_count", -1) != cur:
                n = 0
                for npz in npzs:
                    try:
                        with np.load(npz, allow_pickle=True) as data:
                            if "num_moves" in data.files:
                                n += int(data["num_moves"])
                    except Exception:
                        pass
                cache[line] = {"boards": n, "npz_count": cur}
                dirty = True
            total += cache[line]["boards"]
        out[it] = total
    if dirty:
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return out


def _load_train_results(exp_dir: Path) -> dict[int, dict]:
    """Parse per-iteration train results `===RESULT===` JSON lines.

    Supports both the current layout (`logs/itN/train.log`) and the legacy
    flat layout (`logs/train-itN.log`) so iterations that haven't been
    migrated yet still show up.
    """
    logs = exp_dir / "logs"
    candidates = list(logs.glob("it*/train.log")) + list(logs.glob("train-it*.log"))
    out: dict[int, dict] = {}
    for log in sorted(candidates):
        text = log.read_text()
        marker = "===RESULT==="
        idx = text.rfind(marker)
        if idx < 0:
            continue
        tail = text[idx + len(marker):].strip().splitlines()
        if not tail:
            continue
        try:
            rec = json.loads(tail[0])
        except json.JSONDecodeError:
            continue
        out[int(rec["iteration"])] = rec
    return out


def _load_collect_timing(exp_dir: Path) -> dict[int, dict]:
    """Parse per-iteration collect stats from timing/collect-itN.json."""
    out: dict[int, dict] = {}
    for f in sorted((exp_dir / "timing").glob("collect-it*.json")) if (exp_dir / "timing").exists() else []:
        with open(f) as fp:
            rec = json.load(fp)
        out[int(rec["iteration"])] = rec
    return out


def plot_wall_time(exp_dir: Path):
    """Grouped bar chart: collect + train wall-time per iteration, in minutes.

    Sourced from `timing/collect-it*.json` (written by collect_driver.py) and
    train `===RESULT===` JSON in `logs/train-it*.log`. Missing values are
    shown as zero-height bars so the iteration index still appears.
    """
    collect_timing = _load_collect_timing(exp_dir)
    train_results = _load_train_results(exp_dir)
    iters = sorted(set(collect_timing) | set(train_results))
    if not iters:
        print("No timing data — run a collect iteration first")
        return

    collect_min = [collect_timing.get(i, {}).get("elapsed_seconds", 0) / 60.0 for i in iters]
    train_min = [train_results.get(i, {}).get("elapsed_seconds", 0) / 60.0 for i in iters]

    # Also derive a rough throughput marker: games completed per minute during
    # collect. Prefer the explicit `total_games` field (current league driver
    # writes it); fall back to the legacy heuristic for older timing files
    # whose `modes` are "selfplay" / "eval-*" with num_games_per_mode meaning
    # per-color.
    throughput = []
    for i in iters:
        c = collect_timing.get(i)
        if not c or c.get("elapsed_seconds", 0) <= 0:
            throughput.append(np.nan)
            continue
        if "total_games" in c:
            total_games = int(c["total_games"])
        else:
            modes = c.get("modes", [])
            n = c.get("num_games_per_mode", 0)
            total_games = sum(n if m == "selfplay" else 2 * n for m in modes)
        throughput.append(total_games / (c["elapsed_seconds"] / 60.0))

    xs = np.arange(len(iters))
    w = 0.4

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{EXP_NAME}: per-iteration wall time", fontweight="bold")

    ax.bar(xs - w/2, collect_min, w, label="collect", color="tab:blue", alpha=0.8)
    ax.bar(xs + w/2, train_min, w, label="train", color="tab:orange", alpha=0.8)
    for x, c, t in zip(xs, collect_min, train_min):
        if c:
            ax.text(x - w/2, c, f"{c:.0f}", ha="center", va="bottom", fontsize=8)
        if t:
            ax.text(x + w/2, t, f"{t:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in iters])
    ax.set_xlabel("Iteration"); ax.set_ylabel("Wall time (minutes)")
    ax.set_title("Collect vs train duration"); ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax2.plot(iters, throughput, "o-", color="tab:green")
    for x, y in zip(iters, throughput):
        if np.isfinite(y):
            ax2.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Games / min")
    ax2.set_title("Collect throughput (attempted games / collect wall min)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = exp_dir / "iteration_timing.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def plot_progress(exp_dir: Path, stats: dict[int, dict], data_dir: Path):
    if not stats:
        return
    iters = sorted(stats.keys())
    train_results = _load_train_results(exp_dir)
    opponents = _all_opponents(stats)
    colors = _color_map(opponents)

    def _wr_series(opp: str, key: str) -> list[float]:
        """WR (%) per iteration, NaN if opp absent that iteration."""
        return [
            (stats[i]["opponents"].get(opp, _EMPTY_OPP)[key] or np.nan) * 100
            for i in iters
        ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))
    fig.suptitle(f"{EXP_NAME}: progress per katago opponent", fontweight="bold")

    # Row 0, cols 0-2: WR vs each opponent — overall / as-black / as-white.
    for ax, key, title in zip(
        axes[0, :3],
        ("wr", "wr_b", "wr_w"),
        ("Win rate (overall)", "Win rate as BLACK", "Win rate as WHITE"),
    ):
        for opp in opponents:
            ax.plot(iters, _wr_series(opp, key), "o-", color=colors[opp], label=_label(opp))
        ax.set_xlabel("Iteration"); ax.set_ylabel("Win Rate (%)")
        ax.set_title(title); ax.grid(alpha=0.3)
        ax.set_ylim(-2, 102)
    axes[0, 0].legend(fontsize=8, loc="best")

    # Row 0, col 3: WR vs number of training boards for the checkpoint being
    # measured. `dataset-it{N}.txt` is the manifest train.py uses for iter N,
    # so summing `num_moves` across every NPZ it references gives the exact
    # training-board count (including bootstrap data from other experiments).
    # Cleaner cross-experiment comparison than walltime, which depends on
    # cluster size.
    boards_per_iter = _boards_per_checkpoint(exp_dir, data_dir.parent.parent, iters)

    def _agg_wr(i: int) -> float:
        opps = stats[i]["opponents"]
        games = sum(o["games"] for o in opps.values())
        if games == 0:
            return np.nan
        wins = sum(o["games"] * o["wr"] for o in opps.values()
                   if o["games"] and np.isfinite(o["wr"]))
        return wins / games * 100

    ax = axes[0, 3]
    xs_boards = [boards_per_iter.get(i, 0) for i in iters]
    for opp in opponents:
        ax.plot(xs_boards, _wr_series(opp, "wr"), "o-",
                color=colors[opp], alpha=0.4, linewidth=1, label=_label(opp))
    ax.plot(xs_boards, [_agg_wr(i) for i in iters], "k-D",
            linewidth=2, label="aggregate")
    ax.set_xlabel("Training boards (cumulative num_moves)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win rate vs training boards")
    ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
    ax.set_ylim(-2, 102)

    # Row 1, col 0-1: Point delta boxplots split by our color.
    # N opponents per iteration → pack them across a shared slot width.
    xs = list(range(len(iters)))
    n = max(len(opponents), 1)
    slot = 0.85 / n
    offsets = [(-((n - 1) / 2) + i) * slot for i in range(n)]
    _safe = lambda arrs: [a if len(a) else np.array([np.nan]) for a in arrs]

    for ax, side, title in zip(
        axes[1, :2],
        ("deltas_b", "deltas_w"),
        ("Point delta (we played black)", "Point delta (we played white)"),
    ):
        handles = []
        for opp, off in zip(opponents, offsets):
            vals = [stats[i]["opponents"].get(opp, _EMPTY_OPP)[side] for i in iters]
            bp = ax.boxplot(
                _safe(vals), positions=[x + off for x in xs],
                widths=slot * 0.9, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=colors[opp], alpha=0.6),
                medianprops=dict(color="black"),
            )
            handles.append(bp["boxes"][0])
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in iters])
        if handles:
            ax.legend(handles, [_label(o) for o in opponents], fontsize=8)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Point delta (our - kata)")
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")

    # Share ylim across the black/white delta panels.
    y_lo = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    y_hi = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(y_lo, y_hi); axes[1, 1].set_ylim(y_lo, y_hi)

    # Row 1, col 2: Train accuracy.
    ax = axes[1, 2]
    if train_results:
        t_iters = sorted(train_results.keys())
        pol = [train_results[i].get("train_policy_acc", np.nan) * 100 for i in t_iters]
        val = [train_results[i].get("train_value_acc", np.nan) * 100 for i in t_iters]
        ax.plot(t_iters, pol, "b-o", label="train policy acc")
        ax.plot(t_iters, val, "g-s", label="train value acc", alpha=0.8)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Train accuracy"); ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.set_title("Train accuracy (no logs/train-it*.log yet)")

    axes[1, 3].axis("off")

    plt.tight_layout()
    out = exp_dir / "progress.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def print_summary(stats: dict[int, dict], exp_dir: Path):
    train_results = _load_train_results(exp_dir)
    opponents = _all_opponents(stats)
    kl_sources = _all_kl_sources(stats)

    # Header: one (games, wr%, dlt) block per opponent, then train accs, then
    # one (kl_med, dis%) block per KL source.
    header = [f"{'iter':>4}"]
    for opp in opponents:
        short = _label(opp).replace("katago-", "k-")[:12]
        header += [f"{'G_' + short:>8}", f"{'wr_' + short + '%':>10}",
                   f"{'dlt_' + short:>10}"]
    header += [f"{'pol_acc':>8}", f"{'val_acc':>8}"]
    for source in kl_sources:
        short = _label(source).replace("katago-", "k-")[:10]
        header += [f"{'KLmed_' + short:>12}", f"{'dis_' + short + '%':>11}"]
    print("\n" + " ".join(header))
    print("-" * sum(len(h) + 1 for h in header))

    for it in sorted(stats.keys()):
        s = stats[it]
        row = [f"{it:>4}"]
        for opp in opponents:
            o = s["opponents"].get(opp, _EMPTY_OPP)
            row += [f"{o['games']:>8}",
                    f"{o['wr']*100:>9.1f}%" if o['games'] else f"{'-':>10}",
                    f"{o['delta']:>+10.2f}" if np.isfinite(o['delta']) else f"{'-':>10}"]
        tr = train_results.get(it, {})
        ta = tr.get("train_policy_acc", np.nan)
        va = tr.get("train_value_acc", np.nan)
        row += [f"{ta*100:>7.1f}%", f"{va*100:>7.1f}%"]
        for source in kl_sources:
            k = s.get("kl", {}).get(source, {}) or {}
            row += [f"{k.get('kl_median', np.nan):>12.3f}",
                    f"{k.get('disagreement_pct', np.nan):>10.1f}%"]
        print(" ".join(row))


def plot_league(exp_dir: Path):
    """Step plot: per-iteration league champion for black and white.

    Source: `league_state.json["history"]` (one record appended per call to
    update_league.py — `{iter, best_black, best_white}`). Y axis is the
    champion's iteration number; the diagonal `y == x` would mean every iter
    immediately dethrones the prior champion.
    """
    state_file = exp_dir / "league_state.json"
    if not state_file.exists():
        print(f"No {state_file.name} yet — run a collect iteration first")
        return
    history = json.loads(state_file.read_text()).get("history", [])
    if not history:
        print(f"{state_file.name} has empty history")
        return
    iters = [h["iter"] for h in history]
    best_black = [h["best_black"] for h in history]
    best_white = [h["best_white"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(iters, best_black, "o-", where="post", color="black",
            label="best black-side checkpoint")
    ax.step(iters, best_white, "s-", where="post", color="tab:blue",
            label="best white-side checkpoint")
    ax.plot(iters, iters, ":", color="grey", alpha=0.5, label="y = x (always dethroned)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Champion checkpoint (iteration #)")
    ax.set_title(f"{EXP_NAME}: league champions over time")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_xticks(iters)
    ax.set_yticks(sorted(set(best_black) | set(best_white)))

    plt.tight_layout()
    out = exp_dir / "league.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", default=str(DEFAULT_EXP))
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--mode", choices=["summary", "plot", "both"], default="both")
    args = p.parse_args()
    exp_dir = Path(args.exp_dir); data_dir = Path(args.data_dir)
    stats = collect_iters(data_dir)
    if args.mode in ("summary", "both"):
        print_summary(stats, exp_dir)
    if args.mode in ("plot", "both"):
        plot_progress(exp_dir, stats, data_dir)
        plot_kl(exp_dir, stats)
        plot_wall_time(exp_dir)
        plot_league(exp_dir)


if __name__ == "__main__":
    main()
