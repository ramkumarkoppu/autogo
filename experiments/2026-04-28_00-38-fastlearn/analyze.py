"""Analysis for the champion-only learngo fork.

Plots:
- progress.png   — per-iteration win rate vs the reigning league champion
                   (as black vs best_white, as white vs best_black), with the
                   0.55 promotion threshold drawn in.
- league.png     — step plot of the champion checkpoint per iteration, sourced
                   from `league_state.json["history"]`.
- iteration_timing.png — collect+train wall-time per iter (sourced from
                         `timing/collect-it*.json` and the `===RESULT===`
                         line at the end of each `logs/it*/train.log`).
- holdout_eval.png — policy + value accuracy on a frozen held-out NPZ set.
- eval_extra.png — win rate + point-delta boxplots against external eval
                   opponents launched via launch_eval_extra.py.
- train_accuracy.png — train_policy_acc / train_value_acc per iter from
                       each `logs/it*/train.log` `===RESULT===` line.
- kl_divergence.png — KL(MCTS || Policy) + MCTS entropy + argmax
                      disagreement % across iterations, split by color to
                      move, one series per data source (selfplay /
                      as-black / as-white / eval-* if present).
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_NAME = Path(__file__).resolve().parent.name
DEFAULT_EXP = Path(f"experiments/{EXP_NAME}")
DEFAULT_DATA = Path(f"/nfs/game_data_root/experiments/{EXP_NAME}")

PROMOTION_THRESHOLD = 0.55

# Match every `eval-<source>-it<N>` data dir under DEFAULT_DATA.
_EVAL_DIR_RE = re.compile(r"^eval-(.+)-it(\d+)$")


def _figures_dir(exp_dir: Path) -> Path:
    """Return `<exp_dir>/figures/`, creating it on first use."""
    out = exp_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_league(exp_dir: Path) -> dict | None:
    f = exp_dir / "league_state.json"
    return json.loads(f.read_text()) if f.exists() else None


def _load_train_results(exp_dir: Path) -> dict[int, dict]:
    """Parse per-iteration train results from `===RESULT===` JSON lines."""
    out: dict[int, dict] = {}
    for log in sorted((exp_dir / "logs").glob("it*/train.log")) \
            if (exp_dir / "logs").exists() else []:
        text = log.read_text()
        idx = text.rfind("===RESULT===")
        if idx < 0:
            continue
        tail = text[idx + len("===RESULT==="):].strip().splitlines()
        if not tail:
            continue
        try:
            rec = json.loads(tail[0])
        except json.JSONDecodeError:
            continue
        out[int(rec["iteration"])] = rec
    return out


def _load_holdout_eval(exp_dir: Path) -> dict[int, dict]:
    """Read holdout_eval/it*.json (written by train.py)."""
    out: dict[int, dict] = {}
    h_dir = exp_dir / "holdout_eval"
    if not h_dir.exists():
        return out
    for f in sorted(h_dir.glob("it*.json")):
        with open(f) as fp:
            rec = json.load(fp)
        out[int(rec["iteration"])] = rec
    return out


def plot_holdout(exp_dir: Path) -> None:
    """Policy + value accuracy on the held-out selfplay-it40 set, per iter,
    split by side-to-move (we played black vs we played white).

    Two panels: policy-acc on the left, value-acc on the right; each panel
    shows the overall curve plus the black-to-move and white-to-move splits.
    Reads holdout_eval/it*.json (written by train.py at end-of-train and by
    `--eval-only` retroactive jobs).
    """
    by_iter = _load_holdout_eval(exp_dir)
    if not by_iter:
        print("No holdout_eval/ data — run train.py with --eval-only on the "
              "checkpoints (or wait for a fresh training iter).")
        return
    iters = sorted(by_iter)
    n_samp = by_iter[iters[0]].get("n_samples", "?")
    n_b = by_iter[iters[0]].get("n_samples_b", "?")
    n_w = by_iter[iters[0]].get("n_samples_w", "?")

    def _series(key):
        return [by_iter[i].get(key) * 100 if by_iter[i].get(key) is not None
                else np.nan for i in iters]

    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f"{EXP_NAME}: held-out accuracy (selfplay-it40,"
                 f" n={n_samp}: B={n_b}, W={n_w})", fontweight="bold")

    panels = [
        (ax_p, "Policy accuracy", "policy_acc",   "policy_acc_b",   "policy_acc_w"),
        (ax_v, "Value accuracy",  "value_acc",    "value_acc_b",    "value_acc_w"),
    ]
    for ax, title, k_all, k_b, k_w in panels:
        ax.plot(iters, _series(k_all), "k-o",  label="overall", linewidth=2)
        ax.plot(iters, _series(k_b),   "C0--D", label="we played BLACK", alpha=0.9)
        ax.plot(iters, _series(k_w),   "C3:s",  label="we played WHITE", alpha=0.9)
        ax.set_xlabel("Iteration")
        ax.set_xticks(iters)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    ax_p.set_ylabel("Accuracy (%)")

    plt.tight_layout()
    out = _figures_dir(exp_dir) / "holdout_eval.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def _parse_delta(result: str, our_is_black: bool) -> float:
    """`B+13.5` / `W+44.5` -> signed point delta from our perspective.

    Returns NaN if the result string is missing the score (e.g. resign-only
    or empty). Signed so positive == we won by that margin.
    """
    if "+" not in result:
        return float("nan")
    side, _, pts_s = result.partition("+")
    try:
        pts = float(pts_s)
    except ValueError:
        return float("nan")
    we_won = (side.startswith("B") and our_is_black) or \
             (side.startswith("W") and not our_is_black)
    return pts if we_won else -pts


def _eval_stats_by_iter(data_dir: Path) -> dict[str, dict[int, dict]]:
    """Walk eval-*-it{N}/ NPZs and aggregate our-side stats per (source, iter).

    Returns {source: {iter: {games, wins, games_b, wins_b, games_w, wins_w,
                              deltas, deltas_b, deltas_w}}}.
    "Our" side is the agent whose name doesn't contain "Kata". Iterations
    with no NPZs are dropped. `deltas*` are lists of signed point margins
    (our score - opponent score), only populated when the NPZ's `result`
    string carries a numeric margin.
    """
    out: dict[str, dict[int, dict]] = {}
    if not data_dir.exists():
        return out
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        m = _EVAL_DIR_RE.match(d.name)
        if not m:
            continue
        source, it = m.group(1), int(m.group(2))
        bucket = out.setdefault(source, {}).setdefault(
            it, {"games": 0, "wins": 0, "games_b": 0, "wins_b": 0,
                 "games_w": 0, "wins_w": 0,
                 "deltas": [], "deltas_b": [], "deltas_w": []})
        for npz in sorted(d.rglob("*.npz")):
            data = dict(np.load(npz, allow_pickle=True))
            if "winner" not in data:
                continue
            ba = str(data.get("black_agent", ""))
            our_is_black = "Kata" not in ba
            winner = int(data["winner"])
            if winner not in (1, 2):
                continue
            our_won = (winner == 1 and our_is_black) or (winner == 2 and not our_is_black)
            bucket["games"] += 1
            bucket["wins"] += int(our_won)
            delta = _parse_delta(str(data.get("result", "")), our_is_black)
            if np.isfinite(delta):
                bucket["deltas"].append(delta)
            if our_is_black:
                bucket["games_b"] += 1
                bucket["wins_b"] += int(our_won)
                if np.isfinite(delta):
                    bucket["deltas_b"].append(delta)
            else:
                bucket["games_w"] += 1
                bucket["wins_w"] += int(our_won)
                if np.isfinite(delta):
                    bucket["deltas_w"].append(delta)
    return {s: {i: v for i, v in by_it.items() if v["games"] > 0}
            for s, by_it in out.items()}


def plot_eval_extra(exp_dir: Path, data_dir: Path) -> None:
    """Win rate + point-delta boxplots vs each eval opponent, per iteration.

    Sourced from `<data_dir>/eval-<source>-it{N}/`. Top row: overall +
    as-black + as-white win-rate lines per opponent. Bottom row: matching
    point-delta boxplots (our score - opponent score) per iter, one box per
    opponent at each iter slot. Iters with no games are skipped.
    """
    by_source = _eval_stats_by_iter(data_dir)
    if not by_source:
        print(f"No eval-*-it*/ data under {data_dir} — nothing to plot")
        return

    sources = sorted(by_source)
    cmap = plt.get_cmap("tab10")
    colors = {s: cmap(i % 10) for i, s in enumerate(sources)}
    all_iters = sorted({i for by_it in by_source.values() for i in by_it})

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{EXP_NAME}: eval vs external opponents", fontweight="bold")

    # Top row: win-rate lines.
    for ax, (g_key, w_key, title) in zip(axes[0], [
        ("games",   "wins",   "Win rate (overall)"),
        ("games_b", "wins_b", "Win rate as BLACK"),
        ("games_w", "wins_w", "Win rate as WHITE"),
    ]):
        for source in sources:
            its = sorted(by_source[source])
            wr = [(by_source[source][i][w_key] / by_source[source][i][g_key] * 100)
                  if by_source[source][i][g_key] else np.nan for i in its]
            ax.plot(its, wr, "o-", color=colors[source], label=source)
            for x, y in zip(its, wr):
                if np.isfinite(y):
                    n = by_source[source][x][g_key]
                    ax.text(x, y, f"{y:.0f}%\n(n={n})", ha="center", va="bottom",
                            fontsize=7, color=colors[source])
        ax.axhline(50, ls=":", color="grey", alpha=0.5)
        ax.set_xlabel("Iteration"); ax.set_title(title)
        ax.set_xticks(all_iters)
        ax.set_ylim(-2, 110); ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Win rate (%)")
    axes[0, 0].legend(loc="best", fontsize=9)

    # Bottom row: point-delta boxplots — one box per (source, iter), with
    # source slots packed inside each iter's x-position so multiple opponents
    # don't overlap.
    n_src = max(len(sources), 1)
    slot = 0.8 / n_src
    offsets = [(-((n_src - 1) / 2) + i) * slot for i in range(n_src)]
    iter_to_x = {it: x for x, it in enumerate(all_iters)}

    for ax, (key, title) in zip(axes[1], [
        ("deltas",   "Point delta (overall)"),
        ("deltas_b", "Point delta (we played BLACK)"),
        ("deltas_w", "Point delta (we played WHITE)"),
    ]):
        handles = []
        for source, off in zip(sources, offsets):
            xs = []; vals = []
            for it in all_iters:
                d = by_source[source].get(it, {}).get(key, [])
                xs.append(iter_to_x[it] + off)
                vals.append(d if d else [np.nan])
            bp = ax.boxplot(
                vals, positions=xs, widths=slot * 0.9,
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor=colors[source], alpha=0.6),
                medianprops=dict(color="black"))
            handles.append(bp["boxes"][0])
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(list(iter_to_x.values()))
        ax.set_xticklabels([str(i) for i in all_iters])
        ax.set_xlabel("Iteration"); ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if handles:
            ax.legend(handles, sources, fontsize=9, loc="best")
    axes[1, 0].set_ylabel("Point delta (our - opp)")

    # Share y-limits across the three delta panels.
    y_lo = min(a.get_ylim()[0] for a in axes[1])
    y_hi = max(a.get_ylim()[1] for a in axes[1])
    for a in axes[1]:
        a.set_ylim(y_lo, y_hi)

    plt.tight_layout()
    out = _figures_dir(exp_dir) / "eval_extra.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def plot_train_accuracy(exp_dir: Path) -> None:
    """Train policy + value accuracy per iter, from train.log `===RESULT===`."""
    train_results = _load_train_results(exp_dir)
    if not train_results:
        print("No train results yet — train.log RESULT lines missing.")
        return
    iters = sorted(train_results)
    pol = [train_results[i].get("train_policy_acc", np.nan) * 100 for i in iters]
    val = [train_results[i].get("train_value_acc", np.nan) * 100 for i in iters]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{EXP_NAME}: train accuracy per iteration", fontweight="bold")
    ax.plot(iters, pol, "b-o", label="train policy acc")
    ax.plot(iters, val, "g-s", label="train value acc", alpha=0.8)
    for x, y in zip(iters, pol):
        if np.isfinite(y):
            ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8, color="b")
    for x, y in zip(iters, val):
        if np.isfinite(y):
            ax.text(x, y, f"{y:.1f}", ha="center", va="top", fontsize=8, color="g")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(iters); ax.set_ylim(0, 100)
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = _figures_dir(exp_dir) / "train_accuracy.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


# --- KL(MCTS || Policy) ----------------------------------------------------
# Same shape as the parent fork's analyze.py, adapted for this fork's data
# layout: per-chunk subdirs (rglob), and source labels include the
# champion-mode dirs (as-black, as-white, selfplay) plus any eval-* dirs.

def _iter_source_from_name(name: str) -> tuple[int, str] | None:
    """Map an experiment data-dir name to (iter, source_label)."""
    m = re.match(r"^as-black-it(\d+)$", name)
    if m:
        return int(m.group(1)), "as-black"
    m = re.match(r"^as-white-it(\d+)$", name)
    if m:
        return int(m.group(1)), "as-white"
    m = re.match(r"^selfplay-it(\d+)$", name)
    if m:
        return int(m.group(1)), "selfplay"
    m = _EVAL_DIR_RE.match(name)
    if m:
        return int(m.group(2)), m.group(1)  # source = the bit between eval- and -it{N}
    return None


def compute_kl(data_dir: Path, max_games: int = 200, positions_per_game: int = 20,
               force: bool = False) -> dict | None:
    """Compute KL(MCTS || Policy) stats from `mcts_visits` + `mcts_policy_priors`.

    Reads NPZs recursively (per-chunk subdirs) and caches the result to
    `<data_dir>/policy_kl.json` so re-runs are cheap. Returns None if no
    NPZs carry MCTS targets (e.g. random-vs-random bootstrap data).
    """
    cache = data_dir / "policy_kl.json"
    if cache.exists() and not force:
        try:
            cached = json.loads(cache.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}
        if "by_color" in cached:
            return cached

    npz_files = sorted(data_dir.rglob("*.npz"))[:max_games]
    if not npz_files:
        return None

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
        idxs = np.random.choice(n_moves, min(positions_per_game, n_moves),
                                replace=False)
        for idx in idxs:
            v = visits[idx]; s = v.sum()
            if s == 0:
                continue
            mp = v / s
            pp = priors[idx]
            mask = mp > 0
            kl = float(np.sum(mp[mask] * np.log(mp[mask] /
                                                np.clip(pp[mask], 1e-8, None))))
            m2 = pp > 0
            color = "black" if int(idx) % 2 == 0 else "white"
            b = buckets[color]
            b["kl_divs"].append(kl)
            b["disagree"].append(int(np.argmax(pp) != np.argmax(mp)))
            b["pol_ents"].append(
                float(-np.sum(pp[m2] * np.log(np.clip(pp[m2], 1e-8, None)))))
            b["mcts_ents"].append(
                float(-np.sum(mp[mask] * np.log(mp[mask]))))

    total = sum(len(b["kl_divs"]) for b in buckets.values())
    if total == 0:
        return None

    def _summarise(b):
        kl = b["kl_divs"]
        return {
            "kl_mean": float(np.mean(kl)) if kl else float("nan"),
            "kl_median": float(np.median(kl)) if kl else float("nan"),
            "kl_divs": [round(x, 5) for x in kl],
            "disagreement_pct": (float(np.mean(b["disagree"]) * 100)
                                 if b["disagree"] else float("nan")),
            "policy_entropies": [round(x, 5) for x in b["pol_ents"]],
            "mcts_entropies": [round(x, 5) for x in b["mcts_ents"]],
            "n_positions": len(kl),
        }

    by_color = {c: _summarise(b) for c, b in buckets.items()}
    result = {
        "by_color": by_color,
        "n_positions": total,
    }
    cache.write_text(json.dumps(result))
    return result


def _collect_kl(data_dir: Path) -> dict[int, dict[str, dict]]:
    """{iter: {source: kl_dict}} for every recognised data dir under data_dir."""
    out: dict[int, dict[str, dict]] = {}
    if not data_dir.exists():
        return out
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        match = _iter_source_from_name(d.name)
        if match is None:
            continue
        it, source = match
        kl = compute_kl(d)
        if kl is not None:
            out.setdefault(it, {})[source] = kl
    return out


def plot_kl(exp_dir: Path, data_dir: Path) -> None:
    """KL(MCTS || Policy) + MCTS entropy + argmax disagreement, split by color."""
    by_iter = _collect_kl(data_dir)
    if not by_iter:
        print(f"No KL data under {data_dir} — no NPZs with mcts_visits / priors")
        return
    sources = sorted({s for d in by_iter.values() for s in d})
    items = sorted(by_iter.items())
    labels = [str(it) for it, _ in items]
    xs = list(range(len(items)))
    cmap = plt.get_cmap("tab10")
    colors = {s: cmap(i % 10) for i, s in enumerate(sources)}
    n = len(sources)
    slot = 0.85 / n
    offsets = [(-((n - 1) / 2) + i) * slot for i in range(n)]

    def _color_kl(s_dict: dict, source: str, color: str) -> dict | None:
        kl = s_dict.get(source)
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
                            widths=slot * 0.9, showfliers=False,
                            patch_artist=True,
                            boxprops=dict(facecolor=colors[source], alpha=0.6),
                            medianprops=dict(color="black"))
            handles.append(bp["boxes"][0])
        ax.legend(handles, sources, fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")

    def _disagree_bars(ax, color, title):
        for source, off in zip(sources, offsets):
            vals = [(_color_kl(s, source, color) or {}).get("disagreement_pct",
                                                            np.nan)
                    for _, s in items]
            ax.bar([x + off for x in xs], vals, width=slot * 0.9,
                   color=colors[source], alpha=0.7, label=source)
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
    out = _figures_dir(exp_dir) / "kl_divergence.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def _load_collect_timing(exp_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    timing_dir = exp_dir / "timing"
    if not timing_dir.exists():
        return out
    for f in sorted(timing_dir.glob("collect-it*.json")):
        with open(f) as fp:
            rec = json.load(fp)
        out[int(rec["iteration"])] = rec
    return out


def plot_progress(exp_dir: Path) -> None:
    """Per-iteration win rate vs the reigning champion (both colors).

    Win rates come from `league_state.json["by_iter"]`. Iter 0 has no gauntlet
    games (it bootstraps the league), so it shows up as a missing point.
    The 0.55 promotion threshold is drawn as a dashed horizontal line; bars
    that exceed it on BOTH sides correspond to iterations marked
    `promoted: True` in the league state.
    """
    state = _load_league(exp_dir)
    if not state or not state.get("by_iter"):
        print("No league_state.json yet — run a collect+update_league iteration first")
        return

    by_iter = state["by_iter"]
    iters = sorted(int(k) for k in by_iter)
    as_b = [by_iter[str(i)].get("as_black_wr") for i in iters]
    as_w = [by_iter[str(i)].get("as_white_wr") for i in iters]
    promoted = [bool(by_iter[str(i)].get("promoted")) for i in iters]
    chal_b = [by_iter[str(i)].get("challenged_black") for i in iters]
    chal_w = [by_iter[str(i)].get("challenged_white") for i in iters]

    xs = np.arange(len(iters))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(f"{EXP_NAME}: iter vs reigning champion", fontweight="bold")

    def _to_pct(vals):
        return [(v * 100) if v is not None else np.nan for v in vals]

    bars_b = ax.bar(xs - w/2, _to_pct(as_b), w, label="as black vs best_white",
                    color="black", alpha=0.75)
    bars_w = ax.bar(xs + w/2, _to_pct(as_w), w, label="as white vs best_black",
                    color="tab:blue", alpha=0.75)

    # Annotate each bar with the win rate and the opponent it played.
    for i, (bb, bw) in enumerate(zip(bars_b, bars_w)):
        if as_b[i] is not None:
            ax.text(bb.get_x() + bb.get_width() / 2, as_b[i] * 100,
                    f"{as_b[i]*100:.0f}%\nvs it{chal_w[i]}",
                    ha="center", va="bottom", fontsize=8, color="black")
        if as_w[i] is not None:
            ax.text(bw.get_x() + bw.get_width() / 2, as_w[i] * 100,
                    f"{as_w[i]*100:.0f}%\nvs it{chal_b[i]}",
                    ha="center", va="bottom", fontsize=8, color="tab:blue")

    ax.axhline(PROMOTION_THRESHOLD * 100, ls="--", color="tab:red", lw=1,
               label=f"promotion threshold ({PROMOTION_THRESHOLD:.0%})")

    # Mark promoted iters with a star above the bars.
    y_top = 110
    for x, p in zip(xs, promoted):
        if p:
            ax.text(x, y_top, "★", ha="center", va="top",
                    fontsize=14, color="tab:green")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in iters])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("★ = promoted to champion (both win rates > 0.55)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = _figures_dir(exp_dir) / "progress.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def plot_league(exp_dir: Path) -> None:
    """Step plot: per-iteration league champion for black and white."""
    state = _load_league(exp_dir)
    if not state:
        print("No league_state.json yet — run a collect iteration first")
        return
    history = state.get("history", [])
    if not history:
        print("league_state.json has empty history")
        return
    iters = [h["iter"] for h in history]
    best_black = [h["best_black"] for h in history]
    best_white = [h["best_white"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(iters, best_black, "o-", where="post", color="black",
            label="best black-side checkpoint")
    ax.step(iters, best_white, "s--", where="post", color="tab:blue",
            label="best white-side checkpoint")
    ax.plot(iters, iters, ":", color="grey", alpha=0.5,
            label="y = x (always dethroned)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Champion checkpoint (iteration #)")
    ax.set_title(f"{EXP_NAME}: league champions over time")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_xticks(iters)
    ax.set_yticks(sorted(set(best_black) | set(best_white)))

    plt.tight_layout()
    out = _figures_dir(exp_dir) / "league.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def plot_wall_time(exp_dir: Path) -> None:
    """Grouped bar chart: collect + train wall-time per iter, in minutes."""
    collect_timing = _load_collect_timing(exp_dir)
    train_results = _load_train_results(exp_dir)
    iters = sorted(set(collect_timing) | set(train_results))
    if not iters:
        print("No timing data — run a collect iteration first")
        return

    collect_min = [collect_timing.get(i, {}).get("elapsed_seconds", 0) / 60.0 for i in iters]
    train_min = [train_results.get(i, {}).get("elapsed_seconds", 0) / 60.0 for i in iters]
    throughput = []
    for i in iters:
        c = collect_timing.get(i)
        if not c or c.get("elapsed_seconds", 0) <= 0:
            throughput.append(np.nan)
            continue
        throughput.append(int(c.get("total_games", 0)) / (c["elapsed_seconds"] / 60.0))

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
    ax2.set_title("Collect throughput (games / collect wall min)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = _figures_dir(exp_dir) / "iteration_timing.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def print_summary(exp_dir: Path) -> None:
    state = _load_league(exp_dir)
    if not state:
        print("No league_state.json yet")
        return
    train_results = _load_train_results(exp_dir)
    by_iter = state.get("by_iter", {})
    print(f"\n{'iter':>4} {'champ_b':>8} {'champ_w':>8} "
          f"{'as_b_wr':>9} {'as_w_wr':>9} {'promoted':>9} "
          f"{'pol_acc':>8} {'val_acc':>8}")
    print("-" * 80)

    def _pct(v):
        return f"{v*100:>8.1f}%" if v is not None else f"{'-':>9}"

    def _pa(v):
        return f"{v*100:>7.1f}%" if v is not None else f"{'-':>8}"

    for h in state.get("history", []):
        i = h["iter"]
        s = by_iter.get(str(i), {})
        tr = train_results.get(i, {})
        prom = "★" if s.get("promoted") else " "
        print(f"{i:>4} iter{h['best_black']:<4} iter{h['best_white']:<4} "
              f"{_pct(s.get('as_black_wr'))} {_pct(s.get('as_white_wr'))} "
              f"{prom:>9} "
              f"{_pa(tr.get('train_policy_acc'))} {_pa(tr.get('train_value_acc'))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", default=str(DEFAULT_EXP))
    p.add_argument("--data-dir", default=str(DEFAULT_DATA),
                   help="Where to look for eval-*-it{N}/ NPZ dirs.")
    p.add_argument("--mode", choices=["summary", "plot", "both"], default="both")
    args = p.parse_args()
    exp_dir = Path(args.exp_dir)
    data_dir = Path(args.data_dir)
    if args.mode in ("summary", "both"):
        print_summary(exp_dir)
    if args.mode in ("plot", "both"):
        plot_progress(exp_dir)
        plot_league(exp_dir)
        plot_wall_time(exp_dir)
        plot_holdout(exp_dir)
        plot_eval_extra(exp_dir, data_dir)
        plot_train_accuracy(exp_dir)
        plot_kl(exp_dir, data_dir)


if __name__ == "__main__":
    main()
