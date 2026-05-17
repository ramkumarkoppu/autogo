"""Claude-as-a-Go-player agent.

Shells out to the `claude` CLI in --print mode for every move. Routes through
the user's existing Claude Code OAuth, so usage is billed against the
Claude Agent SDK monthly credit (Max 5x plan) rather than a separate API key.

Requires the `claude` CLI installed and OAuth-logged-in on the host running
self_play / play.py (i.e. in WSL Ubuntu in our setup).

Caveat: Claude is a language model with no Go-specific training. Expect it
to play badly against any real Go engine. Useful for entertainment, not
benchmarking.
"""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Optional

import alpha_go_cpp
import numpy as np

from alpha_go.agents.base import Agent, PASS, register_agent

# Go column letters skip 'I' by convention (looks like 1).
_COLS = "ABCDEFGHJKLMNOPQRST"


def _idx_to_label(row: int, col: int, board_size: int) -> str:
    """(row=0..n-1 top→bottom, col=0..n-1 left→right) → e.g. 'E5'."""
    return f"{_COLS[col]}{board_size - row}"


def _label_to_idx(label: str, board_size: int) -> Optional[tuple[int, int]]:
    """'E5' → (row, col) in internal indexing. Returns None on parse failure."""
    label = label.strip().upper()
    if len(label) < 2:
        return None
    col_letter, row_str = label[0], label[1:]
    if col_letter not in _COLS[:board_size]:
        return None
    if not row_str.isdigit():
        return None
    row_num = int(row_str)
    if not (1 <= row_num <= board_size):
        return None
    col = _COLS.index(col_letter)
    row = board_size - row_num
    return (row, col)


def _board_ascii(board: alpha_go_cpp.GoBoard) -> str:
    """Render board as readable ASCII grid. Current player = X, opponent = O."""
    size = board.size()
    arr = board.to_numpy().astype(int)
    me = 1 if board.to_play() == alpha_go_cpp.GoBoard.BLACK else 2
    them = 2 if me == 1 else 1

    lines = ["   " + " ".join(_COLS[c] for c in range(size))]
    for r in range(size):
        row_label = f"{size - r:2d} "
        cells = []
        for c in range(size):
            v = int(arr[r, c])
            if v == me:
                cells.append("X")
            elif v == them:
                cells.append("O")
            else:
                cells.append(".")
        lines.append(row_label + " ".join(cells))
    return "\n".join(lines)


def _compute_groups(
    board_np: np.ndarray,
) -> list[tuple[int, set[tuple[int, int]], set[tuple[int, int]]]]:
    """Flood-fill to find each connected group and its liberties.
    Returns list of (color, stones, liberties) per group."""
    size = board_np.shape[0]
    visited = np.zeros_like(board_np, dtype=bool)
    groups: list[tuple[int, set[tuple[int, int]], set[tuple[int, int]]]] = []
    for r0 in range(size):
        for c0 in range(size):
            if board_np[r0, c0] == 0 or visited[r0, c0]:
                continue
            color = int(board_np[r0, c0])
            stones: set[tuple[int, int]] = set()
            liberties: set[tuple[int, int]] = set()
            stack = [(r0, c0)]
            while stack:
                r, c = stack.pop()
                if (r, c) in stones:
                    continue
                stones.add((r, c))
                visited[r, c] = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < size and 0 <= nc < size):
                        continue
                    if board_np[nr, nc] == 0:
                        liberties.add((nr, nc))
                    elif int(board_np[nr, nc]) == color:
                        if (nr, nc) not in stones:
                            stack.append((nr, nc))
            groups.append((color, stones, liberties))
    return groups


def _group_summary(board: alpha_go_cpp.GoBoard) -> str:
    """Per-group liberty report. Black = X, White = O."""
    board_np = board.to_numpy().astype(int)
    size = board.size()
    me = 1 if board.to_play() == alpha_go_cpp.GoBoard.BLACK else 2

    def label(color: int) -> str:
        if color == me:
            return "X (you)"
        return "O (opponent)"

    lines = []
    groups = _compute_groups(board_np)
    # Sort: opponent-in-atari first (capturable!), then your own in atari/danger,
    # then everything else.
    def priority(g: tuple[int, set[tuple[int, int]], set[tuple[int, int]]]) -> tuple[int, int]:
        color, stones, libs = g
        lib_count = len(libs)
        # Capturable opponent groups float to top, then your own danger, then rest.
        if color != me and lib_count == 1:
            return (0, -len(stones))
        if color == me and lib_count <= 2:
            return (1, lib_count)
        return (2, lib_count)

    for color, stones, libs in sorted(groups, key=priority):
        stones_str = ",".join(_idx_to_label(r, c, size) for r, c in sorted(stones))
        libs_str = ",".join(_idx_to_label(r, c, size) for r, c in sorted(libs)) if libs else "(captured!)"
        tag = "  IN ATARI" if len(libs) == 1 else ("  low-liberty" if len(libs) == 2 else "")
        lines.append(
            f"  {label(color)} group {{{stones_str}}}: {len(libs)} liberties ({libs_str}){tag}"
        )
    return "\n".join(lines) if lines else "  (board is empty)"


def _build_prompt(
    board: alpha_go_cpp.GoBoard, history: list[str], move_no: int
) -> str:
    size = board.size()
    me_color = "Black" if board.to_play() == alpha_go_cpp.GoBoard.BLACK else "White"
    history_str = ", ".join(history) if history else "(none — this is move 1)"

    return (
        f"You are playing Go on a {size}x{size} board as {me_color}.\n"
        f"On the board: X = your stones, O = opponent's stones.\n"
        f"Komi: 7.5. Rules: Tromp-Taylor (so leave no dead stones inside opponent territory).\n\n"
        f"Move history (alternating B,W): {history_str}\n\n"
        f"Current position (move {move_no} about to be played by you):\n"
        f"{_board_ascii(board)}\n\n"
        f"Group/liberty status (pre-computed so you don't have to count manually):\n"
        f"{_group_summary(board)}\n\n"
        f"Think carefully before playing. Read tactical sequences explicitly: "
        f"'if I play X then opponent plays Y, then I play Z, then …'. Check:\n"
        f"  1. Is any opponent group in atari? Capturing is usually huge.\n"
        f"  2. Are any of YOUR groups in atari or low-liberty? Save them or sacrifice deliberately.\n"
        f"  3. Life and death: every group needs 2 real eyes to live. Watch for eye-stealing moves.\n"
        f"  4. Ladders and nets: would playing X start a sequence that captures opponent stones?\n"
        f"  5. Don't fill your own eyes or play self-atari.\n"
        f"  6. Endgame: prefer sente moves (ones the opponent must answer) before gote.\n"
        f"  7. On 9x9 with 7.5 komi, Black needs ~46 points to win; White needs ~36.\n\n"
        f"After reasoning, the LAST LINE of your reply must contain ONLY the move coordinate "
        f"in the form 'E5' (column letter A-J skipping I, then row digit 1-{size}), or 'pass'. "
        f"Everything before the last line is ignored by the parser."
    )


_MOVE_RE = re.compile(r"\b([A-HJ-T])\s*([1-9]|1[0-9])\b", re.IGNORECASE)


def _parse_move(response: str, board_size: int) -> tuple[int, int] | None:
    """Find a move in Claude's reply, preferring the LAST line of the response."""
    text = response.strip()
    # Try last non-empty line first — that's where we asked Claude to put the answer.
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if re.search(r"\bpass\b", line, re.IGNORECASE):
            return PASS
        matches = list(_MOVE_RE.finditer(line))
        if matches:
            label = matches[-1].group(0).replace(" ", "")
            idx = _label_to_idx(label, board_size)
            if idx is not None:
                return idx
    # Fall back to scanning the whole response if nothing parsed from last line.
    if re.search(r"\bpass\b", text, re.IGNORECASE):
        return PASS
    for match in _MOVE_RE.finditer(text):
        label = match.group(0).replace(" ", "")
        idx = _label_to_idx(label, board_size)
        if idx is not None:
            return idx
    return None


@register_agent("claude-opus-xhigh")
class ClaudeOpusXHighAgent(Agent):
    """Opus 4.7 at xhigh effort, one `claude -p` call per move.
    ~30-90 s per move (extended thinking). Prompt includes group/liberty
    analysis + move history. Falls back to a near-center legal move only
    if Claude's reply doesn't parse."""

    def __init__(self) -> None:
        self.model = "opus"
        self.timeout_s = 300  # xhigh effort can take 60-180s per move
        self._last_response: str | None = None
        self._history: list[str] = []
        self._move_no = 0
        self._board_size = 9

    def start_game(self, board_size: int) -> None:
        self._history = []
        self._move_no = 0
        self._board_size = board_size

    def notify_move(self, row: int, col: int) -> None:
        if row == -1 and col == -1:
            self._history.append("pass")
        else:
            self._history.append(_idx_to_label(row, col, self._board_size))
        self._move_no += 1

    def select_move(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]:
        size = board.size()
        prompt = _build_prompt(board, self._history, self._move_no + 1)

        result = subprocess.run(
            [
                "claude", "-p",
                "--model", self.model,
                "--effort", "xhigh",
                "--no-session-persistence",
                "--max-budget-usd", "1.0",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        self._last_response = result.stdout

        # Stderr-log so play.py background log shows what Claude said.
        snippet = (result.stdout or "").strip().replace("\n", " | ")[:200]
        print(f"[claude-opus] move {self._move_no + 1} response: {snippet!r}",
              file=sys.stderr, flush=True)
        if result.returncode != 0:
            print(f"[claude-opus] subprocess rc={result.returncode}, stderr={result.stderr[:200]!r}",
                  file=sys.stderr, flush=True)

        move = _parse_move(result.stdout, size)

        legal_flat = set(board.get_legal_moves_flat())
        if move == PASS:
            return PASS
        if move is not None:
            row, col = move
            flat = row * size + col
            if flat in legal_flat:
                return move
            print(f"[claude-opus] move {self._move_no + 1}: parsed {move} but ILLEGAL — falling back",
                  file=sys.stderr, flush=True)

        # Fallback: pick a near-center legal move, not just A9, so it's clear
        # in the resulting game that Claude wasn't parsed.
        if not legal_flat:
            return PASS
        cx = cy = size // 2
        best = min(legal_flat, key=lambda f: abs(f // size - cy) + abs(f % size - cx))
        print(f"[claude-opus] move {self._move_no + 1}: FALLBACK firing",
              file=sys.stderr, flush=True)
        return (best // size, best % size)
