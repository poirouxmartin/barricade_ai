"""Generate a self-play position dataset for the learned eval.

Plays a mix of games (kernel vs kernel at different strengths, kernel vs
greedy, roles alternated) and records, for every position, the 11 feature row
(`kernel.nn_features_row`) and the game outcome from P1's perspective:
+1 P1 wins, -1 P2 wins, 0 draw/lockup.

Usage:
    python tools/selfplay.py --games 100 --out data/positions.npz
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barricade.engine import kernel
from barricade.engine.greedy import GreedyEngine
from barricade.engine.numba_engine import KernelEngine
from barricade.game import Barricade

LOCKUP_PLIES = 120
MAX_POS_PER_GAME = 40  # cap so long games don't dominate the training set


def make_kernel(depth, time_limit):
    return KernelEngine(max_depth=depth, time_limit=time_limit, tt_size=1 << 18)


def play_game(engines, zob):
    """One game; returns (features Nx11, labels N,) or None."""
    g = Barricade()
    rows = [kernel.nn_features_row(g, zob)]  # include the true start position
    for ply in range(LOCKUP_PLIES * 2):
        if g.winner is not None:
            break
        eng = engines[0 if g.turn == 0 else 1]
        action = eng.choose_move(g)
        if action is None:
            g.winner = 1 - g.turn
            g.game_over_reason = "resign"
            break
        try:
            g.apply(action)
        except ValueError:
            g.winner = 1 - g.turn
            g.game_over_reason = "illegal"
            break
        rows.append(kernel.nn_features_row(g, zob))
    else:
        g.winner = None  # lockup
        g.game_over_reason = "lockup"
    if not rows:
        return None
    if len(rows) > MAX_POS_PER_GAME:
        step = len(rows) / MAX_POS_PER_GAME
        rows = [rows[int(i * step)] for i in range(MAX_POS_PER_GAME)]
    label = 0 if g.winner is None else (1 if g.winner == 0 else -1)
    return np.asarray(rows, np.float64), np.full(len(rows), label, np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--out", default="data/positions.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    zob = kernel.make_zobrist()
    pairs = [
        (make_kernel(4, 0.15), make_kernel(6, 0.3)),
        (make_kernel(6, 0.3), make_kernel(4, 0.15)),
        (make_kernel(5, 0.2), GreedyEngine()),
        (GreedyEngine(), make_kernel(5, 0.2)),
    ]
    feats_all = []
    labels_all = []
    # keep the game mix balanced: half the games won by P1, half by P2
    target = max(1, args.games // 2)
    won = [0, 0]
    n_rows = 0
    t0 = time.time()
    attempts = 0
    while won[0] < target or won[1] < target:
        a, b = random.choice(pairs)
        result = play_game([a, b], zob)
        attempts += 1
        if result is None:
            continue
        feats, labels = result
        winner = 1 if (labels[0] > 0) else (0 if labels[0] < 0 else None)
        if winner is None or won[winner] >= target:
            continue
        won[winner] += 1
        feats_all.append(feats)
        labels_all.append(labels)
        n_rows += len(labels)
        if attempts % 40 == 0:
            print(f"  {won[0] + won[1]}/{target * 2} balanced games "
                  f"(P1 {won[0]}, P2 {won[1]}), {n_rows} positions, "
                  f"{attempts} attempts ({time.time() - t0:.0f}s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    X = np.concatenate(feats_all)
    y = np.concatenate(labels_all)
    np.savez(out, X=X, y=y)
    print(f"saved {n_rows} positions from {target * 2} balanced games "
          f"(P1 {won[0]}, P2 {won[1]}, {attempts} attempts) -> {out}")
    print(f"label balance: +1 {int((y > 0).sum())} / -1 {int((y < 0).sum())} "
          f"/ 0 {int((y == 0).sum())}")


if __name__ == "__main__":
    main()