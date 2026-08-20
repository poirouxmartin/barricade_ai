"""Generate a policy dataset: (features, kernel's chosen action) per position.

Same game mix as selfplay.py, but records for every position the 292-feature
board row (`kernel.nn_policy_features_row`) and the action the engine actually
played, encoded as an int in [0, 208] (0..80 moves, 81..208 walls). The MCTS
policy net learns these as move priors via behavior cloning.

Usage:
    python tools/selfplay_policy.py --games 150 --out data/policy.npz
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
MAX_POS_PER_GAME = 40


def make_kernel(depth, time_limit):
    return KernelEngine(max_depth=depth, time_limit=time_limit, tt_size=1 << 18)


def play_game(engines, zob):
    """One game; returns (features Nx292, actions N,) or None."""
    g = Barricade()
    rows = []
    actions = []
    for _ in range(LOCKUP_PLIES * 2):
        if g.winner is not None:
            break
        rows.append(kernel.nn_policy_features_row(g, zob))
        eng = engines[0 if g.turn == 0 else 1]
        action = eng.choose_move(g)
        if action is None:
            break
        try:
            actions.append(kernel.encode_action(g, action))
            g.apply(action)
        except ValueError:
            break
    if not rows or len(rows) != len(actions):
        return None
    if len(rows) > MAX_POS_PER_GAME:
        step = len(rows) / MAX_POS_PER_GAME
        idx = [int(i * step) for i in range(MAX_POS_PER_GAME)]
        rows = [rows[i] for i in idx]
        actions = [actions[i] for i in idx]
    return np.asarray(rows, np.float64), np.asarray(actions, np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--out", default="data/policy.npz")
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
    acts_all = []
    n_rows = 0
    t0 = time.time()
    for i in range(args.games):
        a, b = random.choice(pairs)
        result = play_game([a, b], zob)
        if result is None:
            continue
        feats, acts = result
        feats_all.append(feats)
        acts_all.append(acts)
        n_rows += len(acts)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.games} games, {n_rows} positions "
                  f"({time.time() - t0:.0f}s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    X = np.concatenate(feats_all)
    A = np.concatenate(acts_all)
    np.savez(out, X=X, A=A)
    n_walls = int((A >= kernel.WALL_BASE).sum())
    print(f"saved {n_rows} positions from {len(feats_all)} games -> {out}")
    print(f"action balance: moves {n_rows - n_walls} / walls {n_walls} "
          f"({100.0 * n_walls / n_rows:.1f}% walls)")


if __name__ == "__main__":
    main()