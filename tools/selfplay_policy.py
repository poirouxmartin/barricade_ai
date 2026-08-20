"""Generate a policy + value dataset: (features, kernel's action, race value).

Same game mix as selfplay.py, but records for every position the NN_IN-feature
board row (`kernel.nn_policy_features_row`), the action the engine played
(encoded as an int in [0, 208]: 0..80 moves, 81..208 walls), and a value
target = the deterministic race projection (tools/mcts.race_value) from the
mover's perspective. The policy head learns the actions (behavior cloning) and
the value head distills the race lookahead, so the net can replace the MCTS
playouts.

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
from barricade.engine.mcts import race_value
from barricade.engine.numba_engine import KernelEngine
from barricade.game import Barricade

LOCKUP_PLIES = 120
MAX_POS_PER_GAME = 40
KEEP_LAST = 10  # conversion plies always kept (near-terminal moves are rare)


def make_kernel(depth, time_limit):
    return KernelEngine(max_depth=depth, time_limit=time_limit, tt_size=1 << 18)


def play_game(engines, zob):
    """One game; returns (features NxNN_IN, actions N, values N) or None."""
    g = Barricade()
    rows = []
    actions = []
    values = []
    for _ in range(LOCKUP_PLIES * 2):
        if g.winner is not None:
            break
        rows.append(kernel.nn_policy_features_row(g, zob))
        st = kernel.make_state(g, zob)
        values.append(float(race_value(st, zob)))
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
        k = min(KEEP_LAST, len(rows))
        head = len(rows) - k
        step = head / (MAX_POS_PER_GAME - k)
        idx = ([int(i * step) for i in range(MAX_POS_PER_GAME - k)]
               + list(range(head, len(rows))))
        rows = [rows[i] for i in idx]
        actions = [actions[i] for i in idx]
        values = [values[i] for i in idx]
    return (np.asarray(rows, np.float64), np.asarray(actions, np.int64),
            np.asarray(values, np.float64))


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
    vals_all = []
    n_rows = 0
    t0 = time.time()
    for i in range(args.games):
        a, b = random.choice(pairs)
        result = play_game([a, b], zob)
        if result is None:
            continue
        feats, acts, vals = result
        feats_all.append(feats)
        acts_all.append(acts)
        vals_all.append(vals)
        n_rows += len(acts)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.games} games, {n_rows} positions "
                  f"({time.time() - t0:.0f}s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    X = np.concatenate(feats_all)
    A = np.concatenate(acts_all)
    V = np.concatenate(vals_all)
    np.savez(out, X=X, A=A, V=V)
    n_walls = int((A >= kernel.WALL_BASE).sum())
    print(f"saved {n_rows} positions from {len(feats_all)} games -> {out}")
    print(f"action balance: moves {n_rows - n_walls} / walls {n_walls} "
          f"({100.0 * n_walls / n_rows:.1f}% walls)")
    print(f"value: mean {V.mean():+.3f}, |V|>0.5 on "
          f"{int((np.abs(V) > 0.5).sum())} rows")


if __name__ == "__main__":
    main()