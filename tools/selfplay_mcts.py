"""AlphaZero-lite self-play: MCTS games with visit-distribution targets.

For every position the MCTS (with the current policy priors) searches and we
record the NN_IN-feature row, the root's legal-action mask M and visit counts
P (policy target, proportional after normalization), and the mover's identity.
At game end each row gets a soft outcome value (sign * tanh(DIST_W * loser_dist
/ EVAL_SCALE), the same anti-saturation scale as race_value). Games without a
winner (fortress lockups) are dropped: their outcome is noise and the policy
targets alone would not help the value head.

Default is pure MCTS vs MCTS self-play with exploration (temperature-annealed
visit sampling); --mix adds kernel and greedy opponents for cheap diversity.

Usage:
    python tools/selfplay_mcts.py --games 60 --time 0.3 --out data/az.npz
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
from barricade.engine.mcts import EVAL_SCALE, MctsEngine
from barricade.engine.numba_engine import KernelEngine
from barricade.game import Barricade

LOCKUP_PLIES = 120
MAX_POS_PER_GAME = 60
KEEP_LAST = 10
EXPLORE_TAU = 1.0   # sampling temperature early game
EXPLORE_TAU_END = 0.25


def terminal_mag(g, zob):
    """Soft magnitude sign * tanh(DIST_W * loser_dist / EVAL_SCALE) from the
    winner's perspective at game end (winner must be set)."""
    winner = g.winner
    loser = 1 - winner
    st = kernel.make_state(g, zob)
    pos = st[loser]
    if loser == 0:
        d = int(kernel.flood_dist(pos, st[4], st[5], st[6], st[7],
                                  kernel.GOAL0_HI, kernel.GOAL0_LO))
    else:
        d = int(kernel.flood_dist(pos, st[4], st[5], st[6], st[7],
                                  kernel.GOAL1_HI, kernel.GOAL1_LO))
    return np.tanh(kernel.DIST_W * d / EVAL_SCALE)


def play_game(engines, zob):
    """One game. Returns (X NxNN_IN, M Nx209, P Nx209, V N) or None."""
    g = Barricade()
    rows = []
    masks = []
    counts = []
    movers = []
    for _ in range(LOCKUP_PLIES * 2):
        if g.winner is not None:
            break
        rows.append(kernel.nn_policy_features_row(g, zob))
        movers.append(g.turn)
        eng = engines[0 if g.turn == 0 else 1]
        plies = len(g.history)
        if isinstance(eng, MctsEngine):
            tau = EXPLORE_TAU if plies < 2 * LOCKUP_PLIES // 3 else EXPLORE_TAU_END
            action = eng.choose_move(g, explore=tau)
            if action is None:
                return None
            acts, cnts = eng.root_policy()
            m = np.zeros(kernel.NN_POL, np.float64)
            p = np.zeros(kernel.NN_POL, np.float64)
            for a, c in zip(acts, cnts):
                if 0 <= int(a) < kernel.NN_POL:
                    m[int(a)] = 1.0
                    p[int(a)] = float(c)
        else:
            action = eng.choose_move(g)
            if action is None:
                return None
            m = np.zeros(kernel.NN_POL, np.float64)
            p = np.zeros(kernel.NN_POL, np.float64)
            try:
                for a in g.legal_actions():
                    e = kernel.encode_action(g, a)
                    m[e] = 1.0
                p[kernel.encode_action(g, action)] = 1.0
            except (ValueError, IndexError):
                return None
        masks.append(m)
        counts.append(p)
        try:
            g.apply(action)
        except ValueError:
            return None
    if g.winner is None:
        return None  # fortress lockup: no reliable outcome signal
    if not rows:
        return None
    mag = terminal_mag(g, zob)
    values = np.array([mag if m_ == g.winner else -mag for m_ in movers],
                      np.float64)
    if len(rows) > MAX_POS_PER_GAME:
        k = min(KEEP_LAST, len(rows))
        head = len(rows) - k
        step = head / (MAX_POS_PER_GAME - k)
        idx = ([int(i * step) for i in range(MAX_POS_PER_GAME - k)]
               + list(range(head, len(rows))))
        rows = [rows[i] for i in idx]
        masks = [masks[i] for i in idx]
        counts = [counts[i] for i in idx]
        values = values[idx]
    return (np.asarray(rows, np.float64), np.asarray(masks, np.float64),
            np.asarray(counts, np.float64), np.asarray(values, np.float64))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--time", type=float, default=0.3,
                    help="seconds per MCTS move during self-play")
    ap.add_argument("--mix", action="store_true",
                    help="also play kernel and greedy opponents for diversity")
    ap.add_argument("--out", default="data/az.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    zob = kernel.make_zobrist()
    pairs = [
        (MctsEngine(time_limit=args.time, seed=1),
         MctsEngine(time_limit=args.time, seed=2)),
    ]
    if args.mix:
        pairs += [
            (MctsEngine(time_limit=args.time, seed=3),
             KernelEngine(max_depth=10, time_limit=1.0, tt_size=1 << 18)),
            (KernelEngine(max_depth=10, time_limit=1.0, tt_size=1 << 18),
             MctsEngine(time_limit=args.time, seed=4)),
            (MctsEngine(time_limit=args.time, seed=5), GreedyEngine()),
            (GreedyEngine(), MctsEngine(time_limit=args.time, seed=6)),
        ]
    xs = []
    ms = []
    ps = []
    vs = []
    n_rows = 0
    n_dropped = 0
    t0 = time.time()
    for i in range(args.games):
        a, b = random.choice(pairs)
        result = play_game([a, b], zob)
        if result is None:
            n_dropped += 1
            continue
        X, M, P, V = result
        xs.append(X)
        ms.append(M)
        ps.append(P)
        vs.append(V)
        n_rows += len(V)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.games} games, {n_rows} positions "
                  f"({time.time() - t0:.0f}s)")
    if not xs:
        raise SystemExit("no usable games (all lockups?)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.concatenate(xs), M=np.concatenate(ms),
             P=np.concatenate(ps), V=np.concatenate(vs))
    V = np.concatenate(vs)
    print(f"saved {n_rows} positions from {len(xs)} games ({n_dropped} "
          f"dropped) -> {out}")
    n_walls = int((np.concatenate(ps)[:, kernel.WALL_BASE:].sum(axis=1) > 0).sum())
    print(f"rows with a wall in the target: {100.0 * n_walls / n_rows:.1f}%")
    print(f"value: mean {V.mean():+.3f}, |V|>0.5 on "
          f"{int((np.abs(V) > 0.5).sum())} rows")


if __name__ == "__main__":
    main()