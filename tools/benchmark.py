"""Benchmark harness for engine/evals.

Runs matches between engine configs and reports wins by role and by engine,
draws, resigns, average plies, and lockups. Each pairing is played in both
orders (P1/P2 swapped) to surface first-mover bias.

Usage:
    python tools/benchmark.py kernel tempo --games 8 --time 0.5 --depth 8
    python tools/benchmark.py --list
"""

import argparse
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barricade.engine.greedy import GreedyEngine
from barricade.engine.random import RandomEngine
from barricade.engine.numba_engine import KernelEngine
from barricade.engine.mcts import MctsEngine
from barricade.game import Barricade

LOCKUP_PLIES = 120


class VariantKernelEngine(KernelEngine):
    """KernelEngine bound to a generated eval-variant kernel module."""

    def __init__(self, variant, max_depth=12, time_limit=2.0, tt_size=1 << 20,
                 nn_w=None):
        from tools.make_variant import load

        super().__init__(max_depth=max_depth, time_limit=time_limit,
                         tt_size=tt_size)
        self.kernel = load(variant, nn_w=nn_w)
        self.zob = self.kernel.make_zobrist()


def _nn(nn_w):
    return lambda **kw: VariantKernelEngine("nn", nn_w=nn_w, **kw)


ENGINES = {
    "random": lambda **kw: RandomEngine(),
    "greedy": lambda **kw: GreedyEngine(),
    "kernel": lambda **kw: KernelEngine(**kw),
    "tempo": lambda **kw: VariantKernelEngine("tempo", **kw),
    "contact": lambda **kw: VariantKernelEngine("contact", **kw),
    "detour": lambda **kw: VariantKernelEngine("detour", **kw),
    "fix1c": lambda **kw: VariantKernelEngine("fix1c", **kw),
    "nn": _nn(3),
    "mcts": None,  # handled in make() (own construction)
}


def make(name, time_limit, depth):
    if name == "mcts":
        return MctsEngine(time_limit=time_limit)
    if name not in ENGINES:
        raise SystemExit(f"unknown engine {name!r}; use --list")
    kw = {"time_limit": time_limit, "max_depth": depth}
    return ENGINES[name](**kw)


def play(name_a, name_b, engine_a, engine_b, games, quiet=False):
    wins = {name_a: 0, name_b: 0}
    score = [0, 0]  # wins by role (P1, P2)
    resigns = 0
    plies_sum = 0
    lockups = 0
    for gi in range(games):
        if gi % 2 == 0:
            engines = [engine_a, engine_b]
            names = [name_a, name_b]
        else:
            engines = [engine_b, engine_a]
            names = [name_b, name_a]
        g = Barricade()
        for ply in range(LOCKUP_PLIES * 2):
            if g.winner is not None:
                break
            eng = engines[0 if g.turn == 0 else 1]
            try:
                action = eng.choose_move(g)
            except Exception:
                traceback.print_exc()
                g.winner = 1 - g.turn
                g.game_over_reason = "resign"
                break
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
        else:
            g.winner = None  # lockup
            g.game_over_reason = "lockup"
            lockups += 1
        plies_sum += len(g.history)
        if g.game_over_reason == "resign" or g.game_over_reason == "illegal":
            resigns += 1
        if g.winner is not None:
            wins[names[g.winner]] += 1
            score[g.winner] += 1
        if not quiet:
            w = "draw" if g.winner is None else f"P{g.winner + 1}"
            print(f"  game {gi + 1:2d}: {w} in {len(g.history)} plies "
                  f"({g.game_over_reason})")
    return {
        "games": games,
        "p1_wins": score[0],
        "p2_wins": score[1],
        "draws": games - score[0] - score[1],
        "resigns": resigns,
        "lockups": lockups,
        "avg_plies": plies_sum / games,
        "wins": wins,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engine_a", nargs="?", help="engine name (see --list)")
    ap.add_argument("engine_b", nargs="?", help="engine name (see --list)")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--time", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="list engines and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("engines:", ", ".join(sorted(ENGINES)))
        return

    t0 = time.time()
    res = play(
        args.engine_a, args.engine_b,
        make(args.engine_a, args.time, args.depth),
        make(args.engine_b, args.time, args.depth),
        args.games, quiet=args.quiet,
    )
    dt = time.time() - t0
    print(f"{args.engine_a} vs {args.engine_b}  ({args.games} games, "
          f"{args.time}s, depth {args.depth})")
    print(f"  P1 wins {res['p1_wins']} / P2 wins {res['p2_wins']} / "
          f"draws {res['draws']} / resigns {res['resigns']} / lockups {res['lockups']}")
    print(f"  engine wins: {args.engine_a} {res['wins'][args.engine_a]} - "
          f"{res['wins'][args.engine_b]} {args.engine_b}")
    print(f"  avg plies {res['avg_plies']:.1f}   wall-clock {dt:.1f}s")


if __name__ == "__main__":
    main()