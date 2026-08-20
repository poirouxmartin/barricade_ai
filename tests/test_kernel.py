import random
import unittest

import numpy as np

from barricade.engine import kernel
from barricade.engine.bitboard import GOAL0, GOAL1, INF, build_masks, flood_dist, idx, wall_ok
from barricade.engine.numba_engine import KernelEngine
from barricade.game import Barricade

ZOB = kernel.make_zobrist()


def random_state(rng):
    g = Barricade()
    g.positions = [(rng.randrange(9), rng.randrange(9)), (rng.randrange(9), rng.randrange(9))]
    for _ in range(rng.randrange(8)):
        ori = rng.choice("HV")
        r, c = rng.randrange(8), rng.randrange(8)
        if g.wall_valid(ori, r, c):
            g.apply(("wall", (ori, r, c)))
    return g


def st_from(game):
    return kernel.make_state(game, ZOB)


def masks_of(st):
    (_, _, _, _, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo,
     vs_hi, vs_lo, _, _, _) = st
    return (int(hb_hi) << 64) | int(hb_lo), (int(vb_hi) << 64) | int(vb_lo), \
           (int(hs_hi) << 64) | int(hs_lo), (int(vs_hi) << 64) | int(vs_lo)


class TestKernelParity(unittest.TestCase):
    def test_flood_distance_matches(self):
        rng = random.Random(42)
        for _ in range(300):
            g = random_state(rng)
            st = st_from(g)
            hb_hi, hb_lo, vb_hi, vb_lo = st[4], st[5], st[6], st[7]
            for p, pos in ((0, st[0]), (1, st[1])):
                goal_hi, goal_lo = (kernel.GOAL0_HI, kernel.GOAL0_LO) if p == 0 else (kernel.GOAL1_HI, kernel.GOAL1_LO)
                kd = kernel.flood_dist(pos, hb_hi, hb_lo, vb_hi, vb_lo, goal_hi, goal_lo)
                self.assertEqual(kd, g.dist_to_goal(p))

    def test_movegen_matches(self):
        rng = random.Random(3)
        for _ in range(300):
            g = random_state(rng)
            st = st_from(g)
            pos0, pos1, _, _, hb_hi, hb_lo, vb_hi, vb_lo = st[:8]
            for player, pos, opp in ((0, pos0, pos1), (1, pos1, pos0)):
                out = np.zeros(16, np.int64)
                n = kernel.gen_moves(pos, opp, hb_hi, hb_lo, vb_hi, vb_lo, out)
                py = {r * 9 + c for r, c in g.legal_moves(player)}
                self.assertEqual(set(int(x) for x in out[:n]), py)

    def test_wall_validity_matches(self):
        rng = random.Random(7)
        for _ in range(300):
            g = random_state(rng)
            st = st_from(g)
            hb, vb, hs, vs = masks_of(st)
            i0, i1 = st[0], st[1]
            for ori in ("H", "V"):
                for r in range(8):
                    for c in range(8):
                        if (r + c) % 5:
                            continue
                        kor = 0 if ori == "H" else 1
                        kd = kernel.wall_ok(st, kor, r, c)
                        self.assertEqual(kd, g.wall_valid(ori, r, c))

    def test_gen_walls_are_legal(self):
        rng = random.Random(11)
        for _ in range(200):
            g = random_state(rng)
            st = st_from(g)
            pos0, pos1, wl0, wl1, *_ = st
            for player, my, opp, wl in ((0, pos0, pos1, wl0), (1, pos1, pos0, wl1)):
                out = np.zeros(32, np.int64)
                n = kernel.gen_walls(st, my, opp, out, kernel.WALL_CAP)
                for i in range(n):
                    action = int(out[i])
                    self.assertGreaterEqual(action, kernel.WALL_BASE)
                    w = action - kernel.WALL_BASE
                    ori, slot = divmod(w, 64)
                    r, c = divmod(slot, 8)
                    self.assertTrue(g.wall_valid("H" if ori == 0 else "V", r, c))

    def test_gen_walls_finds_something_when_possible(self):
        rng = random.Random(19)
        found = 0
        for _ in range(100):
            g = random_state(rng)
            st = st_from(g)
            pos0, pos1, wl0, wl1, *_ = st
            hb, vb, hs, vs = masks_of(st)
            for player, my, opp, wl in ((0, pos0, pos1, wl0), (1, pos1, pos0, wl1)):
                if wl == 0:
                    continue
                legal = [w for w in _legal_walls(g) if w is not None]
                if not legal:
                    continue
                out = np.zeros(32, np.int64)
                n = kernel.gen_walls(st, my, opp, out, kernel.WALL_CAP)
                if n > 0:
                    found += 1
        self.assertGreater(found, 30)


def _legal_walls(g):
    out = []
    for ori in ("H", "V"):
        for r in range(8):
            for c in range(8):
                if g.wall_valid(ori, r, c):
                    out.append((ori, r, c))
    return out


class TestKernelEngine(unittest.TestCase):
    def test_plays_full_game_legally(self):
        rng = random.Random(5)
        for _ in range(6):
            g = Barricade()
            e = KernelEngine(max_depth=3, time_limit=0.5)
            moves = 0
            while g.winner is None and moves < 200:
                action = e.choose_move(g)
                self.assertIn(action, g.legal_actions())
                g.apply(action)
                moves += 1
            self.assertIsNotNone(g.winner)

    def test_decision_matches_python_engine(self):
        from barricade.engine.alphabeta import AlphaBetaEngine
        rng = random.Random(23)
        agreed = total = 0
        for _ in range(12):
            g = random_state(rng)
            k = KernelEngine(max_depth=4, time_limit=5.0)
            a = AlphaBetaEngine(max_depth=4, time_limit=5.0)
            k_action = k.choose_move(g)
            a_action = a.choose_move(g)
            if k_action == a_action:
                agreed += 1
            total += 1
        self.assertGreaterEqual(agreed / total, 0.5)


class TestLearnedEval(unittest.TestCase):
    def test_nn_value_bounded(self):
        rng = random.Random(11)
        for _ in range(20):
            g = random_state(rng)
            st = st_from(g)
            v = float(kernel.nn_value(int(st[0]), int(st[1]), int(st[0]),
                                      int(st[1]), int(st[2]), int(st[3]),
                                      int(st[12])))
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_eval_unchanged_with_nn_off(self):
        # Production keeps NN_W = 0, so eval_fn must equal the handcrafted
        # score (matching evaluate.score, no detour/tempo terms); the learned
        # term must not leak into the search.
        self.assertEqual(int(kernel.NN_W), 0)
        rng = random.Random(7)
        for _ in range(20):
            g = random_state(rng)
            st = st_from(g)
            pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
            d0 = int(kernel.flood_dist(pos0, hb_hi, hb_lo, vb_hi, vb_lo,
                                       kernel.GOAL0_HI, kernel.GOAL0_LO))
            d1 = int(kernel.flood_dist(pos1, hb_hi, hb_lo, vb_hi, vb_lo,
                                       kernel.GOAL1_HI, kernel.GOAL1_LO))
            if turn == 0:
                my_d, opp_d, my_w, opp_w = d0, d1, wl0, wl1
            else:
                my_d, opp_d, my_w, opp_w = d1, d0, wl1, wl0
            dist_adv = (opp_d - my_d) * kernel.DIST_W
            if dist_adv > 0:
                conf = max(1.0 - kernel.CONF_W * opp_w, kernel.CONF_FLOOR)
                dist_adv = np.int64(dist_adv * conf)
            expected = int(dist_adv + (my_w - opp_w))
            self.assertEqual(int(kernel.eval_fn(st)), expected)


if __name__ == "__main__":
    unittest.main()