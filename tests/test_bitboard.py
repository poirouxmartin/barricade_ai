import random
import unittest

from barricade.engine.bitboard import build_masks, flood_dist, idx, wall_ok, GOAL0, GOAL1, INF
from barricade.game import Barricade


def random_state(rng):
    g = Barricade()
    g.positions = [(rng.randrange(9), rng.randrange(9)), (rng.randrange(9), rng.randrange(9))]
    for _ in range(rng.randrange(8)):
        ori = rng.choice("HV")
        r, c = rng.randrange(8), rng.randrange(8)
        if g.wall_valid(ori, r, c):
            g.apply(("wall", (ori, r, c)))
    return g


class TestBitboardParity(unittest.TestCase):
    def test_flood_distance_matches(self):
        rng = random.Random(42)
        for _ in range(400):
            g = random_state(rng)
            hb, vb, _, _ = build_masks(g.h_walls, g.v_walls)
            for p in (0, 1):
                i = idx(*g.positions[p])
                goal = GOAL0 if p == 0 else GOAL1
                self.assertEqual(flood_dist(i, hb, vb, goal), g.dist_to_goal(p))

    def test_wall_validity_matches(self):
        rng = random.Random(7)
        for _ in range(400):
            g = random_state(rng)
            hb, vb, hs, vs = build_masks(g.h_walls, g.v_walls)
            i0 = idx(*g.positions[0])
            i1 = idx(*g.positions[1])
            for ori in ("H", "V"):
                for r in range(8):
                    for c in range(8):
                        if (r + c) % 5:  # sample a subset to keep it fast
                            continue
                        self.assertEqual(wall_ok(hb, vb, hs, vs, ori, r, c, i0, i1),
                                         g.wall_valid(ori, r, c))


if __name__ == "__main__":
    unittest.main()