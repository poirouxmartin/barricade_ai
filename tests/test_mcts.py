import unittest

from barricade.game import Barricade

try:
    from barricade.engine.mcts import MctsEngine, mcts_run, seed_rng
    import numpy as np
    _HAS_MCTS = True
except ImportError:
    _HAS_MCTS = False


def make(pos0, pos1, h=(), v=(), walls_left=(10, 10)):
    g = Barricade()
    g.positions = [tuple(pos0), tuple(pos1)]
    g.h_walls = set(h)
    g.v_walls = set(v)
    g.walls_left = list(walls_left)
    return g


@unittest.skipUnless(_HAS_MCTS, "numba not available")
class TestMcts(unittest.TestCase):
    def setUp(self):
        seed_rng(7)

    def test_wins_in_one(self):
        g = make([1, 4], [5, 4])
        eng = MctsEngine(max_nodes=4000, time_limit=2.0)
        self.assertEqual(eng.choose_move(g), ("move", (0, 4)))

    def test_blocks_immediate_loss(self):
        # P1 at (7,4) can win next move; P0 must wall him off.
        g = make([3, 4], [7, 4])
        eng = MctsEngine(max_nodes=8000, time_limit=2.0)
        g.apply(eng.choose_move(g))
        self.assertNotIn((8, 4), g.legal_moves(1))

    def test_returns_legal_action(self):
        g = make([2, 4], [7, 4])
        eng = MctsEngine(max_nodes=2000, time_limit=2.0)
        action = eng.choose_move(g)
        self.assertIn(action, g.legal_actions())

    def test_full_game_finishes(self):
        g = Barricade()
        eng = MctsEngine(max_nodes=2000, time_limit=2.0)
        for _ in range(120):
            if g.winner is not None:
                break
            g.apply(eng.choose_move(g))
        self.assertIsNotNone(g.winner)

    def test_best_action_by_visits(self):
        # (7,4) is the clear advance from the start; it must be the most-visited.
        from barricade.engine import kernel
        g = Barricade()
        zob = kernel.make_zobrist(7)
        st = kernel.make_state(g, zob)
        maxn = 20000
        pool = np.zeros((maxn, 15), np.int64)
        parent = np.full(maxn, -1, np.int64)
        cs = np.zeros(maxn, np.int64)
        cc = np.zeros(maxn, np.int64)
        ca = np.full(maxn * 16, -1, np.int64)
        cv = np.zeros(maxn * 16, np.int64)
        vis = np.zeros(maxn, np.int64)
        vs = np.zeros(maxn, np.float64)
        nn = np.zeros(1, np.int64)
        nn[0] = 1
        nc = np.zeros(1, np.int64)
        mcts_run(st, 20000, pool, parent, cs, cc, ca, cv, vis, vs, nn, nc, zob)
        base = cs[0]
        best_v = -1
        best_a = None
        for k in range(cc[0]):
            c = ca[base + k]
            if c >= 0 and vis[c] > best_v:
                best_v = vis[c]
                best_a = cv[base + k]
        self.assertEqual(kernel.decode_action(best_a), ("move", (7, 4)))


if __name__ == "__main__":
    unittest.main()
