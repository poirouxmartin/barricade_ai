import unittest

from barricade.engine.alphabeta import AlphaBetaEngine
from barricade.engine.evaluate import evaluate
from barricade.game import Barricade


def make(pos0, pos1, h=(), v=(), walls_left=(10, 10)):
    g = Barricade()
    g.positions = [tuple(pos0), tuple(pos1)]
    g.h_walls = set(h)
    g.v_walls = set(v)
    g.walls_left = list(walls_left)
    return g


class TestEvaluate(unittest.TestCase):
    def test_prefers_closer(self):
        gA = make([2, 4], [6, 4], walls_left=(0, 0))
        gB = make([1, 4], [6, 4], walls_left=(0, 0))
        self.assertGreater(evaluate(gB, 0), evaluate(gA, 0))

    def test_wall_advantage(self):
        # same distances, but more walls left for P0 -> higher score for P0
        g1 = make([2, 4], [5, 4], walls_left=(6, 6))
        g2 = make([2, 4], [5, 4], walls_left=(6, 1))
        self.assertGreater(evaluate(g2, 0), evaluate(g1, 0))

    def test_confidence_dampens_lead(self):
        # an ahead position is worth less when the opponent still has walls
        g1 = make([1, 4], [5, 4], walls_left=(0, 0))
        g2 = make([1, 4], [5, 4], walls_left=(0, 10))
        self.assertGreater(evaluate(g1, 0), evaluate(g2, 0))


class TestAlphaBeta(unittest.TestCase):
    def test_wins_in_one(self):
        g = make([1, 4], [8, 4])
        eng = AlphaBetaEngine(max_depth=2, time_limit=5.0)
        self.assertEqual(eng.choose_move(g), ("move", (0, 4)))

    def test_blocks_immediate_loss(self):
        # P1 at (7,4) can win next move by reaching row 8; P0 must wall him off.
        g = make([3, 4], [7, 4])
        eng = AlphaBetaEngine(max_depth=3, time_limit=5.0)
        g.apply(eng.choose_move(g))
        self.assertNotIn((8, 4), g.legal_moves(1))

    def test_returns_legal_action(self):
        g = make([2, 4], [7, 4])
        eng = AlphaBetaEngine(max_depth=2, time_limit=5.0)
        action = eng.choose_move(g)
        self.assertIn(action, g.legal_actions())

    def test_full_game_finishes(self):
        g = Barricade()
        eng = AlphaBetaEngine(max_depth=1, time_limit=5.0)
        for _ in range(100):
            if g.winner is not None:
                break
            g.apply(eng.choose_move(g))
        self.assertIsNotNone(g.winner)


if __name__ == "__main__":
    unittest.main()