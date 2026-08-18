import unittest

from barricade.game import Barricade


def make(pos0, pos1, h=(), v=(), walls_left=(10, 10), turn=0):
    g = Barricade()
    g.positions = [tuple(pos0), tuple(pos1)]
    g.h_walls = set(h)
    g.v_walls = set(v)
    g.walls_left = list(walls_left)
    g.turn = turn
    return g


class TestSetup(unittest.TestCase):
    def test_initial(self):
        g = Barricade()
        self.assertEqual(g.positions, [(8, 4), (0, 4)])
        self.assertEqual(g.walls_left, [10, 10])
        self.assertEqual(g.turn, 0)
        self.assertIsNone(g.winner)
        self.assertIn(("move", (7, 4)), g.legal_actions())

    def test_win_by_reaching_goal(self):
        g = make([1, 4], [8, 4])
        g.apply(("move", (0, 4)))
        self.assertEqual(g.winner, 0)
        self.assertEqual(g.legal_actions(), [])


class TestMovement(unittest.TestCase):
    def test_basic_move(self):
        g = Barricade()
        g.apply(("move", (7, 4)))
        self.assertEqual(g.positions[0], (7, 4))
        self.assertEqual(g.turn, 1)

    def test_illegal_move_raises(self):
        g = make([5, 4], [8, 4])
        with self.assertRaises(ValueError):
            g.apply(("move", (6, 5)))  # diagonal is not a legal move

    def test_wall_blocks_move(self):
        # H wall at (6,4) blocks vertical edge between (6,4) and (7,4)
        g = make([6, 4], [8, 4], h=[(6, 4)])
        self.assertNotIn((7, 4), g.legal_moves(0))
        self.assertIn((5, 4), g.legal_moves(0))

    def test_jump_straight(self):
        g = make([5, 4], [6, 4])
        moves = g.legal_moves(0)
        self.assertIn((7, 4), moves)  # jump straight over opponent
        self.assertNotIn((6, 4), moves)

    def test_jump_diagonal_when_wall_behind(self):
        # H wall at (6,4) blocks the square behind opponent
        g = make([5, 4], [6, 4], h=[(6, 4)])
        moves = g.legal_moves(0)
        self.assertNotIn((7, 4), moves)
        self.assertIn((6, 3), moves)
        self.assertIn((6, 5), moves)

    def test_jump_diagonal_when_edge_behind(self):
        g = make([7, 4], [8, 4])
        moves = g.legal_moves(0)
        self.assertNotIn((9, 4), moves)
        self.assertIn((8, 3), moves)
        self.assertIn((8, 5), moves)

    def test_no_jump_when_wall_between(self):
        g = make([5, 4], [6, 4], h=[(5, 4)])
        moves = g.legal_moves(0)
        self.assertNotIn((6, 4), moves)
        self.assertNotIn((7, 4), moves)
        self.assertNotIn((6, 3), moves)


class TestWalls(unittest.TestCase):
    def test_place_wall(self):
        g = Barricade()
        g.apply(("wall", ("H", 3, 3)))
        self.assertEqual(g.walls_left[0], 9)
        self.assertIn((3, 3), g.h_walls)
        self.assertEqual(g.turn, 1)

    def test_overlap_rejected(self):
        g = make([8, 4], [0, 4], h=[(3, 3)])
        self.assertFalse(g.wall_valid("H", 3, 3))
        self.assertFalse(g.wall_valid("H", 3, 2))  # shares edge
        self.assertTrue(g.wall_valid("H", 3, 1))

    def test_cross_rejected(self):
        g = make([8, 4], [0, 4], v=[(3, 3)])
        self.assertFalse(g.wall_valid("H", 3, 3))
        self.assertTrue(g.wall_valid("H", 4, 3))  # T-junction is legal

    def test_path_cutoff_rejected(self):
        # Seal the top player: all down-edges at row boundary 0 blocked, col 8
        # unreachable. Placing V(0,7) must be rejected.
        g = make([8, 4], [0, 4], h=[(0, 0), (0, 2), (0, 4), (0, 6)])
        self.assertTrue(g.has_path(1))             # gap at col 8 keeps a route
        self.assertFalse(g.wall_valid("V", 0, 7))  # placing it would seal player 1
        g2 = make([8, 4], [0, 4], h=[(0, 0), (0, 2), (0, 4), (0, 6)], v=[(0, 7)])
        self.assertFalse(g2.has_path(1))  # top player is now sealed in row 0
        self.assertTrue(g2.has_path(0))   # bottom player unaffected
        self.assertTrue(g.wall_valid("H", 1, 7))  # a non-sealing wall stays legal

    def test_walls_exhausted(self):
        g = make([8, 4], [0, 4], walls_left=(0, 10))
        self.assertEqual(g.legal_walls(0), [])
        with self.assertRaises(ValueError):
            g.apply(("wall", ("H", 3, 3)))


class TestClock(unittest.TestCase):
    def test_clock_initialized(self):
        g = Barricade(time_control=60)
        self.assertEqual(g.time_left, [60, 60])
        g2 = Barricade()
        self.assertIsNone(g2.time_left)

    def test_clock_charged_and_turn_switches(self):
        g = Barricade(time_control=60)
        g.apply(("move", (7, 4)))  # first move: clock starts, no charge
        self.assertEqual(g.turn, 1)
        self.assertEqual(g.time_left[0], 60)
        g.apply(("move", (1, 4)))  # player 1 charges elapsed time
        self.assertLessEqual(g.time_left[1], 60)
        self.assertEqual(g.turn, 0)
        self.assertIsNotNone(g._last_move_at)

    def test_timeout_loses(self):
        g = Barricade(time_control=60)
        g.apply(("move", (7, 4)))  # player 0 moves, clock started
        g.time_left[1] = 0.0
        g._tick_clock(1)  # any elapsed time makes player 1 go negative
        self.assertEqual(g.winner, 0)
        self.assertEqual(g.game_over_reason, "time")

    def test_clone_preserves_clock(self):
        g = Barricade(time_control=60)
        g.apply(("move", (7, 4)))
        c = g.clone()
        self.assertEqual(c.time_control, 60)
        self.assertAlmostEqual(c.time_left[1], g.time_left[1], places=6)
        self.assertEqual(c.time_left[0], 60)


if __name__ == "__main__":
    unittest.main()