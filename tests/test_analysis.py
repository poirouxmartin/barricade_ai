import unittest
import time

from barricade.game import Barricade
from barricade.engine.analysis import AnalysisSession
from barricade.engine import kernel
from barricade.engine.mcts import EVAL_SCALE


class AnalysisSessionTest(unittest.TestCase):
    def _session(self, position=None):
        g = Barricade()
        if position:
            for m in position:
                g.apply(m, check=False)
        return AnalysisSession(g)

    def test_start_reaches_depth(self):
        s = self._session([("move", (7, 4)), ("move", (1, 4))])
        s.start()
        deadline = time.time() + 30
        st = s.state()
        while time.time() < deadline and st["depth"] < 3:
            time.sleep(0.2)
            st = s.state()
        s.stop()
        self.assertEqual(st["error"], None)
        self.assertGreaterEqual(st["depth"], 3)
        self.assertGreater(st["nodes"], 0)
        self.assertIsNotNone(st["best_move"])

    def test_top_moves_and_pv(self):
        s = self._session([("move", (7, 4)), ("move", (1, 4))])
        s.start()
        deadline = time.time() + 30
        st = s.state()
        while time.time() < deadline and st["depth"] < 4:
            time.sleep(0.2)
            st = s.state()
        s.stop()
        self.assertTrue(st["top_moves"])
        self.assertEqual(st["top_moves"][0]["move"], st["best_move"])
        scores = [t["score"] for t in st["top_moves"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(st["pv"])
        self.assertEqual(st["pv"][0], st["best_move"])
        self.assertGreaterEqual(st["win_chance"], 0.0)
        self.assertLessEqual(st["win_chance"], 1.0)

    def test_win_chance_bounds(self):
        s = self._session()
        for score, expected in ((100000, 1.0), (-100000, 0.0), (0, 0.5)):
            with self.subTest(score=score):
                w = s._win_chance(score)
                if expected == 1.0:
                    self.assertEqual(w, 1.0)
                elif expected == 0.0:
                    self.assertEqual(w, 0.0)
                else:
                    self.assertAlmostEqual(w, 0.5, places=1)

    def test_stop_sets_running_false(self):
        s = self._session()
        s.start()
        time.sleep(0.1)
        s.stop()
        self.assertFalse(s.running)
        self.assertFalse(s.state()["running"])

    def test_stop_during_deep_search_is_immediate(self):
        # the kernel must release the GIL (nogil) so stop() can interrupt the
        # in-flight search at the next node instead of running the budget out
        s = self._session([("move", (7, 4)), ("move", (1, 4))])
        s.start()
        deadline = time.time() + 60
        st = s.state()
        while time.time() < deadline and st["depth"] < 8:
            time.sleep(0.2)
            st = s.state()
        self.assertGreaterEqual(st["depth"], 8)
        t0 = time.time()
        s.stop()
        s._thread.join(timeout=2.0)
        self.assertFalse(s._thread.is_alive(), "analysis thread did not stop promptly")
        self.assertLess(time.time() - t0, 2.0)

    def test_mcts_analysis(self):
        g = Barricade()
        g.apply(("move", (7, 4)), check=False)
        g.apply(("move", (1, 4)), check=False)
        s = AnalysisSession(g, engine="mcts")
        s.start()
        deadline = time.time() + 30
        st = s.state()
        while time.time() < deadline and not st["best_move"]:
            time.sleep(0.2)
            st = s.state()
        s.stop()
        s._thread.join(timeout=2.0)
        self.assertFalse(s._thread.is_alive(), "mcts analysis thread did not stop promptly")
        self.assertEqual(st["error"], None)
        self.assertEqual(st["engine"], "mcts")
        self.assertIsNotNone(st["best_move"])
        self.assertGreater(st["nodes"], 0)
        self.assertGreater(st["nps"], 0)
        self.assertTrue(st["top_moves"])
        self.assertEqual(st["top_moves"][0]["move"], st["best_move"])
        self.assertGreaterEqual(st["score"], -EVAL_SCALE)
        self.assertLessEqual(st["score"], EVAL_SCALE)

    def test_unknown_engine_rejected(self):
        g = Barricade()
        with self.assertRaises(ValueError):
            AnalysisSession(g, engine="quantum")

    def test_pv_legality(self):
        g = Barricade()
        g.apply(("move", (7, 4)), check=False)
        g.apply(("move", (1, 4)), check=False)
        s = AnalysisSession(g)
        s.start()
        deadline = time.time() + 30
        st = s.state()
        while time.time() < deadline and st["depth"] < 4:
            time.sleep(0.2)
            st = s.state()
        s.stop()
        clone = g.clone()
        for m in st["pv"]:
            self.assertIn(m, clone.legal_actions(), f"PV move {m} illegal")
            clone.apply(m, check=False)
            if clone.winner is not None:
                break


if __name__ == "__main__":
    unittest.main()