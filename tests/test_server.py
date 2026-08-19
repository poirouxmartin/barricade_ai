import time
import unittest

from barricade.web.server import App


class AppTest(unittest.TestCase):
    def setUp(self):
        self.app = App()

    def test_history_and_snapshot(self):
        st, err = self.app.play(["move", [7, 4]])
        self.assertIsNone(err)
        st, err = self.app.play(["move", [1, 4]])
        self.assertIsNone(err)
        h = self.app.history()
        self.assertEqual(h["count"], 3)  # initial + 2 moves
        self.assertEqual(len(h["snapshots"]), 3)
        s0 = self.app.snapshot(0)
        self.assertEqual(s0["positions"], [[8, 4], [0, 4]])
        s2 = self.app.snapshot(2)
        self.assertEqual(s2["positions"], [[7, 4], [1, 4]])
        self.assertEqual(self.app.snapshot(99)["positions"], s2["positions"])
        self.assertEqual(self.app.snapshot(-5)["positions"], s0["positions"])

    def test_new_resets_history(self):
        self.app.play(["move", [7, 4]])
        self.app.new("ai2", engine="random")
        self.assertEqual(self.app.history()["count"], 1)

    def test_eval_info(self):
        ev = self.app.eval_info()
        self.assertIn("dist_gap", ev)
        self.assertIn("confidence", ev)
        self.assertEqual(ev["dist_gap"], 0)
        self.assertEqual(ev["score"], 0)
        self.assertEqual(ev["win_chance"], 0.5)
        self.assertEqual(len(ev["components"]), 4)
        self.assertEqual(ev["d0"], ev["d1"])
        self.assertEqual(ev["walls0"], 10)

    def test_eval_after_move_reflects_lead(self):
        # P1 moves one step toward goal: d0 goes 8 -> 7 -> lead +1 for P1
        self.app.play(["move", [7, 4]])
        ev = self.app.eval_info()
        self.assertEqual(ev["turn"], 1)  # P2 to move
        self.assertEqual(ev["dist_gap"], -1)  # from P2's perspective, P2 is behind

    def test_analysis_params(self):
        try:
            s = self.app.analysis_start(engine="kernel", max_depth=1)
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"analysis requires numba: {e}")
        if s.get("error"):
            self.skipTest("analysis requires numba")
        self.assertEqual(s["engine"], "kernel")
        self.app.analysis_stop()

    def test_analysis_state_has_eval_fields(self):
        try:
            s = self.app.analysis_start(engine="kernel", max_depth=1)
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"analysis requires numba: {e}")
        if s.get("error"):
            self.skipTest("analysis requires numba")
        self.assertIn("dist_gap", s)
        self.assertIn("confidence", s)
        self.app.analysis_stop()

    def test_match_random_random(self):
        st, err = self.app.start_match("random", "random", time_limit=0.05, games=2, depth=4)
        self.assertIsNone(err)
        self.assertIsNotNone(st)
        self.assertTrue(self.app.match_running())
        deadline = time.time() + 60
        while self.app.match_running() and time.time() < deadline:
            time.sleep(0.1)
        self.assertFalse(self.app.match_running())
        m = self.app.match_state()
        self.assertEqual(m["status"], "done")
        self.assertIsNone(m["error"])
        self.assertEqual(len(m["results"]), 2)
        self.assertEqual(m["score"][0] + m["score"][1] + m["draws"], 2)
        h = self.app.history()
        self.assertGreater(h["count"], 1, "last match game should be loaded for review")

    def test_match_unknown_engine(self):
        st, err = self.app.start_match("random", "quantum")
        self.assertIsNotNone(err)
        self.assertIsNone(st)

    def test_match_rejected_while_running(self):
        self.app.start_match("random", "random", time_limit=0.05, games=1)
        try:
            st, err = self.app.start_match("random", "random", time_limit=0.05, games=1)
            if self.app.match_running():
                self.assertIsNotNone(err)
            else:
                self.assertIsNone(err)
        finally:
            self.app.stop_match()


if __name__ == "__main__":
    unittest.main()