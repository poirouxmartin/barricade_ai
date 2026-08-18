"""KernelEngine: numba-compiled alpha-beta search.

Drop-in replacement for `AlphaBetaEngine` with the same interface. The search
runs entirely inside numba (`barricade.engine.kernel`); Python only drives
iterative deepening one depth at a time so the time budget can be enforced
with a clock (unavailable in nopython mode).
"""

import random
import time

import numpy as np

from barricade.engine.base import Engine

try:
    from barricade.engine import kernel
    _HAS_NUMBA = True
except ImportError:
    kernel = None
    _HAS_NUMBA = False


class KernelEngine(Engine):
    def __init__(self, max_depth=12, time_limit=2.0, tt_size=1 << 20):
        if not _HAS_NUMBA:
            raise ImportError("numba is required for KernelEngine")
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.tt_size = tt_size
        self.zob = kernel.make_zobrist()
        self.tt_k = np.zeros(tt_size, np.int64)
        self.tt_v = np.zeros(tt_size, np.int64)
        self.tt_d = np.zeros(tt_size, np.int64)
        self.tt_f = np.zeros(tt_size, np.int64)
        self.tt_m = np.zeros(tt_size, np.int64)
        self.budget = np.zeros(1, np.int64)
        self.last_info = {}
        self._warm = False
        self.nps_est = 300_000.0  # smoothed nodes/s across depths and moves

    def _warmup(self):
        if self._warm:
            return
        self._warm = True
        from barricade.game import Barricade
        st = kernel.make_state(Barricade(), self.zob)
        kernel.search_depth(st, 2, -1, 0, self.tt_k, self.tt_v, self.tt_d, self.tt_f,
                            self.tt_m, self.zob, self.budget)

    def choose_move(self, game):
        self._warmup()
        self.tt_k[:] = 0
        self.tt_v[:] = 0
        self.tt_d[:] = 0
        self.tt_f[:] = 0
        self.tt_m[:] = 0
        st = kernel.make_state(game, self.zob)
        deadline = time.time() + self.time_limit
        best = random.choice(game.legal_actions())
        hint = -1
        prev_score = 0
        total_nodes = 0
        for depth in range(1, self.max_depth + 1):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            allowance = int(remaining * self.nps_est) + 1000
            self.budget[0] = allowance
            t0 = time.time()
            action, score, aborted = kernel.search_depth(
                st, depth, hint, prev_score, self.tt_k, self.tt_v, self.tt_d, self.tt_f,
                self.tt_m, self.zob, self.budget)
            dt = max(time.time() - t0, 1e-6)
            used = max(0, allowance - int(self.budget[0]))
            total_nodes += used
            if used > 100 and dt > 0.03:
                instant = used / dt
                instant = min(max(instant, 50_000.0), 5_000_000.0)
                self.nps_est = 0.5 * self.nps_est + 0.5 * instant
            if aborted:
                break
            if action >= 0:
                best = kernel.decode_action(action)
                hint = action
                prev_score = int(score)
            self.last_info = {"depth": depth, "nodes": total_nodes,
                              "score": int(score)}
        return best