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


def _norm_state(st):
    """Force the mask fields to uint64. Crossing the Python/numba boundary
    boxes uint64 masks as Python ints; numba types them int64 if < 2**63 and
    uint64 otherwise, so a single int64 specialization then overflows on a
    bit-63 mask (walls H(7,7), V(7,7), H(7,0), V(7,0))."""
    return (st[0], st[1], st[2], st[3],
            np.uint64(st[4]), np.uint64(st[5]),
            np.uint64(st[6]), np.uint64(st[7]),
            np.uint64(st[8]), np.uint64(st[9]),
            np.uint64(st[10]), np.uint64(st[11]),
            st[12], st[13], st[14])


class KernelEngine(Engine):
    def __init__(self, max_depth=12, time_limit=2.0, tt_size=1 << 20):
        if not _HAS_NUMBA:
            raise ImportError("numba is required for KernelEngine")
        self.kernel = kernel
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.tt_size = tt_size
        self.zob = self.kernel.make_zobrist()
        self.tt_k = np.zeros(tt_size, np.int64)
        self.tt_v = np.zeros(tt_size, np.int64)
        self.tt_d = np.zeros(tt_size, np.int64)
        self.tt_f = np.zeros(tt_size, np.int64)
        self.tt_m = np.zeros(tt_size, np.int64)
        self.killers = np.zeros(2 * self.kernel.MAX_PLY, np.int64)
        self.budget = np.zeros(1, np.int64)
        self.stop = np.zeros(1, np.int64)
        self.out_moves = np.zeros(64, np.int64)
        self.out_scores = np.zeros(64, np.int64)
        self.last_info = {}
        self.progress = {}
        self._warm = False
        self.nps_est = 300_000.0  # smoothed nodes/s across depths and moves
        self.REP_WINDOW = 12  # plies; a key seen again inside this window is a repetition

    def _warmup(self):
        if self._warm:
            return
        self._warm = True
        from barricade.game import Barricade
        st = self.kernel.make_state(Barricade(), self.zob)
        self.kernel.search_depth(st, 2, -1, 0, self.tt_k, self.tt_v, self.tt_d,
                                 self.tt_f, self.tt_m, self.killers, self.zob,
                                 self.budget, self.stop, self.out_moves,
                                 self.out_scores, 0)

    def choose_move(self, game):
        self._warmup()
        self.tt_k[:] = 0
        self.tt_v[:] = 0
        self.tt_d[:] = 0
        self.tt_f[:] = 0
        self.tt_m[:] = 0
        self.killers[:] = 0
        st = self.kernel.make_state(game, self.zob)
        recent = self._recent_keys(game)
        deadline = time.time() + self.time_limit
        best = random.choice(game.legal_actions())
        hint = -1
        prev_score = 0
        total_nodes = 0
        best_action = -1
        best_depth = 0
        best_n = 0
        for depth in range(1, self.max_depth + 1):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            allowance = int(remaining * self.nps_est) + 1000
            self.budget[0] = allowance
            t0 = time.time()
            action, score, aborted, out_n = self.kernel.search_depth(
                st, depth, hint, prev_score, self.tt_k, self.tt_v, self.tt_d,
                self.tt_f, self.tt_m, self.killers, self.zob, self.budget,
                self.stop, self.out_moves, self.out_scores, 1)
            dt = max(time.time() - t0, 1e-6)
            used = max(0, allowance - int(self.budget[0]))
            total_nodes += used
            self.progress = {"depth": depth, "nodes": total_nodes, "score": int(score)}
            if used > 100 and dt > 0.03:
                instant = used / dt
                instant = min(max(instant, 50_000.0), 5_000_000.0)
                self.nps_est = 0.5 * self.nps_est + 0.5 * instant
            if aborted:
                break
            if action >= 0:
                best_action = action
                best_depth = depth
                best_n = out_n
                hint = action
                prev_score = int(score)
            self.last_info = {"depth": depth, "nodes": total_nodes,
                              "score": int(score)}
        if best_action >= 0:
            best = self._guard_repetition(st, best_action, best_depth, best_n, recent)
        return best

    def _recent_keys(self, game):
        """Zobrist keys of the last REP_WINDOW positions (current included),
        reconstructed by replaying game.history."""
        from barricade.game import Barricade

        k = self.kernel
        g = Barricade()
        st = k.make_state(g, self.zob)
        keys = []
        for kind, _player, arg in game.history:
            if kind == "move":
                st = _norm_state(k.apply_move(st, arg[0] * 9 + arg[1], self.zob))
            else:
                ori, r, c = arg
                ob = 0 if ori == "H" else 1
                st = _norm_state(k.apply_wall(st, k.WALL_BASE + ob * 64 + r * 8 + c, self.zob))
            keys.append(st[14])
        if len(keys) <= self.REP_WINDOW:
            return set(keys)
        return set(keys[-self.REP_WINDOW:])

    def _repeats(self, st, action, recent):
        k = self.kernel
        st = _norm_state(st)
        if action < k.WALL_BASE:
            child = k.apply_move(st, action, self.zob)
        else:
            child = k.apply_wall(st, action, self.zob)
        return child[14] in recent

    def _guard_repetition(self, st, action, depth, n_total, recent):
        """If the chosen move would repeat a recent position, re-pick the best
        ranked root move that does not repeat. Prevents shuffle lockups."""
        k = self.kernel
        if not self._repeats(st, action, recent):
            return k.decode_action(action)
        n = k.root_top_moves(st, depth, self.tt_k, self.tt_v, self.tt_d,
                             self.tt_f, self.tt_m, self.killers, self.zob,
                             self.budget, self.stop, self.out_moves,
                             self.out_scores, n_total, min(n_total, 8))
        for i in range(n):
            a = self.out_moves[i]
            if a >= 0 and not self._repeats(st, a, recent):
                return k.decode_action(a)
        return k.decode_action(action)