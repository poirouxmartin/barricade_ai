"""Continuous analysis session for the Barricade kernel.

Iterative deepening keeps running until stopped; a live snapshot exposes the
current depth, score, win probability, node/s rate, best move, the principal
variation, and a ranked list of the top root moves with estimated scores.
"""

import math
import threading
import time

import numpy as np

from barricade.engine import kernel
from barricade.engine.numba_engine import KernelEngine

MAX_DEPTH = 30
TOP_MOVES = 5
TT_SIZE = 1 << 22
PV_LEN = 12
WIN_K = 8.0  # logistic scale: score -> win probability
NODE_BUDGET = 100_000_000
PV_REFRESH_DEPTH = 4  # shallow re-search depth when the TT walk hits a miss


class AnalysisSession:
    def __init__(self, game, max_depth=MAX_DEPTH, top_moves=TOP_MOVES, tt_size=TT_SIZE):
        self._engine = KernelEngine(max_depth=1, time_limit=1.0, tt_size=tt_size)
        self.game = game.clone()
        self.max_depth = max_depth
        self.top_moves_n = top_moves
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._out = np.zeros(64, np.int64)
        self._out_scores = np.zeros(64, np.int64)
        self.depth = 0
        self.score = 0
        self.nodes = 0
        self.nps = 0.0
        self.best_move = None
        self.pv = []
        self.top = []
        self.running = False
        self.error = None

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.running = False

    def state(self):
        with self._lock:
            return {
                "running": self.running,
                "depth": self.depth,
                "score": self.score,
                "win_chance": self._win_chance(self.score),
                "nodes": self.nodes,
                "nps": int(self.nps),
                "best_move": self.best_move,
                "pv": self.pv,
                "top_moves": self.top,
                "turn": self.game.turn,
                "error": self.error,
            }

    # -- internals ----------------------------------------------------------

    def _win_chance(self, score):
        if abs(score) > kernel.MATE - 1000:
            return 1.0 if score > 0 else 0.0
        return 1.0 / (1.0 + math.exp(-score / WIN_K))

    def _run(self):
        try:
            self._engine._warmup()
            st = kernel.make_state(self.game, self._engine.zob)
            hint = -1
            prev = 0
            t0 = time.time()
            total = 0
            depth = 0
            best_move = None
            while not self._stop.is_set() and depth < self.max_depth:
                depth += 1
                self._engine.budget[0] = NODE_BUDGET
                a, s, ab = kernel.search_depth(
                    st, depth, hint, prev,
                    self._engine.tt_k, self._engine.tt_v, self._engine.tt_d,
                    self._engine.tt_f, self._engine.tt_m, self._engine.killers,
                    self._engine.zob, self._engine.budget)
                if ab:
                    break
                used = max(0, NODE_BUDGET - int(self._engine.budget[0]))
                total += used
                if a >= 0:
                    hint = a
                    prev = int(s)
                    best_move = kernel.decode_action(a)
                n = 0
                if depth >= 2 and not self._stop.is_set():
                    self._engine.budget[0] = NODE_BUDGET
                    n = kernel.root_top_moves(
                        st, depth,
                        self._engine.tt_k, self._engine.tt_v, self._engine.tt_d,
                        self._engine.tt_f, self._engine.tt_m, self._engine.killers,
                        self._engine.zob, self._engine.budget,
                        self._out, self._out_scores)
                top = [{"move": kernel.decode_action(int(self._out[i])),
                        "score": int(self._out_scores[i])}
                       for i in range(min(n, self.top_moves_n))]
                pv = self._extract_pv(best_move)
                with self._lock:
                    self.depth = depth
                    self.score = int(s)
                    self.nodes = total
                    self.nps = total / max(time.time() - t0, 1e-6)
                    self.best_move = best_move
                    self.pv = pv
                    self.top = top
        except Exception as e:
            with self._lock:
                self.error = str(e)
        finally:
            self.running = False
            self._stop.clear()

    def _extract_pv(self, best_move):
        pv = [best_move] if best_move else []
        g = self.game.clone()
        if best_move is not None:
            g.apply(best_move, check=False)
        tt_k = self._engine.tt_k
        tt_m = self._engine.tt_m
        for _ in range(PV_LEN):
            st = kernel.make_state(g, self._engine.zob)
            key = int(st[14])
            k2 = key & (tt_k.shape[0] - 1)
            if int(tt_k[k2]) == key and int(tt_m[k2]) >= 0:
                m = int(tt_m[k2])
            else:
                # entry evicted: refresh this node with a shallow search
                self._engine.budget[0] = NODE_BUDGET
                m, _, ab = kernel.search_depth(
                    st, PV_REFRESH_DEPTH, -1, 0,
                    tt_k, self._engine.tt_v, self._engine.tt_d, self._engine.tt_f,
                    tt_m, self._engine.killers, self._engine.zob, self._engine.budget)
                if ab or m < 0:
                    break
            act = kernel.decode_action(m)
            if act not in g.legal_actions():
                break
            pv.append(act)
            g.apply(act, check=False)
            if g.winner is not None:
                break
        return pv