"""Continuous analysis session for the Barricade engines.

Two engine modes:
  * "kernel" (default): iterative deepening negamax. A live snapshot exposes
    the current depth, score, win probability, node/s rate, best move, the
    principal variation, and a ranked list of the top root moves.
  * "mcts": UCT search on a persistent tree. Iterations keep running until
    stopped; the snapshot exposes the best move (most visited child) and the
    top children with their average value, scaled to the kernel score units.
"""

import math
import threading
import time

import numpy as np

from barricade.engine import kernel
from barricade.engine.numba_engine import KernelEngine
from barricade.engine.mcts import MctsEngine, mcts_run, best_action, EVAL_SCALE
from barricade.engine import evaluate

MAX_DEPTH = 30
TOP_MOVES = 5
TT_SIZE = 1 << 22
PV_LEN = 12
WIN_K = 8.0  # logistic scale: score -> win probability
NODE_BUDGET = 100_000_000
PV_REFRESH_DEPTH = 4  # shallow re-search depth when the TT walk hits a miss
MCTS_NODES = 1 << 18
MCTS_TICK = 0.25  # seconds of iterations between snapshot updates


class AnalysisSession:
    def __init__(self, game, engine="kernel", max_depth=MAX_DEPTH, top_moves=TOP_MOVES,
                 tt_size=TT_SIZE, mcts_nodes=MCTS_NODES, mcts_tick=MCTS_TICK):
        if engine not in ("kernel", "mcts"):
            raise ValueError(f"unknown analysis engine: {engine!r}")
        self.engine = engine
        self._tt_size = tt_size
        self._mcts_nodes = mcts_nodes or MCTS_NODES
        self._mcts_tick = mcts_tick or MCTS_TICK
        self._engine = None
        self._mcts = None
        self.game = game.clone()
        self.max_depth = max_depth or MAX_DEPTH
        self.top_moves_n = top_moves
        self._stop = threading.Event()
        self._stop_flag = np.zeros(1, np.int64)
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
        self._stop_flag[0] = 0
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._stop_flag[0] = 1
        self.running = False

    def _position_eval(self):
        """Distance gap (squares) and confidence factor for the analyzed position."""
        g = self.game
        d0 = g.dist_to_goal(0)
        d1 = g.dist_to_goal(1)
        w0 = g.walls_left[0]
        w1 = g.walls_left[1]
        my_d, opp_d, my_w, opp_w = (d0, d1, w0, w1) if g.turn == 0 else (d1, d0, w1, w0)
        raw = (opp_d - my_d) * evaluate.DIST_W
        conf = 1.0
        if raw > 0:
            conf = max(1.0 - evaluate.CONF_W * opp_w, evaluate.CONF_FLOOR)
        return {
            "d0": d0, "d1": d1,
            "dist_gap": opp_d - my_d,
            "confidence": round(conf, 3),
        }

    def state(self):
        with self._lock:
            pe = self._position_eval()
            return {
                "running": self.running,
                "engine": self.engine,
                "depth": self.depth,
                "score": self.score,
                "win_chance": self._win_chance(self.score),
                "nodes": self.nodes,
                "nps": int(self.nps),
                "best_move": self.best_move,
                "pv": self.pv,
                "top_moves": self.top,
                "turn": self.game.turn,
                "positions": self.game.positions,
                "h_walls": sorted(self.game.h_walls),
                "v_walls": sorted(self.game.v_walls),
                "d0": pe["d0"], "d1": pe["d1"],
                "dist_gap": pe["dist_gap"],
                "confidence": pe["confidence"],
                "error": self.error,
            }

    # -- internals ----------------------------------------------------------

    def _win_chance(self, score):
        if abs(score) > kernel.MATE - 1000:
            return 1.0 if score > 0 else 0.0
        return 1.0 / (1.0 + math.exp(-score / WIN_K))

    def _consume(self):
        """Nodes consumed by the last search (budget was reset to NODE_BUDGET)."""
        return NODE_BUDGET - max(0, int(self._engine.budget[0]))

    def _run(self):
        try:
            if self.engine == "mcts":
                self._run_mcts()
            else:
                self._run_kernel()
        except Exception as e:
            with self._lock:
                self.error = str(e)
        finally:
            self.running = False
            self._stop.clear()

    def _run_kernel(self):
        self._engine = KernelEngine(max_depth=1, time_limit=1.0, tt_size=self._tt_size)
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
                if self._stop.is_set():
                    break
                depth += 1
                self._engine.budget[0] = NODE_BUDGET
                a, s, ab, out_n = kernel.search_depth(
                    st, depth, hint, prev,
                    self._engine.tt_k, self._engine.tt_v, self._engine.tt_d,
                    self._engine.tt_f, self._engine.tt_m, self._engine.killers,
                    self._engine.zob, self._engine.budget, self._stop_flag,
                    self._out, self._out_scores, 1)
                total += self._consume()
                if ab:
                    break
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
                        self._engine.zob, self._engine.budget, self._stop_flag,
                        self._out, self._out_scores, out_n, min(self.top_moves_n, 8))
                    total += self._consume()
                top = [{"move": kernel.decode_action(int(self._out[i])),
                        "score": int(self._out_scores[i])}
                       for i in range(min(n, self.top_moves_n))]
                if top and top[0]["move"] != best_move:
                    best_move = top[0]["move"]  # exact score may refine the best
                pv, spent = self._extract_pv(best_move)
                total += spent
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

    def _run_mcts(self):
        me = MctsEngine(max_nodes=self._mcts_nodes, time_limit=self._mcts_tick, seed=7)
        me._warmup()
        me._reset_tree()
        self._mcts = me
        st = kernel.make_state(self.game, me.zob)
        t0 = time.time()
        total = 0
        while not self._stop.is_set():
            allowance = int(self._mcts_tick * me.ips_est) + 1000
            allowance = min(allowance, me.max_nodes - int(me.n_nodes[0]) - 1)
            if allowance <= 0:
                break
            mcts_run(st, allowance, me.pool_st, me.parent, me.cstart, me.ccount,
                     me.ca, me.cv, me.cp, me.visits, me.vsum,
                     me.nval, me.n_nodes, me.next_ch, me.zob,
                     me.policy_on, me.value_on)
            total += allowance
            if self._stop.is_set():
                break
            nc = int(me.ccount[0])
            base = int(me.cstart[0])
            best_a = best_action(me.cstart, me.ccount, me.ca, me.cv, me.visits)
            best_move = kernel.decode_action(best_a) if best_a >= 0 else None
            cand = []
            for k in range(nc):
                c = int(me.ca[base + k])
                if c >= 0 and int(me.visits[c]) > 0:
                    # vsum is accumulated from the child's mover (the opponent)
                    v = -float(me.vsum[c]) / float(me.visits[c])
                    cand.append((kernel.decode_action(int(me.cv[base + k])),
                                 int(me.visits[c]), v))
            cand.sort(key=lambda x: x[1], reverse=True)
            top = [{"move": mv, "score": int(round(v * EVAL_SCALE))}
                   for mv, _n, v in cand[:self.top_moves_n]]
            score = int(round(cand[0][2] * EVAL_SCALE)) if cand else 0
            with self._lock:
                self.depth = 1 if cand else 0
                self.score = score
                self.nodes = total
                self.nps = total / max(time.time() - t0, 1e-6)
                self.best_move = best_move
                self.pv = [best_move] if best_move else []
                self.top = top

    def _extract_pv(self, best_move):
        pv = [best_move] if best_move else []
        spent = 0
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
                m, _, ab, _out_n = kernel.search_depth(
                    st, PV_REFRESH_DEPTH, -1, 0,
                    tt_k, self._engine.tt_v, self._engine.tt_d, self._engine.tt_f,
                    tt_m, self._engine.killers, self._engine.zob, self._engine.budget,
                    self._stop_flag, self._out, self._out_scores, 0)
                spent += self._consume()
                if ab or m < 0:
                    break
            act = kernel.decode_action(m)
            if act not in g.legal_actions():
                break
            pv.append(act)
            g.apply(act, check=False)
            if g.winner is not None:
                break
        return pv, spent