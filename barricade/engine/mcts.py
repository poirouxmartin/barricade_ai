"""MctsEngine: numba-compiled Monte Carlo Tree Search for Barricade.

One node per iteration (children allocated lazily), UCT selection with the
negamax sign convention, and random playouts that fall back to the shared
static evaluation normalized with tanh. Like KernelEngine, Python only drives
batches of iterations against the wall clock because time.time is unavailable
in nopython mode.

Node pool layout (persistent across batches so the tree survives between
time-budget slices):
  pool_st[MAXN][15]   state tuple of each node
  parent[MAXN]        -1 for the root
  cstart/ccount       child window [cstart, cstart+ccount) into ca/cv
  ca[MAXN_CH]         child node id, -1 until the action is first visited
  cv[MAXN_CH]         encoded action
  visits[MAXN]        visit count
  vsum[MAXN]          sum of results from the mover's perspective
  n_nodes, next_ch    1-element persistent cursors
"""

import random
import time

import numpy as np
from numba import njit

from barricade.engine.base import Engine

try:
    from barricade.engine import kernel
    _HAS_NUMBA = True
except ImportError:
    kernel = None
    _HAS_NUMBA = False

WALL_PROB = 0.4
SIM_PLIES = 60
UCT_C = 1.4
EVAL_SCALE = 30.0


@njit(cache=True, inline='always')
def is_terminal(st):
    return st[0] < 9 or st[1] >= 72


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def apply_action(st, action, zob):
    st = norm_state(st)
    if action < 81:
        return kernel.apply_move(st, action, zob)
    return kernel.apply_wall(st, action, zob)


@njit(cache=True)
def expand(node, pool_st, cstart, ccount, ca, cv, next_ch, zob):
    """Fill ca/cv with the legal actions of `node` (node ids stay -1)."""
    st = st_tuple(pool_st[node])
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    my = pos0 if turn == 0 else pos1
    opp = pos1 if turn == 0 else pos0
    wl = wl0 if turn == 0 else wl1

    base = next_ch[0]
    n = 0
    max_ch = ca.shape[0]
    moves = np.zeros(16, np.int64)
    nm = kernel.gen_moves(my, opp, hb_hi, hb_lo, vb_hi, vb_lo, moves)
    for i in range(nm):
        if base + n >= max_ch:
            break
        ca[base + n] = -1
        cv[base + n] = moves[i]
        n += 1
    if wl > 0:
        walls = np.zeros(32, np.int64)
        nw = kernel.gen_walls(st, my, opp, walls, kernel.WALL_CAP)
        for i in range(nw):
            if base + n >= max_ch:
                break
            ca[base + n] = -1
            cv[base + n] = walls[i]
            n += 1
    cstart[node] = base
    ccount[node] = n
    next_ch[0] = base + n


@njit(cache=True)
def simulate(st, zob):
    """BFS-greedy race playout; returns the result from the mover's perspective."""
    st = norm_state(st)
    for _ in range(SIM_PLIES):
        if is_terminal(st):
            break
        pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
        my = pos0 if turn == 0 else pos1
        opp = pos1 if turn == 0 else pos0
        wl = wl0 if turn == 0 else wl1
        if wl > 0 and np.random.rand() < WALL_PROB:
            walls = np.zeros(32, np.int64)
            nw = kernel.gen_walls(st, my, opp, walls, kernel.WALL_CAP)
            if nw > 0:
                st = kernel.apply_wall(st, walls[np.random.randint(0, nw)], zob)
                continue
        buf = np.zeros(81, np.int64)
        kernel.flood_dists(my, hb_hi, hb_lo, vb_hi, vb_lo, buf)
        moves = np.zeros(16, np.int64)
        nm = kernel.gen_moves(my, opp, hb_hi, hb_lo, vb_hi, vb_lo, moves)
        if nm == 0:
            break
        best_d = 1000
        best_n = 0
        best_idx = np.zeros(8, np.int64)
        for i in range(nm):
            d = buf[moves[i]]
            if d < best_d:
                best_d = d
                best_n = 0
            if d == best_d and best_n < 8:
                best_idx[best_n] = i
                best_n += 1
        st = kernel.apply_move(st, moves[best_idx[np.random.randint(0, best_n)]], zob)
    if is_terminal(st):
        return -1.0  # the side to move has just lost
    return np.tanh(kernel.eval_fn(st) / EVAL_SCALE)


@njit(cache=True)
def backprop(node, v, parent, visits, vsum):
    vcur = v
    while node >= 0:
        visits[node] += 1
        vsum[node] += vcur
        vcur = -vcur
        node = parent[node]


@njit(cache=True, inline='always')
def st_array(st):
    return np.array(st, np.int64)


@njit(cache=True, inline='always')
def st_tuple(row):
    return (row[0], row[1], row[2], row[3],
            np.uint64(row[4]), np.uint64(row[5]),
            np.uint64(row[6]), np.uint64(row[7]),
            np.uint64(row[8]), np.uint64(row[9]),
            np.uint64(row[10]), np.uint64(row[11]),
            row[12], row[13], row[14])


@njit(cache=True, inline='always')
def norm_state(st):
    """Force the mask fields to uint64 (a Python crossing may retype them int64)."""
    return (np.int64(st[0]), np.int64(st[1]), np.int64(st[2]), np.int64(st[3]),
            np.uint64(st[4]), np.uint64(st[5]), np.uint64(st[6]), np.uint64(st[7]),
            np.uint64(st[8]), np.uint64(st[9]), np.uint64(st[10]), np.uint64(st[11]),
            np.int64(st[12]), np.int64(st[13]), np.int64(st[14]))


@njit(cache=True)
def mcts_run(st, n_iter, pool_st, parent, cstart, ccount, ca, cv, visits, vsum,
             n_nodes, next_ch, zob):
    pool_st[0] = st_array(st)
    max_nodes = pool_st.shape[0]
    for _ in range(n_iter):
        node = 0
        # selection
        while True:
            if is_terminal(pool_st[node]):
                break
            nc = ccount[node]
            if nc == 0:
                expand(node, pool_st, cstart, ccount, ca, cv, next_ch, zob)
                nc = ccount[node]
                if nc == 0:
                    break
            base = cstart[node]
            exp_child = -1
            best_c = -1
            best_score = -1e300
            for k in range(nc):
                c = ca[base + k]
                if c < 0:
                    exp_child = base + k
                    break
                q = -vsum[c] / visits[c]
                u = UCT_C * np.sqrt(np.log(visits[node] + 1.0) / visits[c])
                score = q + u
                if score > best_score:
                    best_score = score
                    best_c = c
            if exp_child >= 0:
                if n_nodes[0] >= max_nodes:
                    break
                new_id = n_nodes[0]
                n_nodes[0] += 1
                ca[exp_child] = new_id
                parent[new_id] = node
                cstart[new_id] = 0
                ccount[new_id] = 0
                visits[new_id] = 0
                vsum[new_id] = 0.0
                pool_st[new_id] = st_array(apply_action(st_tuple(pool_st[node]), cv[exp_child], zob))
                node = new_id
                break
            node = best_c
        # simulation + backprop
        v = -1.0 if is_terminal(pool_st[node]) else simulate(st_tuple(pool_st[node]), zob)
        backprop(node, v, parent, visits, vsum)


@njit(cache=True)
def best_action(cstart, ccount, ca, cv, visits):
    nc = ccount[0]
    base = cstart[0]
    best_a = -1
    best_v = -1
    for k in range(nc):
        c = ca[base + k]
        if c >= 0 and visits[c] > best_v:
            best_v = visits[c]
            best_a = cv[base + k]
    if best_a < 0 and nc > 0:
        best_a = cv[base]
    return best_a


class MctsEngine(Engine):
    def __init__(self, max_nodes=100_000, time_limit=2.0, seed=7):
        if not _HAS_NUMBA:
            raise ImportError("numba is required for MctsEngine")
        self.time_limit = time_limit
        self.max_nodes = max_nodes
        self.zob = kernel.make_zobrist(seed)
        max_ch = max_nodes * 16
        self.pool_st = np.zeros((max_nodes, 15), np.int64)
        self.parent = np.full(max_nodes, -1, np.int64)
        self.cstart = np.zeros(max_nodes, np.int64)
        self.ccount = np.zeros(max_nodes, np.int64)
        self.ca = np.full(max_ch, -1, np.int64)
        self.cv = np.zeros(max_ch, np.int64)
        self.visits = np.zeros(max_nodes, np.int64)
        self.vsum = np.zeros(max_nodes, np.float64)
        self.n_nodes = np.zeros(1, np.int64)
        self.next_ch = np.zeros(1, np.int64)
        self.ips_est = 10_000.0  # smoothed iterations/s
        self.last_info = {}
        self._warm = False

    def _warmup(self):
        if self._warm:
            return
        self._warm = True
        seed_rng(7)
        from barricade.game import Barricade
        self._reset_tree()
        st = kernel.make_state(Barricade(), self.zob)
        mcts_run(st, 1000, self.pool_st, self.parent, self.cstart, self.ccount,
                 self.ca, self.cv, self.visits, self.vsum,
                 self.n_nodes, self.next_ch, self.zob)

    def _reset_tree(self):
        self.n_nodes[0] = 1
        self.next_ch[0] = 0
        self.parent[0] = -1
        self.cstart[0] = 0
        self.ccount[0] = 0
        self.visits[0] = 0
        self.vsum[0] = 0.0

    def choose_move(self, game):
        self._warmup()
        self._reset_tree()
        st = kernel.make_state(game, self.zob)
        deadline = time.time() + self.time_limit
        best = random.choice(game.legal_actions())
        total_iters = 0
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            allowance = int(remaining * self.ips_est) + 1000
            allowance = min(allowance, self.max_nodes - int(self.n_nodes[0]) - 1)
            if allowance <= 0:
                break
            t0 = time.time()
            mcts_run(st, allowance, self.pool_st, self.parent, self.cstart,
                     self.ccount, self.ca, self.cv, self.visits, self.vsum,
                     self.n_nodes, self.next_ch, self.zob)
            dt = max(time.time() - t0, 1e-6)
            total_iters += allowance
            if dt > 0.03:
                instant = allowance / dt
                instant = min(max(instant, 1000.0), 1_000_000.0)
                self.ips_est = 0.5 * self.ips_est + 0.5 * instant
            if int(self.n_nodes[0]) >= self.max_nodes - 1:
                break
        action = best_action(self.cstart, self.ccount, self.ca, self.cv, self.visits)
        if action >= 0:
            best = kernel.decode_action(action)
        self.last_info = {"iterations": total_iters,
                          "nodes": int(self.n_nodes[0])}
        return best
