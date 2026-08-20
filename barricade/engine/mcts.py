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
  cstart/ccount       child window [cstart, cstart+ccount) into ca/cv/cp
  ca[MAXN_CH]         child node id, -1 until the action is first visited
  cv[MAXN_CH]         encoded action
  cp[MAXN_CH]         PUCT prior (masked softmax over the legal children)
  visits[MAXN]        visit count
  vsum[MAXN]          sum of results from the mover's perspective
  nval[MAXN]          policy-net value of the node (mover's perspective)
  n_nodes, next_ch    1-element persistent cursors

Selection is PUCT with the policy-net priors. The leaf value is the policy
net's value head when loaded and enabled (value_on); otherwise the pure-race
playouts are used as the reliable fallback.
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

WALL_PROB = 0.0
SIM_PLIES = 60
PUCT_C = 1.5
PRIOR_T = 1.2  # softmax temperature on the policy logits (>1 flattens priors)
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
def expand(node, pool_st, cstart, ccount, ca, cv, cp, next_ch, zob, policy_on):
    """Fill ca/cv with the legal actions of `node` (node ids stay -1) and set
    each child's PUCT prior to the masked softmax of the policy-net logits
    over those legal actions (uniform when the policy net is off)."""
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
    keys = np.zeros(16, np.int64)
    for i in range(nm):
        mr = moves[i] // 9
        mc = moves[i] - mr * 9
        if turn == 0:
            keys[i] = mr * 9 + abs(mc - 4)
        else:
            keys[i] = (8 - mr) * 9 + abs(mc - 4)
    for i in range(1, nm):
        mv = moves[i]
        kv = keys[i]
        j = i - 1
        while j >= 0 and keys[j] > kv:
            moves[j + 1] = moves[j]
            keys[j + 1] = keys[j]
            j -= 1
        moves[j + 1] = mv
        keys[j + 1] = kv
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
    if n > 0:
        if policy_on:
            logits = np.empty(kernel.NN_POL, np.float64)
            val = np.zeros(1, np.float64)
            dm, do = kernel.nn_policy_dists(st)
            kernel.nn_policy_value(st, dm, do, logits, val)
            mx = logits[cv[base]]
            for k in range(1, n):
                v = logits[cv[base + k]]
                if v > mx:
                    mx = v
            zsum = 0.0
            for k in range(n):
                cp[base + k] = np.exp((logits[cv[base + k]] - mx) / PRIOR_T)
                zsum += cp[base + k]
            for k in range(n):
                cp[base + k] /= zsum
        else:
            inv = 1.0 / n
            for k in range(n):
                cp[base + k] = inv
    cstart[node] = base
    ccount[node] = n
    next_ch[0] = base + n


@njit(cache=True)
def simulate(st, zob):
    """Pure race playout; result from the mover's perspective.

    Every move advances toward the goal row (random among equal-key targets).
    Walls are never placed in playouts: a wall's value is reflected by the
    detour it forces on the racing opponent, so the tree still learns real
    blocking without over-exploring wall moves.
    """
    st = norm_state(st)
    for _ in range(SIM_PLIES):
        if is_terminal(st):
            break
        pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
        my = pos0 if turn == 0 else pos1
        opp = pos1 if turn == 0 else pos0
        moves = np.zeros(16, np.int64)
        nm = kernel.gen_moves(my, opp, hb_hi, hb_lo, vb_hi, vb_lo, moves)
        if nm == 0:
            break
        # race greedily toward the goal row: minimize (row, |col-4|) for P1,
        # maximize row for P2 (jumps land further and win the tiebreak).
        best_key = 1 << 60
        best_n = 0
        best_idx = np.zeros(8, np.int64)
        for i in range(nm):
            mr = moves[i] // 9
            mc = moves[i] - mr * 9
            if turn == 0:
                key = mr * 9 + abs(mc - 4)
            else:
                key = (8 - mr) * 9 + abs(mc - 4)
            if key < best_key:
                best_key = key
                best_n = 0
            if key == best_key and best_n < 8:
                best_idx[best_n] = i
                best_n += 1
        st = kernel.apply_move(st, moves[best_idx[np.random.randint(0, best_n)]], zob)
    if is_terminal(st):
        return -1.0  # the side to move has just lost
    return np.tanh(kernel.eval_fn(st) / EVAL_SCALE)


@njit(cache=True)
def race_value(st, zob):
    """Deterministic race projection for a position, from the input mover's
    perspective in [-1, 1].

    Races SIM_PLIES plies greedily (both sides advancing toward their goals,
    first best-key move, no randomness). If a side reaches its goal the input
    mover won/lost accordingly; otherwise the final eval (after an even number
    of plies, the mover is the input mover) is used. This is the value target
    the policy net's value head distills: it embeds the tactical race lookahead
    that the static evaluation alone lacks.
    """
    st = norm_state(st)
    input_turn = st[12]
    for _ in range(SIM_PLIES):
        if is_terminal(st):
            break
        pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
        my = pos0 if turn == 0 else pos1
        opp = pos1 if turn == 0 else pos0
        moves = np.zeros(16, np.int64)
        nm = kernel.gen_moves(my, opp, hb_hi, hb_lo, vb_hi, vb_lo, moves)
        if nm == 0:
            break
        best_key = 1 << 60
        best = moves[0]
        for i in range(nm):
            mr = moves[i] // 9
            mc = moves[i] - mr * 9
            if turn == 0:
                key = mr * 9 + abs(mc - 4)
            else:
                key = (8 - mr) * 9 + abs(mc - 4)
            if key < best_key:
                best_key = key
                best = moves[i]
        st = kernel.apply_move(st, best, zob)
    if is_terminal(st):
        # Soften the win/loss label: a flat +-1 near the goal saturates the
        # value head (no gradient to train on) and the search stalls on the
        # wrong prior. Use the loser's remaining BFS distance at the same
        # scale as the non-terminal eval (DIST_W * margin / EVAL_SCALE).
        if st[0] < 9:
            winner = 0
            loser_dist = kernel.flood_dist(st[1], st[4], st[5], st[6], st[7],
                                           kernel.GOAL1_HI, kernel.GOAL1_LO)
        else:
            winner = 1
            loser_dist = kernel.flood_dist(st[0], st[4], st[5], st[6], st[7],
                                           kernel.GOAL0_HI, kernel.GOAL0_LO)
        sign = 1.0 if input_turn == winner else -1.0
        return sign * np.tanh(kernel.DIST_W * loser_dist / EVAL_SCALE)
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
def mcts_run(st, n_iter, pool_st, parent, cstart, ccount, ca, cv, cp, visits,
             vsum, nval, n_nodes, next_ch, zob, policy_on, value_on):
    pool_st[0] = st_array(st)
    max_nodes = pool_st.shape[0]
    for _ in range(n_iter):
        node = 0
        stuck = False
        # selection
        while True:
            if is_terminal(pool_st[node]):
                break
            nc = ccount[node]
            if nc == 0:
                expand(node, pool_st, cstart, ccount, ca, cv, cp, next_ch, zob, policy_on)
                nc = ccount[node]
                if nc == 0:
                    stuck = True
                    break
            base = cstart[node]
            best_c = -1
            best_slot = -1
            best_score = -1e300
            for k in range(nc):
                c = ca[base + k]
                if c >= 0:
                    q = -vsum[c] / visits[c]
                    u = PUCT_C * cp[base + k] * np.sqrt(visits[node] + 1.0) / (1.0 + visits[c])
                    score = q + u
                else:
                    u = PUCT_C * cp[base + k] * np.sqrt(visits[node] + 1.0)
                    score = u
                if score > best_score:
                    best_score = score
                    best_c = c
                    best_slot = base + k
            if best_slot < 0:
                stuck = True
                break
            if best_c < 0:
                # expand the highest-prior unvisited child
                if n_nodes[0] >= max_nodes:
                    break
                new_id = n_nodes[0]
                n_nodes[0] += 1
                ca[best_slot] = new_id
                parent[new_id] = node
                cstart[new_id] = 0
                ccount[new_id] = 0
                visits[new_id] = 0
                vsum[new_id] = 0.0
                pool_st[new_id] = st_array(apply_action(st_tuple(pool_st[node]), cv[best_slot], zob))
                if value_on:
                    val = np.zeros(1, np.float64)
                    lg = np.empty(kernel.NN_POL, np.float64)
                    child_st = st_tuple(pool_st[new_id])
                    dm, do = kernel.nn_policy_dists(child_st)
                    kernel.nn_policy_value(child_st, dm, do, lg, val)
                    nval[new_id] = val[0]
                node = new_id
                break
            node = best_c
        # leaf value + backprop
        if is_terminal(pool_st[node]) or stuck:
            v = -1.0  # the side to move has just lost
        elif value_on:
            v = nval[node]
        else:
            v = simulate(st_tuple(pool_st[node]), zob)
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
    def __init__(self, max_nodes=100_000, time_limit=2.0, seed=7,
                 use_value_net=False):
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
        self.cp = np.zeros(max_ch, np.float64)
        self.visits = np.zeros(max_nodes, np.int64)
        self.vsum = np.zeros(max_nodes, np.float64)
        self.nval = np.zeros(max_nodes, np.float64)
        self.n_nodes = np.zeros(1, np.int64)
        self.next_ch = np.zeros(1, np.int64)
        self.ips_est = 10_000.0  # smoothed iterations/s
        self.last_info = {}
        self.progress = {}
        self._warm = False
        self.REP_WINDOW = 12  # plies; a key seen again inside this window is a repetition
        # Policy priors always use the loaded net (that is the real gain vs the
        # plain playout MCTS). The value head is an experimental leaf
        # evaluator, off by default: even with soft (non-saturating) labels its
        # deterministic distillation of the forward race cannot defend as P2
        # (benchmark: 6-6 vs greedy with a 12-0 P1/P2 split, vs 11-1 balanced
        # for the race playouts), so the random forward-greedy race playouts
        # remain the reliable leaf value.
        self.policy_on = bool(kernel.POLICY_LOADED)
        self.value_on = bool(use_value_net and kernel.POL_VALUE_LOADED)

    def _warmup(self):
        if self._warm:
            return
        self._warm = True
        seed_rng(7)
        from barricade.game import Barricade
        self._reset_tree()
        st = kernel.make_state(Barricade(), self.zob)
        mcts_run(st, 1000, self.pool_st, self.parent, self.cstart, self.ccount,
                 self.ca, self.cv, self.cp, self.visits, self.vsum,
                 self.nval, self.n_nodes, self.next_ch, self.zob,
                 self.policy_on, self.value_on)

    def _reset_tree(self):
        self.n_nodes[0] = 1
        self.next_ch[0] = 0
        self.parent[0] = -1
        self.cstart[0] = 0
        self.ccount[0] = 0
        self.visits[0] = 0
        self.vsum[0] = 0.0

    def _recent_keys(self, game):
        """Zobrist keys of the last REP_WINDOW positions (current included)."""
        from barricade.game import Barricade

        k = kernel
        g = Barricade()
        st = k.make_state(g, self.zob)
        keys = []
        for kind, _player, arg in game.history:
            if kind == "move":
                st = k.apply_move(st, arg[0] * 9 + arg[1], self.zob)
            else:
                ori, r, c = arg
                ob = 0 if ori == "H" else 1
                st = k.apply_wall(st, k.WALL_BASE + ob * 64 + r * 8 + c, self.zob)
            keys.append(st[14])
        if len(keys) <= self.REP_WINDOW:
            return set(keys)
        return set(keys[-self.REP_WINDOW:])

    def _repeats(self, st, action, recent):
        k = kernel
        if action < k.WALL_BASE:
            child = k.apply_move(st, action, self.zob)
        else:
            child = k.apply_wall(st, action, self.zob)
        return child[14] in recent

    def choose_move(self, game):
        self._warmup()
        self._reset_tree()
        st = kernel.make_state(game, self.zob)
        recent = self._recent_keys(game)
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
                     self.ccount, self.ca, self.cv, self.cp, self.visits,
                     self.vsum, self.nval, self.n_nodes, self.next_ch,
                     self.zob, self.policy_on, self.value_on)
            dt = max(time.time() - t0, 1e-6)
            total_iters += allowance
            self.progress = {"iterations": total_iters, "nodes": int(self.n_nodes[0])}
            if dt > 0.03:
                instant = allowance / dt
                instant = min(max(instant, 1000.0), 1_000_000.0)
                self.ips_est = 0.5 * self.ips_est + 0.5 * instant
            if int(self.n_nodes[0]) >= self.max_nodes - 1:
                break
        action = best_action(self.cstart, self.ccount, self.ca, self.cv, self.visits)
        if action >= 0:
            best = self._guard_repetition(st, action, recent)
            win = self._immediate_win(game)
            if win is not None:
                best = win
        self.last_info = {"iterations": total_iters,
                          "nodes": int(self.n_nodes[0])}
        return best

    def _immediate_win(self, game):
        """If the mover can reach its goal row in one move, return it. The
        search sometimes stalls one ply short of a win (saturated leaf
        values), so conversion must never be left to the visit tie-break.
        Uses the pure-Python game rules (a move to row 0 / row 8 wins), not
        the kernel state: crossing uint64 masks >= 2**63 back to numba from
        Python overflows in numba's tuple unboxing."""
        goal = 0 if game.turn == 0 else 8
        for a in game.legal_actions():
            if a[0] == "move" and a[1][0] == goal:
                return a
        return None

    def _guard_repetition(self, st, action, recent):
        """If the most-visited child repeats a recent position, pick the next
        most-visited non-repeating child."""
        if not self._repeats(st, action, recent):
            return kernel.decode_action(action)
        base = int(self.cstart[0])
        nc = int(self.ccount[0])
        children = []
        for k in range(nc):
            c = int(self.ca[base + k])
            if c >= 0:
                children.append((int(self.visits[c]), int(self.cv[base + k])))
        children.sort(reverse=True)
        for _v, a in children:
            if a >= 0 and not self._repeats(st, a, recent):
                return kernel.decode_action(a)
        return kernel.decode_action(action)
