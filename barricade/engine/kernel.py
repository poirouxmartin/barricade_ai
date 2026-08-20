"""Numba-compiled search kernel for Barricade.

A self-contained negamax search (alpha-beta + transposition table, iterative
deepening) compiled with numba nopython mode.

81-bit masks do not fit in a native 64-bit int, so every 81-bit mask is
represented as a (hi, lo) pair of uint64 (hi holds bits 64..80).

State is an immutable 15-tuple of int64:
    (pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo,
     hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key)

hb/vb are edge masks, hs/vs slot masks; `key` is a Zobrist hash of the
position, maintained incrementally by apply_move/apply_wall.

Actions are encoded as single ints:
  * 0..80   -> move to that cell index
  * 81..208 -> wall: 81 + ori*64 + slot (ori 0 = horizontal, 1 = vertical,
               slot = r*8 + c with r, c in 0..7)

Time budget is enforced by the Python wrapper (time.time is unavailable in
nopython mode), which drives iterative deepening one depth at a time through
`search_depth`; the kernel itself never checks the clock.
"""

import os

import numpy as np
from numba import njit

M17 = np.uint64((1 << 17) - 1)  # 17 low bits of the hi word (board bits 64..80)
M64 = np.uint64((1 << 64) - 1)  # full 64-bit lo word
U0 = np.uint64(0)
U1 = np.uint64(1)

# column-0 / column-8 cell masks (used to stop horizontal edge shifts)
_col0 = sum(1 << (r * 9) for r in range(9))
COL0_HI, COL0_LO = np.uint64(_col0 >> 64), np.uint64(_col0 & 0xFFFFFFFFFFFFFFFF)
del _col0

_col8 = sum(1 << (r * 9 + 8) for r in range(9))
COL8_HI, COL8_LO = np.uint64(_col8 >> 64), np.uint64(_col8 & 0xFFFFFFFFFFFFFFFF)
del _col8

GOAL0 = (1 << 9) - 1                      # player 0 goal: row 0
GOAL1 = ((1 << 81) - 1) ^ ((1 << 72) - 1)  # player 1 goal: row 8
GOAL0_HI, GOAL0_LO = np.uint64(GOAL0 >> 64), np.uint64(GOAL0 & 0xFFFFFFFFFFFFFFFF)
GOAL1_HI, GOAL1_LO = np.uint64(GOAL1 >> 64), np.uint64(GOAL1 & 0xFFFFFFFFFFFFFFFF)

INF = 1 << 30
MATE = 100000
ABORT = 1 << 62
WALL_BASE = 81

WALL_CAP = 12  # max wall moves generated at an interior node
ROOT_WALL_CAP = 24
MAX_PLY = 64  # killer-move table depth (plies)

DIST_W = 4
CONF_W = 0.12  # opponent-wall discount: conf = max(1 - CONF_W * opp_w, CONF_FLOOR)
CONF_FLOOR = 0.3
TEMPO = np.int64(2)     # MCTS playout eval: bonus for the side to move
DETOUR_W = np.int64(2)  # MCTS playout eval: obstruction term weight

# ---- learned evaluation (small MLP, trained offline) -----------------------
# `nn_value` maps 11 handcrafted features to a win probability in [-1, 1]
# from P1's perspective. Weights are loaded from nn_weights.npz next to this
# module; when absent the net is a no-op (zeros) and NN_W is 0, so the base
# eval is unchanged. NN_W is overridden to a positive weight in the generated
# `nn` variants (tools/make_variant.py) used to A/B test the learned term.

NN_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn_weights.npz")
MLP_HID = 32
MLP_FEATS = 11
MLP_SCALE = np.int64(8)  # eval units per full +/-1 win-probability swing
NN_W = np.int64(0)       # blend weight for the learned term (0 = off)

# Fixed weight buffers: weights are copied INTO these arrays so numba (which
# captures the array objects at compile time and caches the compiled code on
# disk) always sees the current values, even after a retrain in another run.
MLP_W1 = np.zeros((MLP_HID, MLP_FEATS), np.float64)
MLP_b1 = np.zeros(MLP_HID, np.float64)
MLP_W2 = np.zeros((MLP_HID, MLP_HID), np.float64)
MLP_b2 = np.zeros(MLP_HID, np.float64)
MLP_W3 = np.zeros(MLP_HID, np.float64)
MLP_b3 = np.zeros(1, np.float64)


def _load_mlp_weights():
    if not os.path.exists(NN_WEIGHTS_PATH):
        return
    _d = np.load(NN_WEIGHTS_PATH)
    MLP_W1[...] = _d["w1"]
    MLP_b1[...] = _d["b1"]
    MLP_W2[...] = _d["w2"]
    MLP_b2[...] = _d["b2"]
    MLP_W3[...] = _d["w3"]
    MLP_b3[...] = _d["b3"]


_load_mlp_weights()


@njit(cache=True, inline='always', nogil=True)
def nn_features(d0, d1, pos0, pos1, wl0, wl1, plies, f):
    """Fill `f` (float64[11]) with the learned-eval features.

    All features are absolute (P1 vs P2) and normalized to ~[0, 1]; the
    perspective is handled by the caller. Same function feeds both the numba
    eval and the offline trainer, so there is no separate feature scaler.
    Pawn rows are encoded as *distance to each pawn's own goal* (adv0 = r0,
    adv1 = 8 - r1) so the starting position is symmetric, preventing the net
    from learning a wrong base rate at symmetric states.
      0,1  BFS distances to goal, capped at 16
      2,3  P1 row advance / column (0..8)
      4,5  P2 row advance / column (0..8)
      6,7  walls left
      8,9  obstruction (BFS - manhattan ideal), capped at 16
      10   plies, capped at 40
    """
    f[0] = min(d0, 16) / 16.0
    f[1] = min(d1, 16) / 16.0
    f[2] = (pos0 // 9) / 8.0
    f[3] = (pos0 % 9) / 8.0
    f[4] = (8 - pos1 // 9) / 8.0
    f[5] = (pos1 % 9) / 8.0
    f[6] = wl0 / 10.0
    f[7] = wl1 / 10.0
    r0 = pos0 // 9
    c0 = pos0 % 9
    r1 = pos1 // 9
    c1 = pos1 % 9
    f[8] = min(d0 - (r0 + abs(c0 - 4)), 16) / 16.0
    f[9] = min(d1 - ((8 - r1) + abs(c1 - 4)), 16) / 16.0
    f[10] = min(plies, 40) / 40.0


@njit(cache=True, inline='always', nogil=True)
def nn_value(d0, d1, pos0, pos1, wl0, wl1, plies):
    """Learned win probability from P1's perspective, in [-1, 1]."""
    f = np.empty(MLP_FEATS, np.float64)
    nn_features(d0, d1, pos0, pos1, wl0, wl1, plies, f)
    h = np.empty(MLP_HID, np.float64)
    for i in range(MLP_HID):
        acc = MLP_b1[i]
        for j in range(MLP_FEATS):
            acc += MLP_W1[i, j] * f[j]
        h[i] = np.tanh(acc)
    h2 = np.empty(MLP_HID, np.float64)
    for i in range(MLP_HID):
        acc = MLP_b2[i]
        for j in range(MLP_HID):
            acc += MLP_W2[i, j] * h[j]
        h2[i] = np.tanh(acc)
    out = MLP_b3[0]
    for j in range(MLP_HID):
        out += MLP_W3[j] * h2[j]
    return np.tanh(out)


# ---- learned policy + value (move priors and leaf value for MCTS) ----------
# A two-head net maps the board to raw policy logits over the 209 actions
# (81 moves + 128 wall slots) and a scalar value in [-1, 1], both from the
# side-to-move perspective. MCTS softmaxes the logits over its legal children
# (PUCT priors) and uses the value as the leaf evaluation. Weights are loaded
# from policy_weights.npz next to this module; when absent the buffers stay
# zero so logits are all 0 (uniform priors) and MCTS falls back to playouts.

POLICY_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_weights.npz")
NN_IN = 294   # my pawn + opp pawn one-hot, 64 H slots, 64 V slots, walls left,
              # then my/opp BFS distance to goal (distillation of the eval)
NN_HID = 64
NN_POL = 209  # 81 moves + 128 wall slots

# Fixed weight buffers (same convention as MLP_* above).
POL_W1 = np.zeros((NN_HID, NN_IN), np.float64)
POL_b1 = np.zeros(NN_HID, np.float64)
POL_W2 = np.zeros((NN_HID, NN_HID), np.float64)
POL_b2 = np.zeros(NN_HID, np.float64)
POL_Wp = np.zeros((NN_POL, NN_HID), np.float64)
POL_bp = np.zeros(NN_POL, np.float64)
POL_Wv = np.zeros(NN_HID, np.float64)
POL_bv = np.zeros(1, np.float64)

POLICY_LOADED = False
POL_VALUE_LOADED = False


def _load_policy_weights():
    global POLICY_LOADED, POL_VALUE_LOADED
    if not os.path.exists(POLICY_WEIGHTS_PATH):
        return
    _d = np.load(POLICY_WEIGHTS_PATH)
    POL_W1[...] = _d["w1"]
    POL_b1[...] = _d["b1"]
    POL_W2[...] = _d["w2"]
    POL_b2[...] = _d["b2"]
    POL_Wp[...] = _d["wp"]
    POL_bp[...] = _d["bp"]
    POLICY_LOADED = True
    if "wv" in _d and "bv" in _d:
        POL_Wv[...] = _d["wv"]
        POL_bv[...] = _d["bv"]
        POL_VALUE_LOADED = True


_load_policy_weights()


@njit(cache=True, inline='always', nogil=True)
def nn_policy_features(st, f):
    """Fill `f` (float64[NN_IN]) with board features from the mover's perspective.

    Symmetric for both players: my pawn one-hot (0..81), opp pawn one-hot
    (81..162), horizontal wall slots (162..226), vertical wall slots
    (226..290), my walls left / opp walls left (290..292). The two BFS
    distances (my/opp to their goals) are appended by the caller so the value
    head can distill the kernel's evaluation without reconstructing wall
    connectivity. Same function feeds the numba MCTS and the offline trainer.
    """
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    for i in range(NN_IN):
        f[i] = 0.0
    if turn == 0:
        f[pos0] = 1.0
        f[81 + pos1] = 1.0
        f[290] = wl0 / 10.0
        f[291] = wl1 / 10.0
    else:
        f[pos1] = 1.0
        f[81 + pos0] = 1.0
        f[290] = wl1 / 10.0
        f[291] = wl0 / 10.0
    mask = hs_lo | hs_hi
    for i in range(64):
        if (mask >> np.uint64(i)) & np.uint64(1):
            f[162 + i] = 1.0
    mask = vs_lo | vs_hi
    for i in range(64):
        if (mask >> np.uint64(i)) & np.uint64(1):
            f[226 + i] = 1.0


@njit(cache=True, inline='always', nogil=True)
def nn_policy_dists(st):
    """(my BFS distance to my goal, opp BFS distance) from the mover's view."""
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    d0 = int(flood_dist(pos0, hb_hi, hb_lo, vb_hi, vb_lo, GOAL0_HI, GOAL0_LO))
    d1 = int(flood_dist(pos1, hb_hi, hb_lo, vb_hi, vb_lo, GOAL1_HI, GOAL1_LO))
    if turn == 0:
        return d0, d1
    return d1, d0


@njit(cache=True, inline='always', nogil=True)
def nn_policy_value(st, d_my, d_opp, logits, value):
    """Policy logits over the 209 actions and value in [-1, 1], both from the
    mover's perspective, sharing one trunk forward (no softmax on the logits)."""
    f = np.empty(NN_IN, np.float64)
    nn_policy_features(st, f)
    f[NN_IN - 2] = d_my / 16.0
    f[NN_IN - 1] = d_opp / 16.0
    h = np.empty(NN_HID, np.float64)
    for i in range(NN_HID):
        acc = POL_b1[i]
        for j in range(NN_IN):
            acc += POL_W1[i, j] * f[j]
        h[i] = np.tanh(acc)
    h2 = np.empty(NN_HID, np.float64)
    for i in range(NN_HID):
        acc = POL_b2[i]
        for j in range(NN_HID):
            acc += POL_W2[i, j] * h[j]
        h2[i] = np.tanh(acc)
    for k in range(NN_POL):
        acc = POL_bp[k]
        for j in range(NN_HID):
            acc += POL_Wp[k, j] * h2[j]
        logits[k] = acc
    acc = POL_bv[0]
    for j in range(NN_HID):
        acc += POL_Wv[j] * h2[j]
    value[0] = np.tanh(acc)

# ---- zobrist layout -------------------------------------------------------
ZB_POS1 = 81
ZB_HB = 162
ZB_VB = 243
ZB_HS = 324
ZB_VS = 388
ZB_WL0 = 452
ZB_WL1 = 463
ZB_TURN = 474
ZB_SIZE = 476


# ---- mask pair helpers ----------------------------------------------------

@njit(cache=True, inline='always', nogil=True)
def bit(hi, lo, i):
    """Bit `i` (0..80) of an 81-bit mask stored as (hi, lo), as np.int64."""
    if i < 64:
        return np.int64((lo >> np.uint64(i)) & U1)
    return np.int64((hi >> (np.uint64(i) - np.uint64(64))) & U1)


@njit(cache=True, inline='always', nogil=True)
def set_bit(hi, lo, i):
    """Set bit `i`; returns the new (hi, lo) pair."""
    if i < 64:
        return hi, lo | (U1 << np.uint64(i))
    return hi | (U1 << (np.uint64(i) - np.uint64(64))), lo


@njit(cache=True, inline='always', nogil=True)
def shl(hi, lo, k):
    """(hi, lo) << k, truncated to the 81-bit board."""
    ku = np.uint64(k)
    return ((hi << ku) | (lo >> (np.uint64(64) - ku))) & M17, (lo << ku) & M64


@njit(cache=True, inline='always', nogil=True)
def shr(hi, lo, k):
    """(hi, lo) >> k."""
    ku = np.uint64(k)
    return (hi >> ku), ((lo >> ku) | ((hi & ((U1 << ku) - U1)) << (np.uint64(64) - ku)))


# ---- flood fill -----------------------------------------------------------

@njit(cache=True, nogil=True)
def flood_dist(start, hb_hi, hb_lo, vb_hi, vb_lo, goal_hi, goal_lo):
    """Shortest-path distance from `start` to any cell of `goal`, or INF."""
    if bit(goal_hi, goal_lo, start):
        return 0
    if start < 64:
        f_hi, f_lo = U0, U1 << np.uint64(start)
    else:
        f_hi, f_lo = U1 << (np.uint64(start) - np.uint64(64)), U0
    seen_hi, seen_lo = f_hi, f_lo
    up_hi, up_lo = shl(hb_hi, hb_lo, 9)
    left_hi, left_lo = shl(vb_hi, vb_lo, 1)
    d = 0
    while (f_hi | f_lo) != U0:
        d += 1
        nxt_hi, nxt_lo = shr(f_hi & ~up_hi, f_lo & ~up_lo, 9)
        t_hi, t_lo = shl(f_hi & ~hb_hi, f_lo & ~hb_lo, 9)
        nxt_hi |= t_hi
        nxt_lo |= t_lo
        t_hi, t_lo = shr(f_hi & ~left_hi & ~COL0_HI, f_lo & ~left_lo & ~COL0_LO, 1)
        nxt_hi |= t_hi
        nxt_lo |= t_lo
        t_hi, t_lo = shl(f_hi & ~vb_hi & ~COL8_HI, f_lo & ~vb_lo & ~COL8_LO, 1)
        nxt_hi |= t_hi
        nxt_lo |= t_lo
        nxt_hi &= M17 & ~seen_hi
        nxt_lo &= M64 & ~seen_lo
        if ((nxt_hi & goal_hi) | (nxt_lo & goal_lo)) != U0:
            return d
        seen_hi |= nxt_hi
        seen_lo |= nxt_lo
        f_hi, f_lo = nxt_hi, nxt_lo
    return INF


@njit(cache=True, nogil=True)
def flood_dists(start, hb_hi, hb_lo, vb_hi, vb_lo, dist):
    """Fill `dist` (int64[81]) with BFS distances from `start` (-1 unreachable)."""
    for i in range(81):
        dist[i] = -1
    dist[start] = 0
    q = np.zeros(81, np.int64)
    q[0] = start
    head = 0
    tail = 1
    while head < tail:
        cur = q[head]
        head += 1
        r = cur // 9
        c = cur - r * 9
        if r > 0 and bit(hb_hi, hb_lo, cur - 9) == 0 and dist[cur - 9] < 0:
            dist[cur - 9] = dist[cur] + 1
            q[tail] = cur - 9
            tail += 1
        if r < 8 and bit(hb_hi, hb_lo, cur) == 0 and dist[cur + 9] < 0:
            dist[cur + 9] = dist[cur] + 1
            q[tail] = cur + 9
            tail += 1
        if c > 0 and bit(vb_hi, vb_lo, cur - 1) == 0 and dist[cur - 1] < 0:
            dist[cur - 1] = dist[cur] + 1
            q[tail] = cur - 1
            tail += 1
        if c < 8 and bit(vb_hi, vb_lo, cur) == 0 and dist[cur + 1] < 0:
            dist[cur + 1] = dist[cur] + 1
            q[tail] = cur + 1
            tail += 1


# ---- move generation ------------------------------------------------------

@njit(cache=True, inline='always', nogil=True)
def edge_free(a, b, hb_hi, hb_lo, vb_hi, vb_lo):
    """True if the edge between adjacent cells `a` and `b` is open.

    An edge is indexed by the cell nearer the origin: the upper cell for
    vertical edges (hb), the left cell for horizontal edges (vb).
    """
    if a - b == 9:
        return bit(hb_hi, hb_lo, b) == 0
    if b - a == 9:
        return bit(hb_hi, hb_lo, a) == 0
    if a - b == 1:
        return bit(vb_hi, vb_lo, b) == 0
    if b - a == 1:
        return bit(vb_hi, vb_lo, a) == 0
    return False


@njit(cache=True, inline='always', nogil=True)
def push_move(out, n, t):
    out[n] = t
    return n + 1


@njit(cache=True, inline='always', nogil=True)
def try_jump(pos, t, hb_hi, hb_lo, vb_hi, vb_lo, out, n):
    """`t` is the opponent cell: try the straight jump, then the diagonals."""
    pr = pos // 9
    pc = pos - pr * 9
    tr = t // 9
    tc = t - tr * 9
    dr = tr - pr
    dc = tc - pc
    sr = tr + dr
    sc = tc + dc
    if 0 <= sr < 9 and 0 <= sc < 9 and edge_free(t, sr * 9 + sc, hb_hi, hb_lo, vb_hi, vb_lo):
        return push_move(out, n, sr * 9 + sc)
    for ddr, ddc in ((dc, dr), (-dc, -dr)):
        nr = tr + ddr
        nc = tc + ddc
        if 0 <= nr < 9 and 0 <= nc < 9 and edge_free(t, nr * 9 + nc, hb_hi, hb_lo, vb_hi, vb_lo):
            n = push_move(out, n, nr * 9 + nc)
    return n


@njit(cache=True, nogil=True)
def gen_moves(pos, opp, hb_hi, hb_lo, vb_hi, vb_lo, out):
    """Write legal move targets for the pawn at `pos` into `out`; returns count."""
    n = 0
    r = pos // 9
    c = pos - r * 9
    if r > 0 and bit(hb_hi, hb_lo, pos - 9) == 0:
        t = pos - 9
        if t == opp:
            n = try_jump(pos, t, hb_hi, hb_lo, vb_hi, vb_lo, out, n)
        else:
            n = push_move(out, n, t)
    if r < 8 and bit(hb_hi, hb_lo, pos) == 0:
        t = pos + 9
        if t == opp:
            n = try_jump(pos, t, hb_hi, hb_lo, vb_hi, vb_lo, out, n)
        else:
            n = push_move(out, n, t)
    if c > 0 and bit(vb_hi, vb_lo, pos - 1) == 0:
        t = pos - 1
        if t == opp:
            n = try_jump(pos, t, hb_hi, hb_lo, vb_hi, vb_lo, out, n)
        else:
            n = push_move(out, n, t)
    if c < 8 and bit(vb_hi, vb_lo, pos) == 0:
        t = pos + 1
        if t == opp:
            n = try_jump(pos, t, hb_hi, hb_lo, vb_hi, vb_lo, out, n)
        else:
            n = push_move(out, n, t)
    return n


# ---- wall generation ------------------------------------------------------

@njit(cache=True, inline='always', nogil=True)
def wall_ok(st, ori, r, c):
    """Mechanical check + both pawns still reach their goals with wall (ori, r, c)."""
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    slot = r * 8 + c
    if bit(hs_hi, hs_lo, slot) or bit(vs_hi, vs_lo, slot):
        return False
    if ori == 0:
        e1 = r * 9 + c
        e2 = r * 9 + c + 1
        if bit(hb_hi, hb_lo, e1) or bit(hb_hi, hb_lo, e2):
            return False
        hb_hi, hb_lo = set_bit(hb_hi, hb_lo, e1)
        hb_hi, hb_lo = set_bit(hb_hi, hb_lo, e2)
        hs_hi, hs_lo = set_bit(hs_hi, hs_lo, slot)
    else:
        e1 = r * 9 + c
        e2 = (r + 1) * 9 + c
        if bit(vb_hi, vb_lo, e1) or bit(vb_hi, vb_lo, e2):
            return False
        vb_hi, vb_lo = set_bit(vb_hi, vb_lo, e1)
        vb_hi, vb_lo = set_bit(vb_hi, vb_lo, e2)
        vs_hi, vs_lo = set_bit(vs_hi, vs_lo, slot)
    if flood_dist(pos0, hb_hi, hb_lo, vb_hi, vb_lo, GOAL0_HI, GOAL0_LO) >= INF:
        return False
    if flood_dist(pos1, hb_hi, hb_lo, vb_hi, vb_lo, GOAL1_HI, GOAL1_LO) >= INF:
        return False
    return True


@njit(nogil=True)
def gen_walls(st, my, opp, out, cap):
    """Candidate legal walls near the opponent's path, capped; returns count."""
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    buf = np.zeros(81, np.int64)
    flood_dists(opp, hb_hi, hb_lo, vb_hi, vb_lo, buf)

    # cell with minimal distance to the opponent's goal, then walk back
    best = -1
    bd = INF
    if opp == pos0:
        for i in range(9):
            if buf[i] >= 0 and buf[i] < bd:
                bd = buf[i]
                best = i
    else:
        for i in range(72, 81):
            if buf[i] >= 0 and buf[i] < bd:
                bd = buf[i]
                best = i
    if best < 0:
        return 0
    path = np.zeros(81, np.int64)
    np_ = 0
    cur = best
    path[np_] = cur
    np_ += 1
    while buf[cur] > 0:
        d = buf[cur] - 1
        r = cur // 9
        c = cur - r * 9
        found = -1
        if r > 0 and buf[cur - 9] == d:
            found = cur - 9
        elif r < 8 and buf[cur + 9] == d:
            found = cur + 9
        elif c > 0 and buf[cur - 1] == d:
            found = cur - 1
        elif c < 8 and buf[cur + 1] == d:
            found = cur + 1
        if found < 0:
            break
        path[np_] = found
        np_ += 1
        cur = found

    # candidate slots: corners of every path cell + window around our pawn
    seen = np.full(64, -1, np.int64)
    cands = np.zeros(64, np.int64)
    nsl = 0
    for i in range(np_):
        r = path[i] // 9
        c = path[i] - r * 9
        for rr in (r - 1, r):
            for cc in (c - 1, c):
                if 0 <= rr < 8 and 0 <= cc < 8:
                    s = rr * 8 + cc
                    if seen[s] < 0:
                        seen[s] = 1
                        cands[nsl] = s
                        nsl += 1
    pr = my // 9
    pc = my - pr * 9
    for rr in range(max(0, pr - 1), min(8, pr + 2)):
        for cc in range(max(0, pc - 1), min(8, pc + 2)):
            s = rr * 8 + cc
            if seen[s] < 0:
                seen[s] = 1
                cands[nsl] = s
                nsl += 1

    # order by Manhattan distance to the opponent pawn (first path-corner
    # candidate stays first: deliberate ordering heuristic)
    or_ = np.zeros(64, np.int64)
    keys = np.zeros(64, np.int64)
    or_[0] = cands[0]
    keys[0] = 0
    for i in range(1, nsl):
        s = cands[i]
        r = s // 8
        c = s % 8
        kv = abs(r - opp // 9) + abs(c - (opp - opp // 9 * 9))
        j = i - 1
        while j >= 0 and keys[j] > kv:
            keys[j + 1] = keys[j]
            or_[j + 1] = or_[j]
            j -= 1
        keys[j + 1] = kv
        or_[j + 1] = s

    nw = 0
    for i in range(nsl):
        if nw >= cap:
            break
        s = or_[i]
        r = s // 8
        c = s % 8
        if wall_ok(st, 0, r, c):
            out[nw] = WALL_BASE + s
            nw += 1
        if nw >= cap:
            break
        if wall_ok(st, 1, r, c):
            out[nw] = WALL_BASE + 64 + s
            nw += 1
    return nw


# ---- evaluation -----------------------------------------------------------

@njit(cache=True, inline='always', nogil=True)
def eval_fn(st):
    """Static score from the perspective of the side to move.

    Matches evaluate.score exactly: distance advantage scaled by how much we
    trust it given the opponent's remaining walls (deficit never discounted),
    plus wall-count advantage. Earlier versions added a detour term
    (DETOUR_W * obstruction) and a tempo bonus; they were removed because the
    unscaled obstruction dwarfed the distance signal, making the engine
    wall-spam and lose to a greedy opponent (see audit). The MCTS playout eval
    keeps those terms in playout_eval below: a randomized forward search
    rewards the detour signal, a deep alpha-beta does not.
    """
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    d0 = flood_dist(pos0, hb_hi, hb_lo, vb_hi, vb_lo, GOAL0_HI, GOAL0_LO)
    d1 = flood_dist(pos1, hb_hi, hb_lo, vb_hi, vb_lo, GOAL1_HI, GOAL1_LO)
    if turn == 0:
        my_d, opp_d, my_w, opp_w = d0, d1, wl0, wl1
    else:
        my_d, opp_d, my_w, opp_w = d1, d0, wl1, wl0
    dist_adv = (opp_d - my_d) * DIST_W
    if dist_adv > 0:
        conf = max(1.0 - CONF_W * opp_w, CONF_FLOOR)
        dist_adv = np.int64(dist_adv * conf)
    base = dist_adv + (my_w - opp_w)
    if NN_W != 0:
        # learned term (off by default; enabled in the generated nn variants)
        nn_t = nn_value(d0, d1, pos0, pos1, wl0, wl1, plies)
        if turn == 1:
            nn_t = -nn_t
        base += NN_W * np.int64(nn_t * MLP_SCALE)
    return base


@njit(cache=True, inline='always', nogil=True)
def playout_eval(st):
    """MCTS playout/race leaf value from the side to move.

    eval_fn plus TEMPO and the obstruction term (who is actually more
    detoured by walls: (BFS distance - manhattan ideal) for each side). The
    MCTS normalizes this with tanh. These terms are deliberately NOT in the
    kernel's eval_fn: in a randomized forward playout the detour signal helps
    the search value real blocking, while in a deep alpha-beta the same
    unscaled term made the engine wall-spam against greedy. The two engines
    therefore use different leaf evaluations (kernel: eval_fn, mcts:
    playout_eval).
    """
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    d0 = flood_dist(pos0, hb_hi, hb_lo, vb_hi, vb_lo, GOAL0_HI, GOAL0_LO)
    d1 = flood_dist(pos1, hb_hi, hb_lo, vb_hi, vb_lo, GOAL1_HI, GOAL1_LO)
    if turn == 0:
        my_d, opp_d, my_w, opp_w = d0, d1, wl0, wl1
        my_p, opp_p = pos0, pos1
    else:
        my_d, opp_d, my_w, opp_w = d1, d0, wl1, wl0
        my_p, opp_p = pos1, pos0
    dist_adv = (opp_d - my_d) * DIST_W
    if dist_adv > 0:
        conf = max(1.0 - CONF_W * opp_w, CONF_FLOOR)
        dist_adv = np.int64(dist_adv * conf)
    m_my = (my_p // 9 if turn == 0 else 8 - my_p // 9) + abs(my_p % 9 - 4)
    m_opp = (opp_p // 9 if turn != 0 else 8 - opp_p // 9) + abs(opp_p % 9 - 4)
    obstruction = (opp_d - m_opp) - (my_d - m_my)
    return dist_adv + (my_w - opp_w) + TEMPO + DETOUR_W * obstruction


# ---- state transitions ----------------------------------------------------

@njit(nogil=True)
def apply_move(st, t, zob):
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    if turn == 0:
        key ^= zob[pos0] ^ zob[t]
        pos0 = t
    else:
        key ^= zob[ZB_POS1 + pos1] ^ zob[ZB_POS1 + t]
        pos1 = t
    turn = 1 - turn
    key ^= zob[ZB_TURN]
    plies += 1
    return (pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo,
            hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key)


@njit(nogil=True)
def apply_wall(st, action, zob):
    w = action - WALL_BASE
    ori = w // 64
    slot = w % 64
    r = slot // 8
    c = slot % 8
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    if ori == 0:
        e1 = r * 9 + c
        e2 = r * 9 + c + 1
        hb_hi, hb_lo = set_bit(hb_hi, hb_lo, e1)
        hb_hi, hb_lo = set_bit(hb_hi, hb_lo, e2)
        hs_hi, hs_lo = set_bit(hs_hi, hs_lo, slot)
        key ^= zob[ZB_HB + e1] ^ zob[ZB_HB + e2] ^ zob[ZB_HS + slot]
    else:
        e1 = r * 9 + c
        e2 = (r + 1) * 9 + c
        vb_hi, vb_lo = set_bit(vb_hi, vb_lo, e1)
        vb_hi, vb_lo = set_bit(vb_hi, vb_lo, e2)
        vs_hi, vs_lo = set_bit(vs_hi, vs_lo, slot)
        key ^= zob[ZB_VB + e1] ^ zob[ZB_VB + e2] ^ zob[ZB_VS + slot]
    if turn == 0:
        key ^= zob[ZB_WL0 + wl0]
        wl0 -= 1
        key ^= zob[ZB_WL0 + wl0]
    else:
        key ^= zob[ZB_WL1 + wl1]
        wl1 -= 1
        key ^= zob[ZB_WL1 + wl1]
    turn = 1 - turn
    key ^= zob[ZB_TURN]
    plies += 1
    return (pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo,
            hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key)


# ---- search ---------------------------------------------------------------

@njit(nogil=True)
def negamax(st, depth, alpha, beta, ply, tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop):
    if stop[0]:
        return ABORT
    budget[0] -= 1
    if budget[0] < 0:
        return ABORT
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    # a terminal state means the side to move has just lost (the previous
    # mover reached its goal row), so the value is always a loss: plies - MATE.
    if pos0 < 9 or pos1 >= 72:
        return plies - MATE
    if depth <= 0:
        return eval_fn(st)

    k2 = key & (tt_k.shape[0] - 1)
    hint = -1
    if tt_k[k2] == key:
        hint = tt_m[k2]
        if tt_d[k2] >= depth:
            if tt_f[k2] == 0:
                return tt_v[k2]
            if tt_f[k2] == 1 and tt_v[k2] >= beta:
                return tt_v[k2]
            if tt_f[k2] == 2 and tt_v[k2] <= alpha:
                return tt_v[k2]

    my = pos0 if turn == 0 else pos1
    opp = pos1 if turn == 0 else pos0
    original_alpha = alpha
    best = -MATE - 1
    best_action = -1
    fail_high = False

    moves = np.zeros(16, np.int64)
    nm = gen_moves(my, opp, hb_hi, hb_lo, vb_hi, vb_lo, moves)
    if depth >= 4:
        buf = np.zeros(81, np.int64)
        flood_dists(my, hb_hi, hb_lo, vb_hi, vb_lo, buf)
        for i in range(1, nm):
            mv = moves[i]
            kv = buf[mv]
            j = i - 1
            while j >= 0 and buf[moves[j]] > kv:
                moves[j + 1] = moves[j]
                j -= 1
            moves[j + 1] = mv
    if 0 <= hint < 81:
        for i in range(nm):
            if moves[i] == hint:
                for j in range(i, 0, -1):
                    moves[j] = moves[j - 1]
                moves[0] = hint
                break
    kk = killers[2 * ply]
    if kk >= 0 and kk < 81:
        for i in range(1, nm):
            if moves[i] == kk:
                for j in range(i, 1, -1):
                    moves[j] = moves[j - 1]
                moves[1] = kk
                break
    kk = killers[2 * ply + 1]
    if kk >= 0 and kk < 81:
        for i in range(1, nm):
            if moves[i] == kk:
                for j in range(i, 1, -1):
                    moves[j] = moves[j - 1]
                moves[1] = kk
                break

    for i in range(nm):
        red = 0
        if depth >= 4 and i >= 2:
            red = 1
        raw = negamax(apply_move(st, moves[i], zob), depth - 1 - red, -beta, -alpha, ply + 1,
                      tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop)
        if raw == ABORT:
            return ABORT
        s = -raw
        if red > 0 and s > alpha and s < beta:
            raw = negamax(apply_move(st, moves[i], zob), depth - 1, -beta, -alpha, ply + 1,
                          tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop)
            if raw == ABORT:
                return ABORT
            s = -raw
        if s >= beta:
            best = s
            fail_high = True
            best_action = moves[i]
            killers[2 * ply + 1] = killers[2 * ply]
            killers[2 * ply] = moves[i]
            break
        if s > best:
            best = s
            best_action = moves[i]
        if s > alpha:
            alpha = s
            killers[2 * ply + 1] = killers[2 * ply]
            killers[2 * ply] = moves[i]

    if not fail_high:
        wl = wl0 if turn == 0 else wl1
        if wl > 0:
            walls = np.zeros(32, np.int64)
            nw = gen_walls(st, my, opp, walls, WALL_CAP)
            if hint >= WALL_BASE:
                for i in range(nw):
                    if walls[i] == hint:
                        for j in range(i, 0, -1):
                            walls[j] = walls[j - 1]
                        walls[0] = hint
                        break
            kk = killers[2 * ply]
            if kk >= WALL_BASE:
                for i in range(1, nw):
                    if walls[i] == kk:
                        for j in range(i, 1, -1):
                            walls[j] = walls[j - 1]
                        walls[1] = kk
                        break
            kk = killers[2 * ply + 1]
            if kk >= WALL_BASE:
                for i in range(1, nw):
                    if walls[i] == kk:
                        for j in range(i, 1, -1):
                            walls[j] = walls[j - 1]
                        walls[1] = kk
                        break
            for i in range(nw):
                red = 0
                if depth >= 4 and i >= 2:
                    red = 1
                if depth >= 6 and i >= 4:
                    red = 2
                raw = negamax(apply_wall(st, walls[i], zob), depth - 1 - red, -beta, -alpha, ply + 1,
                              tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop)
                if raw == ABORT:
                    return ABORT
                s = -raw
                if red > 0 and s > alpha and s < beta:
                    raw = negamax(apply_wall(st, walls[i], zob), depth - 1, -beta, -alpha, ply + 1,
                                  tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop)
                    if raw == ABORT:
                        return ABORT
                    s = -raw
                if s >= beta:
                    best = s
                    fail_high = True
                    best_action = walls[i]
                    killers[2 * ply + 1] = killers[2 * ply]
                    killers[2 * ply] = walls[i]
                    break
                if s > best:
                    best = s
                    best_action = walls[i]
                if s > alpha:
                    alpha = s
                    killers[2 * ply + 1] = killers[2 * ply]
                    killers[2 * ply] = walls[i]

    if abs(best) < MATE - 1000:
        tt_k[k2] = key
        tt_d[k2] = depth
        tt_v[k2] = best
        tt_m[k2] = best_action
        if fail_high:
            tt_f[k2] = 1
        elif best > original_alpha:
            tt_f[k2] = 0
        else:
            tt_f[k2] = 2
    return best


@njit(nogil=True)
def _root_search(st, depth, alpha, beta, moves, nm, walls, nw, ply,
                 tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop,
                 out_moves, out_scores, want_out):
    """Root move loop for one depth level; returns (action, score, aborted).

    When `want_out` is nonzero, each root move's score is recorded in
    `out_moves`/`out_scores` (moves first, then walls) for multi-PV display.
    """
    if stop[0]:
        return (-1, -MATE - 1, True)
    best_action = -1
    best_score = -MATE - 1
    for i in range(nm):
        raw = negamax(apply_move(st, moves[i], zob), depth - 1, -beta, -alpha, ply + 1,
                      tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop)
        if raw == ABORT:
            return (best_action, best_score, True)
        s = -raw
        if want_out:
            out_moves[i] = moves[i]
            out_scores[i] = s
        if s > best_score:
            best_score = s
            best_action = moves[i]
        if s > alpha:
            alpha = s
    for i in range(nw):
        raw = negamax(apply_wall(st, walls[i], zob), depth - 1, -beta, -alpha, ply + 1,
                      tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop)
        if raw == ABORT:
            return (best_action, best_score, True)
        s = -raw
        if want_out:
            out_moves[nm + i] = walls[i]
            out_scores[nm + i] = s
        if s > best_score:
            best_score = s
            best_action = walls[i]
        if s > alpha:
            alpha = s
    return (best_action, best_score, False)


@njit(nogil=True)
def search_depth(st, depth, hint, prev_score, tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop,
                 out_moves, out_scores, want_out):
    """One iterative-deepening level at the root. Returns (action, score, aborted, out_n)
    where `out_n` is the number of root moves recorded when `want_out` is set."""
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    my = pos0 if turn == 0 else pos1
    opp = pos1 if turn == 0 else pos0

    moves = np.zeros(16, np.int64)
    nm = gen_moves(my, opp, hb_hi, hb_lo, vb_hi, vb_lo, moves)
    buf = np.zeros(81, np.int64)
    flood_dists(my, hb_hi, hb_lo, vb_hi, vb_lo, buf)
    for i in range(1, nm):
        mv = moves[i]
        kv = buf[mv]
        j = i - 1
        while j >= 0 and buf[moves[j]] > kv:
            moves[j + 1] = moves[j]
            j -= 1
        moves[j + 1] = mv
    if hint >= 0 and hint < 81:
        for i in range(nm):
            if moves[i] == hint:
                for j in range(i, 0, -1):
                    moves[j] = moves[j - 1]
                moves[0] = hint
                break

    wl = wl0 if turn == 0 else wl1
    walls = np.zeros(32, np.int64)
    nw = 0
    if wl > 0:
        nw = gen_walls(st, my, opp, walls, ROOT_WALL_CAP)
        if hint >= WALL_BASE:
            for i in range(nw):
                if walls[i] == hint:
                    for j in range(i, 0, -1):
                        walls[j] = walls[j - 1]
                    walls[0] = hint
                    break

    if depth >= 3:
        margin = 16 + depth * 2
        alpha = prev_score - margin
        beta = prev_score + margin
    else:
        alpha = -MATE - 1
        beta = MATE + 1
    action, score, aborted = _root_search(st, depth, alpha, beta, moves, nm, walls, nw, 0,
                                          tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop,
                                          out_moves, out_scores, want_out)
    if not aborted and action >= 0 and (score <= alpha or score >= beta):
        action, score, aborted = _root_search(st, depth, -MATE - 1, MATE + 1, moves, nm,
                                              walls, nw, 0, tt_k, tt_v, tt_d, tt_f, tt_m,
                                              killers, zob, budget, stop,
                                              out_moves, out_scores, want_out)
    return (action, score, aborted, nm + nw)


@njit(nogil=True)
def root_top_moves(st, depth, tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop,
                   out, out_scores, n_total, n_refine):
    """Sort the root moves recorded by `search_depth` by score descending, then
    re-search the top `n_refine` at depth - 1 with a full window for exact scores
    (the main search only has window-bounded values). Returns the number ranked."""
    if stop[0] or n_total <= 0:
        return 0
    for i in range(1, n_total):
        ai = out[i]
        si = out_scores[i]
        j = i - 1
        while j >= 0 and out_scores[j] < si:
            out[j + 1] = out[j]
            out_scores[j + 1] = out_scores[j]
            j -= 1
        out[j + 1] = ai
        out_scores[j + 1] = si
    if depth >= 3 and n_refine > 0:
        limit = min(n_refine, n_total)
        for i in range(limit):
            if stop[0]:
                break
            budget[0] -= 1
            if budget[0] < 0:
                break
            a = out[i]
            if a < WALL_BASE:
                child = apply_move(st, a, zob)
            else:
                child = apply_wall(st, a, zob)
            raw = negamax(child, depth - 1, -MATE - 1, MATE + 1, 1,
                          tt_k, tt_v, tt_d, tt_f, tt_m, killers, zob, budget, stop)
            if raw == ABORT:
                break
            out_scores[i] = -raw
        # exact scores may have changed the order: re-sort the refined prefix
        for i in range(1, limit):
            ai = out[i]
            si = out_scores[i]
            j = i - 1
            while j >= 0 and out_scores[j] < si:
                out[j + 1] = out[j]
                out_scores[j + 1] = out_scores[j]
                j -= 1
            out[j + 1] = ai
            out_scores[j + 1] = si
    return n_total


# ---- Python-facing helpers ------------------------------------------------

def make_state(game, zob):
    """Build the kernel state tuple and its Zobrist key from a Game."""
    from barricade.engine.bitboard import build_masks

    p0r, p0c = game.positions[0]
    p1r, p1c = game.positions[1]
    pos0 = p0r * 9 + p0c
    pos1 = p1r * 9 + p1c
    hb, vb, hs, vs = build_masks(game.h_walls, game.v_walls)
    hb_hi = np.uint64(hb >> 64)
    hb_lo = np.uint64(hb & 0xFFFFFFFFFFFFFFFF)
    vb_hi = np.uint64(vb >> 64)
    vb_lo = np.uint64(vb & 0xFFFFFFFFFFFFFFFF)
    hs_hi = np.uint64(hs >> 64)
    hs_lo = np.uint64(hs & 0xFFFFFFFFFFFFFFFF)
    vs_hi = np.uint64(vs >> 64)
    vs_lo = np.uint64(vs & 0xFFFFFFFFFFFFFFFF)
    turn = game.turn
    plies = len(game.history)

    key = int(zob[pos0]) ^ int(zob[ZB_POS1 + pos1])
    key ^= int(zob[ZB_WL0 + game.walls_left[0]]) ^ int(zob[ZB_WL1 + game.walls_left[1]])
    if turn == 1:
        key ^= int(zob[ZB_TURN])
    for m in (hb, vb):
        x = m
        while x:
            b = x & -x
            i = b.bit_length() - 1
            off = ZB_HB if m is hb else ZB_VB
            key ^= int(zob[off + i])
            x &= x - 1
    for m in (hs, vs):
        x = m
        while x:
            b = x & -x
            i = b.bit_length() - 1
            off = ZB_HS if m is hs else ZB_VS
            key ^= int(zob[off + i])
            x &= x - 1
    return (np.int64(pos0), np.int64(pos1), np.int64(game.walls_left[0]),
            np.int64(game.walls_left[1]), hb_hi, hb_lo, vb_hi, vb_lo,
            hs_hi, hs_lo, vs_hi, vs_lo, np.int64(turn), np.int64(plies),
            np.int64(key))


def make_zobrist(seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(-(1 << 62), 1 << 62, size=ZB_SIZE, dtype=np.int64)


def nn_features_row(game, zob):
    """11-feature row for the learned eval, from a Game object."""
    st = make_state(game, zob)
    d0 = int(flood_dist(st[0], st[4], st[5], st[6], st[7], GOAL0_HI, GOAL0_LO))
    d1 = int(flood_dist(st[1], st[4], st[5], st[6], st[7], GOAL1_HI, GOAL1_LO))
    f = np.zeros(MLP_FEATS, np.float64)
    nn_features(d0, d1, st[0], st[1], st[2], st[3], st[12], f)
    return f


def nn_policy_features_row(game, zob):
    """NN_IN-feature policy row from a Game object (mover's perspective),
    including the two BFS distances to each goal."""
    st = make_state(game, zob)
    f = np.zeros(NN_IN, np.float64)
    nn_policy_features(st, f)
    d0 = int(flood_dist(st[0], st[4], st[5], st[6], st[7], GOAL0_HI, GOAL0_LO))
    d1 = int(flood_dist(st[1], st[4], st[5], st[6], st[7], GOAL1_HI, GOAL1_LO))
    if st[12] == 0:
        f[NN_IN - 2] = d0 / 16.0
        f[NN_IN - 1] = d1 / 16.0
    else:
        f[NN_IN - 2] = d1 / 16.0
        f[NN_IN - 1] = d0 / 16.0
    return f


def decode_action(action):
    """Kernel action int -> game action tuple (always plain Python ints)."""
    action = int(action)
    if action < 81:
        r, c = divmod(action, 9)
        return ("move", (r, c))
    w = action - WALL_BASE
    ori, slot = divmod(w, 64)
    r, c = divmod(slot, 8)
    return ("wall", ("H" if ori == 0 else "V", r, c))


def encode_action(game, action):
    """game action tuple -> kernel action int (assumes the action is legal)."""
    kind, value = action
    if kind == "move":
        r, c = value
        return r * 9 + c
    ori, r, c = value
    slot = r * 8 + c
    return WALL_BASE + (0 if ori == "H" else 64) + slot