"""Bitboard flood fill for the search hot path (no external deps).

The 9x9 board is one 81-bit integer (bit = row*9 + col).
  hb: vertical edges blocked (bit r,c blocks movement between cells (r,c)-(r+1,c))
  vb: horizontal edges blocked (bit r,c blocks movement between cells (r,c)-(r,c+1))
  hs/vs: 64-bit wall-slot masks used for the exact mechanical wall checks.

Semantics are identical to `barricade.game`; these functions exist only for
speed (a few bit operations per BFS step instead of Python dict walks).
"""

from barricade.game import COLS, ROWS

SIZE = ROWS * COLS  # 81
BOARD = (1 << SIZE) - 1
COL0 = sum(1 << (r * COLS) for r in range(ROWS))
COL8 = sum(1 << (r * COLS + COLS - 1) for r in range(ROWS))
GOAL0 = (1 << COLS) - 1  # top row
GOAL1 = ((1 << SIZE) - 1) & ~((1 << (SIZE - COLS)) - 1)  # bottom row
INF = 10 ** 9


def idx(r, c):
    return r * COLS + c


def build_masks(h_walls, v_walls):
    hb = vb = hs = vs = 0
    for (r, c) in h_walls:
        hs |= 1 << (r * (COLS - 1) + c)
        hb |= 1 << (r * COLS + c)
        hb |= 1 << (r * COLS + c + 1)
    for (r, c) in v_walls:
        vs |= 1 << (r * (COLS - 1) + c)
        vb |= 1 << (r * COLS + c)
        vb |= 1 << ((r + 1) * COLS + c)
    return hb, vb, hs, vs


def flood_dist(start, hb, vb, goal):
    """BFS distance from cell `start` to any cell in `goal` mask (INF if none)."""
    if goal >> start & 1:
        return 0
    frontier = 1 << start
    seen = frontier
    up_b = hb << COLS   # bit i set => edge above cell i blocked
    left_b = vb << 1    # bit i set => edge left of cell i blocked
    d = 0
    while frontier:
        d += 1
        nxt = (frontier & ~up_b) >> COLS       # up
        nxt |= (frontier & ~hb) << COLS        # down
        nxt |= (frontier & ~left_b & ~COL0) >> 1  # left
        nxt |= (frontier & ~vb & ~COL8) << 1   # right
        nxt &= BOARD & ~seen
        if nxt & goal:
            return d
        seen |= nxt
        frontier = nxt
    return INF


def flood_dists(start, hb, vb, buf):
    """Fill `buf` (len SIZE) with BFS distances from `start`. -1 = unreachable."""
    for i in range(SIZE):
        buf[i] = -1
    buf[start] = 0
    frontier = 1 << start
    seen = frontier
    up_b = hb << COLS
    left_b = vb << 1
    d = 0
    while frontier:
        d += 1
        nxt = (frontier & ~up_b) >> COLS
        nxt |= (frontier & ~hb) << COLS
        nxt |= (frontier & ~left_b & ~COL0) >> 1
        nxt |= (frontier & ~vb & ~COL8) << 1
        nxt &= BOARD & ~seen
        cells = nxt
        while cells:
            b = cells & -cells
            buf[b.bit_length() - 1] = d
            cells &= cells - 1
        seen |= nxt
        frontier = nxt
    return d - 1 if d else 0


def wall_ok(hb, vb, hs, vs, ori, r, c, idx0, idx1):
    """Full wall validity: mechanical checks + both players keep a path."""
    slot = 1 << (r * (COLS - 1) + c)
    if ori == "H":
        e1 = 1 << (r * COLS + c)
        e2 = 1 << (r * COLS + c + 1)
        if hs & slot or vs & slot or hb & e1 or hb & e2:
            return False
        hb2 = hb | e1 | e2
        vb2 = vb
    else:
        e1 = 1 << (r * COLS + c)
        e2 = 1 << ((r + 1) * COLS + c)
        if vs & slot or hs & slot or vb & e1 or vb & e2:
            return False
        hb2 = hb
        vb2 = vb | e1 | e2
    return (flood_dist(idx0, hb2, vb2, GOAL0) < INF
            and flood_dist(idx1, hb2, vb2, GOAL1) < INF)