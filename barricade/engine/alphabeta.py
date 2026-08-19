"""Alpha-beta (negamax) engine.

Iterative deepening with a transposition table.

The search hot path (flood-fill distances, wall validity) runs on bitboards
(`barricade.engine.bitboard`) instead of the rule engine's Python BFS.

Branching control:
  * wall moves are only considered near the opponent's shortest path to its
    goal (plus a small window around our own pawn), capped to `wall_cap`;
  * move actions are searched before wall actions, so early beta cutoffs
    skip the expensive wall generation entirely.

`choose_move` returns the best legal action. `last_info` reports the search
statistics (depth reached, nodes, score).
"""

import random
import time

from barricade.engine import evaluate
from barricade.engine.base import Engine
from barricade.engine.bitboard import (
    GOAL0,
    GOAL1,
    INF,
    build_masks,
    flood_dist,
    flood_dists,
    idx,
    wall_ok,
)
from barricade.game import COLS, ROWS

MATE = 100000


class AlphaBetaEngine(Engine):
    def __init__(self, max_depth=8, time_limit=2.0, wall_cap=12):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.wall_cap = wall_cap
        self.tt = {}
        self.nodes = 0
        self.deadline = 0.0
        self.last_info = {}
        self.progress = {}
        self.buf = [0] * 81  # scratch distance field, reused per node

    def choose_move(self, game):
        self.tt = {}
        self.nodes = 0
        self.deadline = time.time() + self.time_limit
        best = random.choice(game.legal_actions())
        for depth in range(1, self.max_depth + 1):
            if time.time() > self.deadline:
                break
            scored = self._root_negamax(game, depth, best)
            if scored is None:
                break
            best, score = scored
            self.last_info = {"depth": depth, "nodes": self.nodes, "score": score}
            self.progress = dict(self.last_info)
        return best

    # ----- search -----

    def _root_negamax(self, game, depth, hint):
        alpha, beta = -MATE - 1, MATE + 1
        best_action, best_score = None, None
        player = game.turn
        masks = build_masks(game.h_walls, game.v_walls)
        goal = GOAL0 if player == 0 else GOAL1
        flood_dists(idx(*game.positions[player]), masks[0], masks[1], self.buf)
        buf = self.buf

        def consider(action, g):
            nonlocal alpha, best_score, best_action
            s = self._negamax(g, depth - 1, -beta, -alpha)
            if s is None:
                return False
            s = -s
            if best_score is None or s > best_score:
                best_score, best_action = s, action
            if s > alpha:
                alpha = s
            return True

        moves = sorted(game.legal_moves(player), key=lambda m: buf[idx(*m)])
        if hint and hint[0] == "move" and hint[1] in moves:
            moves.remove(hint[1])
            moves.insert(0, hint[1])
        for m in moves:
            g = game.clone()
            g.apply(("move", m), check=False)
            if not consider(("move", m), g):
                return None

        if game.walls_left[player] > 0:
            walls = self._wall_candidates(game, masks, cap=self.wall_cap * 2)
            if hint and hint[0] == "wall" and hint[1] in walls:
                walls.remove(hint[1])
                walls.insert(0, hint[1])
            for wall in walls:
                g = game.clone()
                g.apply(("wall", wall), check=False)
                if not consider(("wall", wall), g):
                    return None

        return best_action, best_score

    def _negamax(self, game, depth, alpha, beta):
        if time.time() > self.deadline:
            return None
        self.nodes += 1
        if game.winner is not None:
            return -MATE + len(game.history)

        hb, vb, hs, vs = build_masks(game.h_walls, game.v_walls)
        i0 = idx(*game.positions[0])
        i1 = idx(*game.positions[1])

        if depth <= 0:
            p = game.turn
            d0 = flood_dist(i0, hb, vb, GOAL0)
            d1 = flood_dist(i1, hb, vb, GOAL1)
            return evaluate.score(d0 if p == 0 else d1, d1 if p == 0 else d0,
                                  game.walls_left[p], game.walls_left[1 - p])

        key = self._key(game)
        entry = self.tt.get(key)
        if entry is not None and entry[0] >= depth:
            val, flag = entry[1], entry[2]
            if flag == 0:
                return val
            if flag == 1 and val >= beta:
                return val
            if flag == 2 and val <= alpha:
                return val

        player = game.turn
        original_alpha = alpha

        # phase 1: moves (cheap), ordered by one distance field from our pawn
        flood_dists(idx(*game.positions[player]), hb, vb, self.buf)
        for m in sorted(game.legal_moves(player), key=lambda m: self.buf[idx(*m)]):
            g = game.clone()
            g.apply(("move", m), check=False)
            s = self._negamax(g, depth - 1, -beta, -alpha)
            if s is None:
                return None
            s = -s
            if s >= beta:
                if abs(s) < MATE - 1000:
                    self.tt[key] = (depth, s, 1)
                return s
            if s > alpha:
                alpha = s

        # phase 2: walls (expensive, generated lazily)
        if game.walls_left[player] > 0:
            for wall in self._wall_candidates(game, (hb, vb, hs, vs), cap=self.wall_cap):
                g = game.clone()
                g.apply(("wall", wall), check=False)
                s = self._negamax(g, depth - 1, -beta, -alpha)
                if s is None:
                    return None
                s = -s
                if s >= beta:
                    if abs(s) < MATE - 1000:
                        self.tt[key] = (depth, s, 1)
                    return s
                if s > alpha:
                    alpha = s

        if abs(alpha) < MATE - 1000:
            self.tt[key] = (depth, alpha, 0 if alpha > original_alpha else 2)
        return alpha

    # ----- move/wall generation helpers -----

    def _wall_candidates(self, game, masks, cap):
        hb, vb, hs, vs = masks
        slots = set()
        for (r, c) in self._opp_path_cells(game, hb, vb):
            for rr in (r - 1, r):
                for cc in (c - 1, c):
                    if 0 <= rr < ROWS - 1 and 0 <= cc < ROWS - 1:
                        slots.add((rr, cc))
        pr, pc = game.positions[game.turn]
        for rr in range(max(0, pr - 1), min(ROWS - 1, pr + 2)):
            for cc in range(max(0, pc - 1), min(ROWS - 1, pc + 2)):
                slots.add((rr, cc))
        op = game.positions[1 - game.turn]
        ordered = sorted(slots, key=lambda s: abs(s[0] - op[0]) + abs(s[1] - op[1]))
        i0 = idx(*game.positions[0])
        i1 = idx(*game.positions[1])
        out = []
        for (r, c) in ordered:
            if len(out) >= cap:
                break
            if wall_ok(hb, vb, hs, vs, "H", r, c, i0, i1):
                out.append(("H", r, c))
            if wall_ok(hb, vb, hs, vs, "V", r, c, i0, i1):
                out.append(("V", r, c))
        return out

    def _opp_path_cells(self, game, hb, vb):
        """Cells of one shortest path from the opponent pawn to its goal
        (reconstructed from the bitboard distance field, no Python BFS)."""
        opp = 1 - game.turn
        start = idx(*game.positions[opp])
        goal = GOAL0 if opp == 0 else GOAL1
        flood_dists(start, hb, vb, self.buf)
        buf = self.buf
        g = goal
        best, bd = -1, -1
        while g:
            b = g & -g
            i = b.bit_length() - 1
            if buf[i] >= 0 and (bd < 0 or buf[i] < bd):
                best, bd = i, buf[i]
            g &= g - 1
        if best < 0:
            return []
        path = [best]
        while buf[path[-1]] > 0:
            cur = path[-1]
            nxt = self._step_back(cur, buf)
            if nxt is None:
                break
            path.append(nxt)
        return [(i // COLS, i % COLS) for i in reversed(path)]

    def _step_back(self, i, buf):
        """A neighbor of cell i whose distance field is one less (path to start)."""
        r, c = divmod(i, COLS)
        d = buf[i] - 1
        for cand in (
            (i - COLS if r > 0 else -1),
            (i + COLS if r < ROWS - 1 else -1),
            (i - 1 if c > 0 else -1),
            (i + 1 if c < COLS - 1 else -1),
        ):
            if cand >= 0 and buf[cand] == d:
                return cand
        return None

    def _key(self, game):
        return (game.positions[0], game.positions[1], game.turn,
                frozenset(game.h_walls), frozenset(game.v_walls))