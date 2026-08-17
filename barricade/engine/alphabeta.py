"""Alpha-beta (negamax) engine.

Iterative deepening with a transposition table.

Perf levers to keep the branching factor small:
  * wall moves are only considered near the opponent's shortest path to its
    goal (plus a small window around our own pawn), capped to `wall_cap`;
  * move actions are searched before wall actions, so early beta cutoffs
    skip the expensive wall generation entirely;
  * moves are ordered by row progress toward the goal (free, no BFS).

`choose_move` returns the best legal action. `last_info` reports the search
statistics (depth reached, nodes, score).
"""

import random
import time
from collections import deque

from barricade.engine.base import Engine
from barricade.engine.evaluate import evaluate
from barricade.game import ROWS, goal_row

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
        return best

    # ----- search -----

    def _root_negamax(self, game, depth, hint):
        alpha, beta = -MATE - 1, MATE + 1
        best_action, best_score = None, None
        player = game.turn

        def consider(action):
            nonlocal alpha, best_score, best_action
            g = game.clone()
            g.apply(action)
            s = self._negamax(g, depth - 1, -beta, -alpha)
            if s is None:
                return False
            s = -s
            if best_score is None or s > best_score:
                best_score, best_action = s, action
            if s > alpha:
                alpha = s
            return True

        moves = sorted(game.legal_moves(player),
                       key=lambda m: self._move_key(m, player))
        if hint and hint[0] == "move" and hint[1] in moves:
            moves.remove(hint[1])
            moves.insert(0, hint[1])
        for m in moves:
            if not consider(("move", m)):
                return None

        if game.walls_left[player] > 0:
            walls = self._wall_candidates(game, cap=self.wall_cap * 2)
            if hint and hint[0] == "wall" and hint[1] in walls:
                walls.remove(hint[1])
                walls.insert(0, hint[1])
            for w in walls:
                if not consider(("wall", w)):
                    return None

        return best_action, best_score

    def _negamax(self, game, depth, alpha, beta):
        if time.time() > self.deadline:
            return None
        self.nodes += 1
        if game.winner is not None:
            return -MATE + len(game.history)
        if depth <= 0:
            return evaluate(game, game.turn)

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

        # phase 1: moves (cheap), ordered by row progress
        for m in sorted(game.legal_moves(player), key=lambda m: self._move_key(m, player)):
            g = game.clone()
            g.apply(("move", m))
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
            for wall in self._wall_candidates(game, cap=self.wall_cap):
                g = game.clone()
                g.apply(("wall", wall))
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

    # ----- move generation helpers -----

    def _move_key(self, m, player):
        # row progress toward the goal: P0 goes up (lower row), P1 goes down
        return m[0] if player == 0 else ROWS - 1 - m[0]

    def _wall_candidates(self, game, cap):
        slots = set()
        for (r, c) in self._opp_path_cells(game):
            for rr in (r - 1, r):
                for cc in (c - 1, c):
                    if 0 <= rr < ROWS - 1 and 0 <= cc < ROWS - 1:
                        slots.add((rr, cc))
        # small window around our own pawn as well
        pr, pc = game.positions[game.turn]
        for rr in range(max(0, pr - 1), min(ROWS - 1, pr + 2)):
            for cc in range(max(0, pc - 1), min(ROWS - 1, pc + 2)):
                slots.add((rr, cc))
        op = game.positions[1 - game.turn]
        ordered = sorted(slots, key=lambda s: abs(s[0] - op[0]) + abs(s[1] - op[1]))
        out = []
        for (r, c) in ordered:
            if len(out) >= cap:
                break
            if game.wall_valid("H", r, c):
                out.append(("H", r, c))
            if game.wall_valid("V", r, c):
                out.append(("V", r, c))
        return out

    def _opp_path_cells(self, game):
        """Cells of one shortest path from the opponent pawn to its goal."""
        opp = 1 - game.turn
        start = game.positions[opp]
        parent = {start: None}
        q = deque([start])
        target = None
        while q:
            cell = q.popleft()
            if cell[0] == goal_row(opp):
                target = cell
                break
            for n in game.neighbors(*cell):
                if n not in parent:
                    parent[n] = cell
                    q.append(n)
        if target is None:
            return []
        path = []
        while target is not None:
            path.append(target)
            target = parent[target]
        return path

    def _key(self, game):
        return (game.positions[0], game.positions[1], game.turn,
                frozenset(game.h_walls), frozenset(game.v_walls))