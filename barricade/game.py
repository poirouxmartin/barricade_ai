"""Barricade game engine (rules only, no UI/AI)."""

import time
from collections import deque

SIZE = 9
ROWS = COLS = SIZE
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
START = {0: (ROWS - 1, SIZE // 2), 1: (0, SIZE // 2)}
N_WALLS = 10


def goal_row(player):
    return 0 if player == 0 else ROWS - 1


class Barricade:
    """A Barricade board.

    Walls are stored as coordinate pairs on the 8x8 grid of wall slots.
      H wall (r, c): blocks vertical movement between rows r/r+1 at columns c and c+1.
      V wall (r, c): blocks horizontal movement between columns c/c+1 at rows r and r+1.
    Crossing (H and V sharing (r, c)) is illegal.

    Optional clock: `time_control` seconds per player for the whole game.
    Each move's wall-clock duration is charged to the mover; a player whose
    clock runs out loses the game (`game_over_reason == "time"`).
    """

    def __init__(self, time_control=None):
        self.positions = [START[0], START[1]]
        self.walls_left = [N_WALLS, N_WALLS]
        self.h_walls = set()
        self.v_walls = set()
        self.turn = 0
        self.winner = None
        self.game_over_reason = None
        self.history = []
        self.time_control = time_control
        self.time_left = [time_control, time_control] if time_control else None
        self._last_move_at = None

    # ----- edges & connectivity -----

    def h_edge_blocked(self, r, c):
        """Vertical edge between cells (r, c) and (r+1, c)."""
        if not (0 <= r < ROWS - 1 and 0 <= c < COLS):
            return True
        return (r, c - 1) in self.h_walls or (r, c) in self.h_walls

    def v_edge_blocked(self, r, c):
        """Horizontal edge between cells (r, c) and (r, c+1)."""
        if not (0 <= r < ROWS and 0 <= c < COLS - 1):
            return True
        return (r - 1, c) in self.v_walls or (r, c) in self.v_walls

    def neighbors(self, r, c):
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if dr and self.h_edge_blocked(min(r, nr), c):
                continue
            if dc and self.v_edge_blocked(r, min(c, nc)):
                continue
            yield (nr, nc)

    def has_path(self, player):
        goal = goal_row(player)
        seen = {self.positions[player]}
        stack = [self.positions[player]]
        while stack:
            r, c = stack.pop()
            if r == goal:
                return True
            for n in self.neighbors(r, c):
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        return False

    def dist_to_goal(self, player):
        goal = goal_row(player)
        if self.positions[player][0] == goal:
            return 0
        q = deque([(self.positions[player], 0)])
        seen = {self.positions[player]}
        while q:
            (r, c), d = q.popleft()
            for n in self.neighbors(r, c):
                if n[0] == goal:
                    return d + 1
                if n not in seen:
                    seen.add(n)
                    q.append((n, d + 1))
        return float("inf")

    # ----- moves -----

    def legal_moves(self, player):
        pr, pc = self.positions[player]
        opp = self.positions[1 - player]
        moves = []
        for dr, dc in DIRS:
            nr, nc = pr + dr, pc + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if dr and self.h_edge_blocked(min(pr, nr), pc):
                continue
            if dc and self.v_edge_blocked(pr, min(pc, nc)):
                continue
            if (nr, nc) == opp:
                rr, cc = nr + dr, nc + dc  # straight behind opponent
                straight = (0 <= rr < ROWS and 0 <= cc < COLS
                            and not (dr and self.h_edge_blocked(min(nr, rr), cc))
                            and not (dc and self.v_edge_blocked(rr, min(nc, cc))))
                if straight:
                    moves.append((rr, cc))
                else:
                    for ar, ac in ((opp[0] + dc, opp[1] + dr), (opp[0] - dc, opp[1] - dr)):
                        if not (0 <= ar < ROWS and 0 <= ac < COLS):
                            continue
                        if dr and self.v_edge_blocked(opp[0], min(opp[1], ac)):
                            continue
                        if dc and self.h_edge_blocked(min(opp[0], ar), opp[1]):
                            continue
                        moves.append((ar, ac))
            else:
                moves.append((nr, nc))
        return list(dict.fromkeys(moves))

    # ----- walls -----

    def _place_wall(self, ori, r, c):
        (self.h_walls if ori == "H" else self.v_walls).add((r, c))

    def _remove_wall(self, ori, r, c):
        (self.h_walls if ori == "H" else self.v_walls).discard((r, c))

    def wall_valid(self, ori, r, c, check_path=True):
        if not (0 <= r < ROWS - 1 and 0 <= c < COLS - 1):
            return False
        walls = self.h_walls if ori == "H" else self.v_walls
        other = self.v_walls if ori == "H" else self.h_walls
        if (r, c) in walls or (r, c) in other:
            return False
        if ori == "H":
            if (r, c - 1) in walls or (r, c + 1) in walls:
                return False
        else:
            if (r - 1, c) in walls or (r + 1, c) in walls:
                return False
        if not check_path:
            return True
        self._place_wall(ori, r, c)
        ok = self.has_path(0) and self.has_path(1)
        self._remove_wall(ori, r, c)
        return ok

    def legal_walls(self, player):
        if self.walls_left[player] <= 0:
            return []
        out = []
        for r in range(ROWS - 1):
            for c in range(COLS - 1):
                if self.wall_valid("H", r, c):
                    out.append(("H", r, c))
                if self.wall_valid("V", r, c):
                    out.append(("V", r, c))
        return out

    def legal_actions(self, player=None):
        if player is None:
            player = self.turn
        if self.winner is not None:
            return []
        return [("move", m) for m in self.legal_moves(player)] \
            + [("wall", w) for w in self.legal_walls(player)]

    # ----- apply -----

    def apply(self, action, check=True):
        kind, arg = action
        if self.winner is not None:
            raise ValueError("game over")
        player = self.turn
        if kind == "move":
            if check and arg not in self.legal_moves(player):
                raise ValueError("illegal move")
            self.positions[player] = arg
            self.history.append(("move", player, arg))
            if arg[0] == goal_row(player):
                self.winner = player
        elif kind == "wall":
            ori, r, c = arg
            if check and (self.walls_left[player] <= 0 or not self.wall_valid(ori, r, c)):
                raise ValueError("illegal wall")
            self._place_wall(ori, r, c)
            self.walls_left[player] -= 1
            self.history.append(("wall", player, arg))
        else:
            raise ValueError("unknown action")
        if self.time_control and self.winner is None:
            self._tick_clock(player)
        if self.winner is not None and self.game_over_reason is None:
            self.game_over_reason = "goal"
        if self.winner is None:
            self.turn = 1 - player

    def _tick_clock(self, player):
        """Charge the elapsed wall-clock to the mover; first move starts the clock."""
        now = time.monotonic()
        if self._last_move_at is not None:
            self.time_left[player] -= now - self._last_move_at
            if self.time_left[player] <= 0:
                self.time_left[player] = 0.0
                self.winner = 1 - player
                self.game_over_reason = "time"
        self._last_move_at = now

    def clone(self):
        g = Barricade.__new__(Barricade)
        g.positions = list(self.positions)
        g.walls_left = list(self.walls_left)
        g.h_walls = set(self.h_walls)
        g.v_walls = set(self.v_walls)
        g.turn = self.turn
        g.winner = self.winner
        g.game_over_reason = self.game_over_reason
        g.history = list(self.history)
        g.time_control = self.time_control
        g.time_left = list(self.time_left) if self.time_left else None
        g._last_move_at = self._last_move_at
        return g

    def to_dict(self):
        return {
            "positions": [list(p) for p in self.positions],
            "turn": self.turn,
            "walls_left": self.walls_left,
            "h_walls": [list(w) for w in sorted(self.h_walls)],
            "v_walls": [list(w) for w in sorted(self.v_walls)],
            "winner": self.winner,
            "game_over": self.winner is not None,
            "game_over_reason": self.game_over_reason,
            "time_control": self.time_control,
            "time_left": list(self.time_left) if self.time_left else None,
            "last_action": self.history[-1] if self.history else None,
            "legal_moves": [list(m) for m in self.legal_moves(self.turn)],
            "legal_walls": [list(w) for w in self.legal_walls(self.turn)],
        }