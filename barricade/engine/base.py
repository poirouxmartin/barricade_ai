"""Engine interface. Future engines (alpha-beta, MCTS, NN) plug in here."""


class Engine:
    def choose_move(self, game):
        """Return a legal action for game.turn: ('move', (r,c)) or ('wall', ('H'|'V', r, c))."""
        raise NotImplementedError