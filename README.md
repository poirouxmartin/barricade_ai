# barricade_ai

Barricade as played on barricade.gg: two players, a 9x9 grid, ten barricades each, jumping over
the opponent, first to the opposite side wins. No dice, so a position has an exact value and a
program can search for it.

Project page: [martinpoiroux.com/en/projects/barricade](https://martinpoiroux.com/en/projects/barricade/)

## Run

```bash
python run.py                    # http://127.0.0.1:8000
python run.py --depth 6 --time 1 # AI params (defaut: depth 8, 2s)
python -m unittest discover tests -v   # tests
```

Modes via "New game": PvP, vs AI (greedy), vs AI (kernel), vs AI (alpha-beta), vs AI (MCTS), AI vs AI.
A timer is optional: type "time mode" in the dialog box (0 = none).

## Structure

- `barricade/game.py`: game rules (moves, barricades, jump, victory, clock). Pure Python, no dependency.
- `barricade/engine/`: AI. Interface `Engine.choose_move(game)`.
  - `alphabeta.py`: Python negamax alpha-beta: iterative deepening, transposition table, walls targeted on the enemy path. Fallback without numba; in the UI the "alpha-beta" option uses the compiled kernel.
  - `kernel.py` + `numba_engine.py`: numba-compiled search (81-bit masks as hi/lo uint64 pairs): ~2.5-2.8M nodes/s, depth 12 in 2s. Default AI if numba is installed.
  - `mcts.py`: numba-compiled MCTS (UCT, one node per iteration, BFS-greedy race playouts, normalised leaf value `tanh(score/30)`). Experimental: correct but weaker than the kernel at equal time (classic MCTS limitation with light playouts on this game).
  - `evaluate.py`: evaluation: distance to goal (difference), confidence reduced only when the opponent holds a barricade surplus, remaining-wall advantage.
  - `bitboard.py`: shared bitboard primitives (flood, `wall_ok`).
  - `greedy.py`, `random.py`: placeholders.
- `barricade/web/`: local server + JSON API + HTML/Canvas UI.
- `tests/`: unit tests.

## AI roadmap

1. ✅ Alpha-beta (negamax): iterative deepening, transposition table, move ordering, walls targeted on enemy path.
2. ✅ Evaluation: distance to goal -> difference with opponent, confidence reduced only on enemy barricade surplus, wall advantage.
3. ✅ Acceleration: bitboards (`engine/bitboard.py`) then numba kernel (`engine/kernel.py`, `numba_engine.py`) -- ~2.5-2.8M nodes/s, depth 12 in 2s (vs ~60k nodes/s Python). Killer moves + LMR + aspiration + TT move ordering.
4. ✅ Experimental numba MCTS (`engine/mcts.py`).
5. ⏳ Neural network (supervised or RL) if needed.

## License

MIT
