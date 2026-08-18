# Barricade

Jeu de stratégie 2 joueurs sur grille 9×9 (type Quoridor). Course vers le côté opposé du plateau, 10 barricades chacun, saut par-dessus l'adversaire.

## Lancer

```bash
python run.py                    # http://127.0.0.1:8000
python run.py --depth 6 --time 1 # AI params (defaut: depth 8, 2s)
python -m unittest discover tests -v   # tests
```

Modes via "New game" : PvP, vs AI (greedy), vs AI (kernel), vs AI (alpha-beta), vs AI (MCTS), AI vs AI.
Un minuteur est optionnel : taper « mode temps » dans la boîte de dialogue (0 = sans).

## Structure

- `barricade/game.py` — règles du jeu (mouvements, barricades, saut, victoire, horloge). Pur Python, sans dépendance.
- `barricade/engine/` — IA. Interface `Engine.choose_move(game)`.
  - `alphabeta.py` — negamax alpha-beta Python : deepening itératif, table de transposition, murs ciblés sur le chemin adverse.
  - `kernel.py` + `numba_engine.py` — recherche compilée numba (masques 81 bits en paires hi/lo uint64) : ~2,4M nœuds/s, depth ~10-11 en 2s. IA par défaut si numba est installé.
  - `mcts.py` — MCTS compilé numba (UCT, un nœud par itération, playouts course BFS-greedy, valeur de feuille normalisée `tanh(score/30)`). Expérimental : correct mais plus faible que le kernel à temps égal (limite classique du MCTS à playouts légers sur ce jeu).
  - `evaluate.py` — évaluation : distance à l'arrivée (différence), ratio de confiance pondéré par les barricades adverses, avantage de murs restants.
  - `bitboard.py` — primitives bitboard partagées (flood, `wall_ok`).
  - `greedy.py`, `random.py` — placeholders.
- `barricade/web/` — serveur local + API JSON + UI HTML/Canvas.
- `tests/` — tests unitaires.

## Feuille de route IA

1. ✅ Alpha-beta (negamax) : deepening itératif, table de transposition, ordre des coups, murs ciblés.
2. ✅ Évaluation : distance à l'arrivée → différence avec l'adversaire, ratio de confiance pondéré par les barricades adverses.
3. ✅ Accélération : bitboards (`engine/bitboard.py`) puis kernel numba (`engine/kernel.py`, `numba_engine.py`) — ~2,4M nœuds/s, depth ~10-11 en 2s (vs ~60k nœuds/s Python).
4. ✅ MCTS numba expérimental (`engine/mcts.py`).
5. ⏳ Réseau de neurones (supervisé ou RL) si besoin.