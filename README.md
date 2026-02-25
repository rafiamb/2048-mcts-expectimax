# 2048 AI Agents — MCTS vs Expectimax

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

A 3×3 variant of 2048 with two AI agents, Monte Carlo Tree Search (MCTS) and Expectimax, compared under limited computational budgets. Includes a live terminal visualizer to watch either agent play.

This project was implemented for CPSC 474: Computational Intelligence for Games at Yale University.

---

## Overview

This project investigates the following research question:

> *How does an MCTS agent compare to a heuristic-based Expectimax agent in 2048 under limited per-move time budgets?*

The key finding is that shallow Expectimax is more reliable under tight compute, but MCTS can outperform it when given sufficient search time, suggesting MCTS performance is sensitive to computational budget rather than inherently suboptimal.

---

## Project Structure

```
.
├── game2048.py      # Core game logic (3×3 board, move simulation, state)
├── expectimax.py    # Heuristic-based Expectimax agent (depth=4)
├── mcts.py          # MCTS agent with UCB tree policy and predicate rollouts
├── visualize.py     # Terminal visualizer — watch the AI play live
└── test_2048.py     # Head-to-head evaluation script
```

No external dependencies — standard library only.

---

## Agents

### Expectimax
A fixed-depth search agent that models tile spawns as chance nodes.

- Search depth: 4
- Chance nodes branch over all empty cells × tile values (2 with p=0.9, 4 with p=0.1)
- Heuristic evaluation combines:
  - Current score
  - Number of empty tiles (weight 100)
  - Max tile in a corner (weight 1.5)
  - Board monotonicity
  - Tile smoothness

### MCTS
A time-limited Monte Carlo Tree Search agent.

- UCB-based tree policy with exploration constant c = 1.5
- Predicate-based rollout evaluation (empty tiles, monotonicity, corner bonus)
- Rollout predicates are updated across moves via exponential moving average
- Time budget is configurable per move (default: 0.3s)

---

## Usage

### Watch an agent play (terminal visualizer)

```bash
python visualize.py                        # Expectimax (default)
python visualize.py --agent mcts           # MCTS
python visualize.py --agent both           # Both agents, then compare scores
python visualize.py --speed fast           # 0.1s delay between moves
python visualize.py --speed slow           # 0.8s delay between moves
python visualize.py --mcts-time 0.5        # Give MCTS 0.5s per move
```

### Run the head-to-head evaluation

```bash
python test_2048.py       # 100 games each, 0.5s/move for MCTS
```

Or with PyPy for faster runs:

```bash
pypy3 test_2048.py
```

To adjust the number of games or MCTS time budget, edit the parameters at the bottom of `test_2048.py`:

```python
evaluate_mcts_vs_expectimax(num_games=100, mcts_time=0.5)
```

---

## Results

Each agent played 100 independent games. A win is counted when an agent achieves a higher final score.

| Time/move | MCTS avg | Expectimax avg | MCTS wins |
|-----------|----------|----------------|-----------|
| 0.25s     | 779      | 818            | 46 / 100  |
| 0.5s      | 836      | 764            | 57 / 100  |
| 1.0s      | 840      | 877            | 41 / 100  |

95% confidence intervals and variance are printed by the test script. Results show high variance across all conditions — a consequence of the stochastic tile spawns on a small 3×3 board.

---

## Implementation Notes

- The game uses a 3×3 board instead of the standard 4×4, which makes reaching 2048 significantly harder and speeds up evaluation.
- Reaching a 2048 tile yields a +100,000 score bonus.
- MCTS rollouts use `_simulate_move` (no random tile) for lookahead scoring to avoid double-spawning tiles, then apply `successor` (with random tile) for the actual transition.
