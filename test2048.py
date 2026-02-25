"""
CPSC 4740 Final Project — Test Script

Game:
2048 (3×3 variant, with tile limit of 2048). Agents play independent games 
and are compared by final score.

Research Question:
How does MCTS compare to shallow Expectimax in 2048 under limited
computational budgets?

Results:
This script shows my winning percentage over 100 games for an MCTS agent
limited to 0.1s/move playing against a depth-4 Expectimax agent with a 
simple heuristic evaluation. Simply type "pypy3 test_2048" into the terminal.

To reproduce extended results, increase the MCTS per-move time limit
(e.g., 0.25s, 0.5s, or 1.0s) by modifying the mcts_time parameter in main.
However, a pre-produced result log can be found in readme.

Conclusion:
Shallow Expectimax is more reliable under tight compute, but MCTS
outperforms Expectimax when given sufficient search time, indicating
that MCTS performance is sensitive to computational budget rather
than inherently suboptimal.
"""

import time
from mcts import mcts_policy
from game2048 import Game2048
from expectimax import ExpectimaxAgent
import math
import statistics

def play_game(policy_fn, game_class, verbose=False):
    """Plays one game using the given policy and returns the final score."""
    game = Game2048()
    state = game.initial_state()
    while not state.is_terminal():
        move = policy_fn(state)
        state = state.successor(move)
        if verbose:
            print(state.board)
            print('-'*15)
    return state.score

def conf_interval(scores, confidence=0.95):
    """Returns a confidence interval for a list of scores."""
    n = len(scores)
    mean = sum(scores)/n
    std = statistics.stdev(scores)
    z = 1.96 if confidence == 0.95 else 1.0
    margin = z * (std / math.sqrt(n))
    return mean - margin, mean + margin

def evaluate_mcts_vs_expectimax(num_games=100, mcts_time=0.1):
    """
    Runs a head-to-head evaluation of MCTS vs Expectimax.

    Each agent plays num_games independent games of 2048.
    A win is counted when MCTS achieves a higher final score.

    Args:
        num_games: number of games per agent
        mcts_time: time limit (seconds) per move for MCTS
    """
    mcts_fn = mcts_policy(time_limit=mcts_time)
    expectimax = ExpectimaxAgent()

    mcts_scores = []
    ex_scores = []

    for i in range(num_games):
        # Play MCTS game
        mcts_score = play_game(mcts_fn, Game2048, verbose=False)
        mcts_scores.append(mcts_score)

        # Play Expectimax game
        game = Game2048()
        state = game.initial_state()
        while not state.is_terminal():
            move = expectimax.select_action(state)
            state = state.successor(move)
        ex_scores.append(state.score)

        print(f"Game {i+1}: MCTS={mcts_score}, Expectimax={ex_scores[-1]}")

    # Summary statistics
    avg_mcts = sum(mcts_scores)/num_games
    avg_ex = sum(ex_scores)/num_games
    wins = sum(m > e for m, e in zip(mcts_scores, ex_scores))

    var_mcts = statistics.variance(mcts_scores)
    var_ex = statistics.variance(ex_scores)
    std_mcts = statistics.stdev(mcts_scores)
    std_ex = statistics.stdev(ex_scores)
    ci_mcts = conf_interval(mcts_scores)
    ci_ex = conf_interval(ex_scores)

    print("\n--- Summary ---")
    print(f"MCTS: avg={avg_mcts:.2f}, std={std_mcts:.2f}, var={var_mcts:.2f}, 95% CI={ci_mcts}")
    print(f"Expectimax: avg={avg_ex:.2f}, std={std_ex:.2f}, var={var_ex:.2f}, 95% CI={ci_ex}")
    print(f"MCTS won {wins}/{num_games} games")

# Entry point
if __name__ == "__main__":
    # Default run used for grading (short runtime)
    evaluate_mcts_vs_expectimax(num_games=100, mcts_time=0.5)
