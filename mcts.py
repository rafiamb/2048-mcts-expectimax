"""
Monte Carlo Tree Search (MCTS) Agent for 2048 (3×3)

This implementation is used to evaluate how MCTS performs under
limited per-move time budgets when compared to a heuristic-based
Expectimax agent.

Key Features:
- UCB-based tree policy with tunable exploration constant
- Predicate-based rollout evaluation (empty tiles, monotonicity,
  max tile in corner)
- Controlled rollout variance by forcing the most probable tile (2)
- Time-limited search per move
"""

import time
import math
import random
from game2048 import Game2048 

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state                  # Game state at this node
        self.parent = parent                # Parent node
        self.children = []                  # Expanded child node
        self.value = 0.0                    # Accumulated rollout value
        self.visits = 0                     # Visit count
        self.remaining_actions = state.get_actions() if not state.is_terminal() else []
        self.action = action                # Action taken from parent

    def is_fully_expanded(self):
        return len(self.remaining_actions) == 0

    def best_child(self, c=1.5):
        """Selects child using UCB (exploit + explore)."""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                choices_weights.append(float('inf'))
            else:
                exploit = child.value / child.visits
                explore = math.sqrt(c * math.log(self.visits + 1) / child.visits)
                choices_weights.append(exploit + explore)
        return self.children[choices_weights.index(max(choices_weights))]

# Predicate weights updated across rollouts to bias evaluation
pred_history = {
    'max_in_corner': 1.0,
    'empty_count': 1.0,
    'monotonic_rows': 1.0,
    'monotonic_cols': 1.0
}

def board_predicates(board):
    """Extracts heuristic predicates used during rollouts."""
    size = len(board)
    preds = {}
    max_tile = max(max(row) for row in board)
    preds['max_in_corner'] = int(max_tile in (board[0][0], board[0][-1], board[-1][0], board[-1][-1]))
    preds['empty_count'] = sum(cell == 0 for row in board for cell in row)
    row_mono = sum(all(row[i] <= row[i+1] for i in range(size-1)) for row in board)
    col_mono = sum(all(board[i][j] <= board[i+1][j] for i in range(size-1)) for j in range(size))
    preds['monotonic_rows'] = row_mono
    preds['monotonic_cols'] = col_mono
    return preds

def predicate_score(board):
    preds = board_predicates(board)
    score = 0
    for key, value in preds.items():
        score += value * pred_history.get(key, 1.0)
    return score

def select(node):
    while node.is_fully_expanded() and not node.state.is_terminal():
        node = node.best_child()
    return node

def expand(node):
    if node.remaining_actions:
        action = node.remaining_actions.pop()
        next_state = node.state.successor(action)
        child = Node(next_state, parent=node, action=action)
        node.children.append(child)
        return child
    return node

def rollout(node):
    """
    Performs a rollout using predicate-based evaluation.
    Randomness is reduced by forcing the most probable tile spawn (2).
    """
    state = node.state
    size = len(state.board)

    while not state.is_terminal():
        actions = state.get_actions()
        best_score = -1
        best_moves = []

        for action in actions:
            # Simulate without random tile so we can force a deterministic 2
            # for lookahead scoring only — avoids double-spawning.
            next_board, _ = state._simulate_move(action, simulate_random_tile=False)
            empty = [(i, j) for i in range(size) for j in range(size) if next_board[i][j] == 0]
            if empty:
                fi, fj = random.choice(empty)
                next_board[fi][fj] = 2
            score = predicate_score(next_board)
            if score > best_score:
                best_score = score
                best_moves = [action]
            elif score == best_score:
                best_moves.append(action)

        action = random.choice(best_moves)
        # Apply the chosen move with a real (random) tile spawn
        state = state.successor(action)

    # Update historical predicate scores
    final_score = state.score
    preds = board_predicates(state.board)
    for key, value in preds.items():
        pred_history[key] = 0.9 * pred_history.get(key, 1.0) + 0.1 * final_score

    return final_score

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent

def mcts_policy(time_limit=1.0):
    """
    Returns a policy function that runs MCTS for a fixed time per move.
    Used in experimental comparison against Expectimax.
    """
    def policy(position):
        root = Node(position)
        f_time = time.time() + time_limit
        max_iter_time = 0.0

        while time.time() < f_time:
            iter_start = time.time()
            node = select(root)
            if not node.state.is_terminal():
                node = expand(node)
            result = rollout(node)
            backpropagate(node, result)
            iter_time = time.time() - iter_start
            max_iter_time = max(max_iter_time, iter_time)

        elapsed = time.time() - (f_time - time_limit)
        if elapsed > time_limit + max_iter_time:
            print(f"WARNING: MCTS exceeded time limit! Time taken: {elapsed:.4f} s, limit: {time_limit} s")
        print(f"Max iteration time this move: {max_iter_time:.6f} s, total time: {elapsed:.4f} s")

        if not root.children:
            return random.choice(position.get_actions())
        best_child_node = max(root.children, key=lambda c: c.visits)
        return best_child_node.action

    return policy
