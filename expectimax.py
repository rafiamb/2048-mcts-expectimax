"""
Expectimax Agent for 2048 (3×3)

This agent serves as a heuristic-based baseline for comparison with
MCTS. It uses a fixed search depth and a handcrafted evaluation
function, making it computationally efficient under tight computational
budgets.
"""

from game2048 import Game2048

class ExpectimaxAgent:
    """
    Heuristic-based Expectimax agent for 2048.

    Uses fixed-depth expectimax search where:
    - Max nodes represent player moves
    - Chance nodes represent random tile spawns (2 with p=0.9, 4 with p=0.1)
    """
    def __init__(self, depth=4, heuristic=None):
        self.depth = depth
        self.h = heuristic

    def select_action(self, state):
        """Selects the best move from the current state."""
        _, move = self.expectimax(state, self.depth, is_max=True)
        return move

    def expectimax(self, state, depth, is_max):
        """
        Recursive expectimax search.
        Returns (expected_value, best_action).
        """
        if depth == 0 or state.is_terminal():
            value = self.h.evaluate(state) if self.h else self.evaluate(state)
            return value, None

        actions = state.get_actions()
        if not actions:
            value = self.h.evaluate(state) if self.h else self.evaluate(state)
            return value, None

        if is_max:
            # Player turn: choose action maximizing heuristic value
            best_value = -float('inf')
            best_move = None
            for move in actions:
                child_state = state.successor(move)
                value, _ = self.expectimax(child_state, depth-1, is_max=False)
                if value > best_value:
                    best_value = value
                    best_move = move
            return best_value, best_move
        else:
            # Chance node: expected value over all tile spawns
            empty_cells = [(i, j) for i in range(len(state.board)) for j in range(len(state.board[0])) if state.board[i][j] == 0]
            if not empty_cells:
                return self.h.evaluate(state) if self.h else self.evaluate(state), None

            expected_value = 0
            for (i, j) in empty_cells:
                for tile, prob in [(2, 0.9), (4, 0.1)]:
                    new_board = [row[:] for row in state.board]
                    new_board[i][j] = tile
                    child_state = type(state)(new_board, state.score)
                    value, _ = self.expectimax(child_state, depth-1, is_max=True)
                    expected_value += value * prob / len(empty_cells)
            return expected_value, None

    def evaluate(self, state):
        """
        Heuristic evaluation function combining:
        - Current score
        - Number of empty tiles
        - Max tile in a corner
        - Board monotonicity
        - Tile smoothness
        """
        board = state.board
        size = len(board)

        # Empty tiles
        empty_tiles = sum(1 for row in board for v in row if v == 0)

        # Corner bonus for largest tile
        max_tile = max(max(row) for row in board)
        corner_bonus = 0
        corners = [board[0][0], board[0][size-1], board[size-1][0], board[size-1][size-1]]
        if max_tile in corners:
            corner_bonus = max_tile * 2

        # Monotonicity penalty (lower is better)
        mono_score = 0
        for row in board:
            for i in range(size-1):
                if row[i] >= row[i+1]:
                    mono_score += row[i+1] - row[i]
        for c in range(size):
            for r in range(size-1):
                if board[r][c] >= board[r+1][c]:
                    mono_score += board[r+1][c] - board[r][c]
        mono_score = -mono_score  # smaller penalty is better

        # Smoothness penalty between adjacent tiles
        smoothness = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    continue
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if nr < size and nc < size and board[nr][nc] != 0:
                        smoothness -= abs(board[r][c] - board[nr][nc])

        # Weighted heuristic sum
        score_weight = 1.0
        empty_weight = 100.0
        corner_weight = 1.5
        mono_weight = 1.0
        smooth_weight = 1.0

        heuristic_value = (
            score_weight * state.score +
            empty_weight * empty_tiles +
            corner_weight * corner_bonus +
            mono_weight * mono_score +
            smooth_weight * smoothness
        )

        return heuristic_value
