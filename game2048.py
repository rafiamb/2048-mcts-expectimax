import random

class Game2048:
    """
    2048 Game (3×3 Variant) Implementation.

    This implementation of the 2048 game uses AI-assisted code generation.
    The game involves combining numbered tiles to reach a tile value of 2048 while maximizing the total score.
    The game ends when no legal moves remain or when a tile value of 2048 is achieved.

    The code is structured to allow for the simulation of game moves, generating random tiles, and computing the score.
    """
    size = 3 # Board size

    def initial_state(self):
        """Return the initial state of the game with two random tiles"""
        board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        board = self._add_random_tile(board)
        board = self._add_random_tile(board)
        return Game2048.State(board, 0)

    def _add_random_tile(self, board):
        """Add a random tile (2 or 4) to an empty space on the board."""
        empty = [(i, j) for i in range(self.size) for j in range(self.size) if board[i][j] == 0]
        if empty:
            i, j = random.choice(empty)
            board[i][j] = 2 if random.random() < 0.9 else 4
        return board

    class State:
        def __init__(self, board, score):
            """Initialize the game state with a deep copy of the board and the current score."""
            self.board = [row[:] for row in board]
            self.score = score

        def get_actions(self):
            """Return a list of legal moves available from the current state."""
            actions = []
            for move in ['up', 'down', 'left', 'right']:
                new_board, _ = self._simulate_move(move, simulate_random_tile=False)
                if not self._boards_equal(new_board, self.board):
                    actions.append(move)
            return actions

        def successor(self, move):
            """Return the new state resulting from applying the given move."""
            new_board, gained_score = self._simulate_move(move, simulate_random_tile=True)
            return Game2048.State(new_board, self.score + gained_score)

        def is_terminal(self):
            # Terminal if no moves left OR any tile >= 2048
            for row in self.board:
                if any(tile >= 2048 for tile in row):
                    return True
            return len(self.get_actions()) == 0

        def payoff(self):
            """Return the payoff of the current state (bonus if 2048 is achieved)."""
            for row in self.board:
                if any(tile >= 2048 for tile in row):
                    return 100000 + self.score  # Bonus for reaching 2048
            return self.score

        def actor(self):
            """Return the actor (always 0; single-player game)."""
            return 0

        def _simulate_move(self, direction, simulate_random_tile=True):
            """Simulate the effect of a move (without modifying the current state)."""
            size = len(self.board)
            score_gain = 0
            new_board = [row[:] for row in self.board]

            def merge(line):
                """Merge tiles in a given line (row or column) and calculate score gain."""
                nonlocal score_gain
                tiles = [x for x in line if x != 0]
                merged = []
                i = 0
                while i < len(tiles):
                    if i+1 < len(tiles) and tiles[i] == tiles[i+1]:
                        merged.append(tiles[i]*2)
                        score_gain += tiles[i]*2
                        i += 2
                    else:
                        merged.append(tiles[i])
                        i += 1
                # pad with zeros
                merged += [0]*(len(line) - len(merged))
                return merged

            if direction == 'left':
                for r in range(size):
                    new_board[r] = merge(new_board[r])

            elif direction == 'right':
                for r in range(size):
                    new_board[r] = merge(new_board[r][::-1])[::-1]

            elif direction == 'up':
                for c in range(size):
                    col = [new_board[r][c] for r in range(size)]
                    merged = merge(col)
                    for r in range(size):
                        new_board[r][c] = merged[r]

            elif direction == 'down':
                for c in range(size):
                    col = [new_board[r][c] for r in range(size)][::-1]
                    merged = merge(col)[::-1]
                    for r in range(size):
                        new_board[r][c] = merged[r]

            if simulate_random_tile:
                empty = [(i, j) for i in range(size) for j in range(size) if new_board[i][j] == 0]
                if empty:
                    i, j = random.choice(empty)
                    new_board[i][j] = 2 if random.random() < 0.9 else 4

            return new_board, score_gain

        def _boards_equal(self, b1, b2):
            """Helper function to compare two boards."""
            for r in range(len(b1)):
                for c in range(len(b1[r])):
                    if b1[r][c] != b2[r][c]:
                        return False
            return True

        def __str__(self):
            """Return a string representation of the current game state."""
            lines = [f"Score: {self.score}"]
            for row in self.board:
                lines.append(' '.join(f"{val:4}" if val != 0 else "   ." for val in row))
            return '\n'.join(lines)
