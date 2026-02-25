"""
visualize.py — Watch the 2048 AI play in your terminal.

Usage:
    python visualize.py                  # Expectimax (default)
    python visualize.py --agent mcts     # MCTS agent
    python visualize.py --agent both     # Play one game each, side by side summary
    python visualize.py --speed fast     # Faster playback (0.1s delay)
    python visualize.py --speed slow     # Slower playback (0.8s delay)
"""

import time
import os
import sys
import argparse
from game2048 import Game2048
from expectimax import ExpectimaxAgent
from mcts import mcts_policy

# Tile colors (ANSI)
TILE_COLORS = {
    0:    ("\033[90m",   "\033[0m"),   # dark gray
    2:    ("\033[97m",   "\033[0m"),   # white
    4:    ("\033[93m",   "\033[0m"),   # yellow
    8:    ("\033[33m",   "\033[0m"),   # orange-ish
    16:   ("\033[91m",   "\033[0m"),   # light red
    32:   ("\033[31m",   "\033[0m"),   # red
    64:   ("\033[95m",   "\033[0m"),   # magenta
    128:  ("\033[94m",   "\033[0m"),   # blue
    256:  ("\033[96m",   "\033[0m"),   # cyan
    512:  ("\033[92m",   "\033[0m"),   # green
    1024: ("\033[32m",   "\033[0m"),   # dark green
    2048: ("\033[1;93m", "\033[0m"),   # bold bright yellow
}

def tile_color(val):
    return TILE_COLORS.get(val, ("\033[1;97m", "\033[0m"))

def render_board(state, agent_name, move_num, last_move, elapsed):
    """Renders the board with colors and stats to stdout."""
    board = state.board
    size = len(board)

    # Top border
    h_line = "┼" + ("─" * 7 + "┼") * size
    top     = "┌" + ("─" * 7 + "┬") * (size - 1) + "─" * 7 + "┐"
    bottom  = "└" + ("─" * 7 + "┴") * (size - 1) + "─" * 7 + "┘"
    mid     = "├" + ("─" * 7 + "┼") * (size - 1) + "─" * 7 + "┤"

    lines = []
    lines.append(f"\n  🎮  2048  —  {agent_name}")
    lines.append(f"  Move: {move_num:<4}  Score: {state.score:<6}  Last: {last_move or '—':<6}  Time: {elapsed:.1f}s")
    lines.append("  " + top)

    for r, row in enumerate(board):
        row_str = "  │"
        for val in row:
            color, reset = tile_color(val)
            cell = f"{val:^5}" if val != 0 else "  ·  "
            row_str += f" {color}{cell}{reset} │"
        lines.append(row_str)
        if r < size - 1:
            lines.append("  " + mid)

    lines.append("  " + bottom)
    return "\n".join(lines)

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def play_visual(agent_name, policy_fn, speed=0.3):
    """Plays one full game with live terminal rendering."""
    game = Game2048()
    state = game.initial_state()
    move_num = 0
    start = time.time()
    last_move = None

    while not state.is_terminal():
        clear()
        print(render_board(state, agent_name, move_num, last_move, time.time() - start))
        time.sleep(speed)

        last_move = policy_fn(state)
        state = state.successor(last_move)
        move_num += 1

    # Final frame
    clear()
    print(render_board(state, agent_name, move_num, last_move, time.time() - start))

    max_tile = max(v for row in state.board for v in row)
    won = max_tile >= 2048
    result = "🏆 Reached 2048!" if won else "💀 Game over"
    print(f"\n  {result}  |  Final score: {state.score}  |  Moves: {move_num}\n")
    return state.score

# CLI 
def main():
    parser = argparse.ArgumentParser(description="Watch 2048 AI agents play.")
    parser.add_argument("--agent", choices=["expectimax", "mcts", "both"], default="expectimax")
    parser.add_argument("--speed", choices=["slow", "normal", "fast"], default="normal")
    parser.add_argument("--mcts-time", type=float, default=0.3,
                        help="MCTS time budget per move in seconds (default: 0.3)")
    args = parser.parse_args()

    speeds = {"slow": 0.8, "normal": 0.3, "fast": 0.1}
    delay = speeds[args.speed]

    agents = []
    if args.agent in ("expectimax", "both"):
        ex = ExpectimaxAgent()
        agents.append(("Expectimax (depth=4)", ex.select_action))
    if args.agent in ("mcts", "both"):
        agents.append((f"MCTS ({args.mcts_time}s/move)", mcts_policy(time_limit=args.mcts_time)))

    scores = {}
    for name, fn in agents:
        if len(agents) > 1:
            input(f"\n  Press Enter to watch {name} play...")
        score = play_visual(name, fn, speed=delay)
        scores[name] = score

    if len(scores) > 1:
        print("  ── Head-to-head ──────────────────────")
        for name, score in scores.items():
            print(f"  {name:<30} {score}")
        winner = max(scores, key=scores.get)
        print(f"\n  Winner: {winner} 🎉\n")

if __name__ == "__main__":
    main()
