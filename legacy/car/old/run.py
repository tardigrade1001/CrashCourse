"""
run_fixed.py — Launch AI Drive game with fixed Gemini connector
Usage:
    python run_fixed.py              (game + AI connector)
    python run_fixed.py --human      (game only, no AI)
"""

import sys
import os

# Ensure we're in the right directory
# Adjust this path to match where your game files actually live
GAME_DIR = os.path.dirname(os.path.abspath(__file__))

# Try import the game
try:
    # If running from the same directory as ai_drive_game.py
    import ai_drive_game as game
except ImportError:
    print("[run] ERROR: Could not import ai_drive_game. Make sure ai_drive_game.py is in the same directory.")
    sys.exit(1)

# Try import the fixed connector
try:
    import gemini_connector_fixed as connector
except ImportError:
    print("[run] ERROR: Could not import gemini_connector_fixed. Make sure gemini_connector_fixed.py is in the same directory.")
    sys.exit(1)


def main():
    ai_mode = "--human" not in sys.argv
    
    if ai_mode:
        print("[run] Starting game in AI mode...")
        print("[run] Gemini 3.1 Flash Live will control the car")
        print("[run] (Ensure GEMINI_API_KEY is set in gemini_connector_fixed.py)")
        connector.start_connector()
    else:
        print("[run] Starting game in HUMAN mode (arrow keys or A/D to move)")
    
    # Game runs on main thread (pygame requirement)
    game.main(ai_mode=ai_mode)


if __name__ == "__main__":
    main()