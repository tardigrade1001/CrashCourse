"""
run_vision.py — Launch AI Drive in VISION-ONLY mode (FIXED)
"""

import sys

try:
    import ai_drive_game_vision as game
except ImportError:
    print("[run] ERROR: Could not import ai_drive_game_vision")
    sys.exit(1)

try:
    import gemini_connector_vision_only as connector
except ImportError:
    print("[run] ERROR: Could not import gemini_connector_vision_only")
    sys.exit(1)


def main():
    ai_mode = "--human" not in sys.argv
    
    if ai_mode:
        print("[run] VISION-ONLY mode (FIXED)")
        print("[run] Varied prompts, no aggressive resets")
        connector.start_connector()
    else:
        print("[run] Human mode")
    
    game.main(ai_mode=ai_mode)


if __name__ == "__main__":
    main()