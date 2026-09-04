"""
run_vision.py - Launcher for vision-only mode
"""

import sys

try:
    import ai_drive_game_vision as game
except ImportError:
    print("ERROR: ai_drive_game_vision not found")
    sys.exit(1)

try:
    import gemini_connector_vision_only as connector
except ImportError:
    print("ERROR: gemini_connector_vision_only not found")
    sys.exit(1)

ai_mode = "--human" not in sys.argv

if ai_mode:
    print("Starting AI mode...")
    connector.start_connector()
else:
    print("Starting human mode...")

game.main(ai_mode=ai_mode)
