"""
Gemini Connector - STRONG SYSTEM PROMPT
Forces model to read state and actually decide
"""

import asyncio
import time
import threading
import re

try:
    import ai_drive_game as game
    HAS_GAME = True
except ImportError:
    HAS_GAME = False

_KEY_FILE = r"F:\PC Files\Projects\car\API_KEY.txt"
try:
    GEMINI_API_KEY = open(_KEY_FILE).read().strip()
except FileNotFoundError:
    GEMINI_API_KEY = None

MODEL = "gemini-3.1-flash-live-preview"
FRAME_INTERVAL = 0.7

SYSTEM_PROMPT = """YOU ARE DRIVING A CAR IN A GAME.

READ THE GAME STATE TEXT CAREFULLY. IT TELLS YOU:
- Which lane your car is in
- Which lanes have obstacles approaching
- Which lanes are CLEAR

YOUR JOB:
1. Read the state text
2. Find the SAFEST lane (avoid obstacles)
3. If your current lane has an obstacle, MOVE TO A CLEAR LANE
4. If road is clear, stay in current lane

YOU MUST RESPOND WITH A LANE DECISION EVERY TIME.

Examples:
- State: "Car lane 2. Lane 2 has obstacle. Clear lanes: [1, 3, 4]" → LANE:1 or LANE:3 or LANE:4
- State: "Car lane 2. Road CLEAR" → LANE:2
- State: "Car lane 3. Lane 3 close. Lane 1 very close. Clear: [2, 4]" → LANE:2 or LANE:4

RESPOND WITH ONLY ONE:
LANE:1
LANE:2
LANE:3
LANE:4

NOTHING ELSE. Just the lane number."""


def _parse_lane(text):
    if not text:
        return None
    text = text.upper().strip()
    m = re.search(r'LANE[:\s]+([1-4])', text)
    if m:
        return int(m.group(1))
    m = re.search(r'\b([1-4])\b', text)
    if m:
        return int(m.group(1))
    return None


async def _one_session(client, types, n_decisions):
    config = {
        "response_modalities": ["AUDIO"],
        "output_audio_transcription": {},
        "system_instruction": SYSTEM_PROMPT,
    }

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        count = 0

        while count < n_decisions:
            if HAS_GAME and game.game_surface is None:
                await asyncio.sleep(0.1)
                continue

            t_start = time.time()

            # Send image
            if HAS_GAME:
                try:
                    jpeg_bytes = game.get_frame_jpeg(game.game_surface, quality=75)
                except Exception:
                    await asyncio.sleep(0.1)
                    continue
            else:
                from PIL import Image
                import io
                img = Image.new("RGB", (480, 360), color=(45, 45, 55))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                jpeg_bytes = buf.getvalue()

            await session.send_realtime_input(
                video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
            )

            # Send game state - CRITICAL
            if HAS_GAME:
                try:
                    with game._state_lock:
                        cur_lane = game._target_lane
                    state_text = game.get_game_state_text(cur_lane, game._obstacles)
                except Exception:
                    state_text = "Car in lane 2. Road CLEAR."
            else:
                state_text = "Car in lane 2. Road CLEAR."

            print(f"[STATE] {state_text}")
            await session.send_realtime_input(text=state_text)

            # Receive
            raw_text = ""
            turn_complete = False

            try:
                async with asyncio.timeout(1.5):
                    async for response in session.receive():
                        if response.server_content:
                            sc = response.server_content
                            if sc.output_transcription and sc.output_transcription.text:
                                t = sc.output_transcription.text.strip()
                                if t:
                                    raw_text += t + " "
                            if sc.model_turn and sc.model_turn.parts:
                                for part in sc.model_turn.parts:
                                    if hasattr(part, 'text') and part.text:
                                        t = part.text.strip()
                                        if t:
                                            raw_text += t + " "
                            if sc.turn_complete:
                                turn_complete = True
                        if response.text and response.text.strip():
                            raw_text += response.text.strip() + " "
                        if turn_complete and raw_text.strip():
                            break
            except asyncio.TimeoutError:
                pass

            raw_text = raw_text.strip()
            if not raw_text:
                await asyncio.sleep(max(0.0, FRAME_INTERVAL - (time.time() - t_start)))
                continue

            # Parse and inject
            lane = _parse_lane(raw_text)
            if lane:
                game.inject_lane(lane)
                label = f"LANE:{lane}"
            else:
                label = f"?({raw_text[:20]})"

            count += 1
            elapsed = time.time() - t_start
            score = f"score {game._score_ref[0]:5d}" if HAS_GAME else ""
            with game._state_lock:
                actual = game._target_lane
            print(f"[DECISION] {label:12s} | actual={actual} | {elapsed*1000:6.0f}ms | {score}")

            await asyncio.sleep(max(0.0, FRAME_INTERVAL - elapsed))


async def run_connector():
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        print("[connector] ERROR: No API key")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    print("[connector] Started")
    while True:
        print("[connector] Session starting")
        try:
            await _one_session(client, types, 999)
        except Exception as e:
            print(f"[connector] ERROR: {e}")
            await asyncio.sleep(1.0)


def start_connector():
    def _run():
        asyncio.run(run_connector())
    threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    asyncio.run(run_connector())