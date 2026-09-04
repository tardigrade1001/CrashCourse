"""
Gemini Connector - Vision Only Mode
Sends image + varied prompts. No text state.
"""

import asyncio
import time
import threading
import re
import sys

try:
    import ai_drive_game_vision as game
    HAS_GAME = True
except ImportError:
    HAS_GAME = False

_KEY_FILE = r"F:\PC Files\Projects\car\API_KEY.txt"
try:
    GEMINI_API_KEY = open(_KEY_FILE).read().strip()
except FileNotFoundError:
    GEMINI_API_KEY = None

MODEL = "gemini-3.1-flash-live-preview"
FRAME_INTERVAL = 1.0

SYSTEM_PROMPT = """You are a vision AI controlling a CYAN car in a 4-lane driving game.

VISUAL UNDERSTANDING:
- Lane numbers 1-4 at TOP (1=left, 4=right)
- CYAN car at BOTTOM
- Colored vehicles are obstacles moving DOWN toward your car
- Grid lines show distance zones: FAR, MID, CLOSE, DANGER

YOUR TASK:
1. Look at BOTTOM THIRD where your car is
2. Find which lanes have obstacles
3. Choose the SAFEST lane with most clearance
4. CHANGE LANES if situation changed

RESPOND ONLY: LANE:1 or LANE:2 or LANE:3 or LANE:4
Never say anything else."""


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

            # Capture frame
            if HAS_GAME:
                try:
                    jpeg_bytes = game.get_frame_jpeg(game.game_surface, quality=75)
                except Exception as e:
                    print(f"[connector] Frame error: {e}")
                    await asyncio.sleep(0.1)
                    continue
            else:
                from PIL import Image
                import io
                img = Image.new("RGB", (480, 360), color=(45, 45, 55))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                jpeg_bytes = buf.getvalue()

            # Send image
            await session.send_realtime_input(
                video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
            )
            print(f"[connector] Frame #{count}")

            # Varied prompts prevent drift
            prompts = [
                "Which lane is safest?",
                "Obstacles approaching? Best lane?",
                "Analyze. Safest lane?",
                "Avoid collision. Which lane?",
                "Traffic ahead. Best lane?",
            ]
            prompt = prompts[count % len(prompts)]
            await session.send_realtime_input(text=prompt)
            print(f"[connector] Prompt: {prompt}")

            # Receive response
            raw_text = ""
            turn_complete = False

            try:
                async with asyncio.timeout(2.0):
                    async for response in session.receive():
                        if response.server_content:
                            sc = response.server_content
                            if sc.output_transcription and sc.output_transcription.text:
                                t = sc.output_transcription.text.strip()
                                if t:
                                    raw_text += t + " "
                                    print(f"[connector] Transcription: {t}")
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
                print(f"[connector] No response for frame #{count}")
                await asyncio.sleep(max(0.0, FRAME_INTERVAL - (time.time() - t_start)))
                continue

            # Parse lane
            lane = _parse_lane(raw_text)
            if lane:
                if HAS_GAME:
                    game.inject_lane(lane)
                label = f"LANE:{lane}"
            else:
                label = f"?({raw_text[:20]})"
                print(f"[connector] Parse failed: {raw_text[:50]}")

            count += 1
            elapsed = time.time() - t_start
            score = ""
            if HAS_GAME and hasattr(game, '_score_ref'):
                score = f"score {game._score_ref[0]:5d}"

            print(f"[connector] {label:12s} | {elapsed*1000:6.0f}ms | {score:15s} | [{count:3d}/{n_decisions}]")

            await asyncio.sleep(max(0.0, FRAME_INTERVAL - elapsed))


async def run_connector():
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        print("[connector] ERROR: No API key")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    session_n = 0

    while True:
        session_n += 1
        print(f"[connector] Session #{session_n} starting")
        try:
            await _one_session(client, types, 999)
        except Exception as e:
            print(f"[connector] ERROR: {e}")
            await asyncio.sleep(1.0)


def start_connector():
    def _run():
        asyncio.run(run_connector())
    threading.Thread(target=_run, daemon=True).start()
    print("[connector] Started")


if __name__ == "__main__":
    asyncio.run(run_connector())
