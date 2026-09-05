"""
gemini_connector_fixed_v2.py — Fixed response parsing (turn_complete timing)
Key change: Don't break on turn_complete until we have text
"""

import asyncio
import time
import threading
import re
import sys

try:
    import ai_drive_game as game
    HAS_GAME = True
except ImportError:
    HAS_GAME = False
    print("[connector] Running in test/standalone mode (no game module)")

_KEY_FILE = r"F:\PC Files\Projects\car\API_KEY.txt"
try:
    GEMINI_API_KEY = open(_KEY_FILE).read().strip()
except FileNotFoundError:
    GEMINI_API_KEY = None
    print(f"[connector] WARNING: API key file not found at {_KEY_FILE}")

MODEL = "gemini-3.1-flash-live-preview"
FRAME_INTERVAL = 0.5
SESSION_REFRESH = 999

SYSTEM_PROMPT = """You are driving a CYAN car in a 4-lane top-down driving game.
Lane numbers 1-4 are printed at the top of each lane (1=far left, 4=far right).
Your commands take about 1 second to reach the car.

RULES — follow in order:
1. Look ONLY at the BOTTOM THIRD of the screen near your cyan car.
2. No obstacles in bottom third → say your current lane (stay put).
3. Obstacle in your lane → check 1 lane left, then 1 lane right for a clear lane.
4. Both adjacent lanes blocked → check 2 lanes away.
5. Always say a lane number. Never skip.

Speak ONLY one of: LANE:1  LANE:2  LANE:3  LANE:4"""

DEBUG = True


def _log(level, msg):
    """Unified logging with timestamp."""
    ts = time.strftime("%H:%M:%S")
    prefix = f"[{ts} connector.{level}]"
    print(f"{prefix:30s} {msg}")


def _parse_lane(text):
    """Extract lane number 1-4 from text."""
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


def game_obstacles():
    """Safely get current obstacle list from game module."""
    if not HAS_GAME or not hasattr(game, '_obstacles'):
        return []
    try:
        return game._obstacles
    except Exception:
        return []


async def _one_session(client, types, n_decisions):
    """Run a single session for n_decisions frames."""
    
    config = {
        "response_modalities": ["AUDIO"],
        "output_audio_transcription": {},
        "system_instruction": SYSTEM_PROMPT,
    }

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            count = 0
            decision_without_reset = 0

            while count < n_decisions:
                if HAS_GAME and game.game_surface is None:
                    await asyncio.sleep(0.1)
                    continue

                t_start = time.time()

                # ── Capture frame ──
                jpeg_bytes = None
                if HAS_GAME:
                    try:
                        jpeg_bytes = game.get_frame_jpeg(game.game_surface, quality=70)
                    except Exception as e:
                        _log("WARN", f"Frame capture failed: {e}")
                        await asyncio.sleep(0.1)
                        continue
                else:
                    from PIL import Image
                    import io
                    img = Image.new("RGB", (480, 360), color=(45, 45, 55))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    jpeg_bytes = buf.getvalue()

                # ── Send frame ──
                await session.send_realtime_input(
                    video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
                )
                if DEBUG:
                    _log("SEND", f"Frame #{count} ({len(jpeg_bytes)} bytes)")

                # ── Generate and send game state text ──
                state_text = None
                if HAS_GAME:
                    try:
                        with game._state_lock:
                            cur_lane = game._target_lane
                        state_text = game.get_game_state_text(cur_lane, game_obstacles())
                    except Exception as e:
                        _log("WARN", f"State text generation failed: {e}")
                        state_text = f"Car is in lane 2. Road unclear."
                else:
                    state_text = f"Car is in lane 2. Road ahead is CLEAR."

                await session.send_realtime_input(text=state_text)
                if DEBUG:
                    _log("SEND", f"State: {state_text[:60]}")

                # ── Soft session reset every 15 decisions ──
                decision_without_reset += 1
                if decision_without_reset > 0 and decision_without_reset % 15 == 0:
                    reset_prompt = (
                        f"FRESH START. Ignore all previous lanes. "
                        f"Respond only with LANE:1, LANE:2, LANE:3, or LANE:4. "
                        f"{state_text}"
                    )
                    await session.send_realtime_input(text=reset_prompt)
                    _log("INFO", f"Soft reset at decision #{count} (prompt injection)")

                # ── Receive response ──
                # KEY FIX: Collect ALL text, don't break early on turn_complete
                raw_text = ""
                turn_complete = False
                
                try:
                    async with asyncio.timeout(1.5):  # Increased to 1.5s
                        async for response in session.receive():
                            if response.server_content:
                                sc = response.server_content
                                
                                # Extract from audio transcription
                                if sc.output_transcription and sc.output_transcription.text:
                                    t = sc.output_transcription.text.strip()
                                    if t:
                                        raw_text += t + " "
                                        if DEBUG:
                                            _log("RECV", f"Transcription: {t}")
                                
                                # Extract from model_turn parts
                                if sc.model_turn and sc.model_turn.parts:
                                    for part in sc.model_turn.parts:
                                        if hasattr(part, 'text') and part.text:
                                            t = part.text.strip()
                                            if t:
                                                raw_text += t + " "
                                                if DEBUG:
                                                    _log("RECV", f"ModelTurn: {t}")
                                
                                # Mark turn complete but DON'T break immediately
                                if sc.turn_complete:
                                    turn_complete = True
                                    if DEBUG:
                                        _log("RECV", "Turn complete signal")
                            
                            # Fallback text
                            if response.text and response.text.strip():
                                t = response.text.strip()
                                raw_text += t + " "
                                if DEBUG:
                                    _log("RECV", f"Direct text: {t}")
                            
                            # Break ONLY if: turn_complete AND we have text
                            if turn_complete and raw_text.strip():
                                break

                except asyncio.TimeoutError:
                    # Timeout is OK; we likely have partial text
                    if DEBUG:
                        _log("DEBUG", f"Timeout at frame #{count}, checking text: '{raw_text[:30]}'")
                    pass

                raw_text = raw_text.strip()
                
                if not raw_text:
                    if DEBUG:
                        _log("WARN", f"No text received for frame #{count}")
                    await asyncio.sleep(max(0.0, FRAME_INTERVAL - (time.time() - t_start)))
                    continue

                # ── Parse lane ──
                lane = _parse_lane(raw_text)
                if lane:
                    if HAS_GAME:
                        game.inject_lane(lane)
                    label = f"LANE:{lane}"
                else:
                    label = f"?({raw_text[:20]})"
                    _log("WARN", f"Parse failed: {raw_text[:50]}")

                count += 1
                elapsed = time.time() - t_start
                
                score_str = ""
                if HAS_GAME and hasattr(game, '_score_ref'):
                    score_str = f"score {game._score_ref[0]:5d}"

                _log("DECISION", f"{label:12s} | {elapsed*1000:6.0f}ms | {score_str:15s} | [{count:3d}/{n_decisions}]")

                await asyncio.sleep(max(0.0, FRAME_INTERVAL - elapsed))

    except Exception as e:
        _log("ERROR", f"Session exception: {type(e).__name__}: {e}")
        raise


async def run_connector():
    """Main connector loop with session management."""
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        _log("ERROR", "No API key found. Set GEMINI_API_KEY or provide API_KEY.txt")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    session_n = 0

    while True:
        session_n += 1
        _log("INFO", f"Session #{session_n} starting ({SESSION_REFRESH} decisions per session)")
        try:
            await _one_session(client, types, SESSION_REFRESH)
        except Exception as e:
            _log("ERROR", f"Session #{session_n} failed: {e}")
            _log("INFO", f"Reconnecting in 1s...")
            await asyncio.sleep(1.0)


def start_connector():
    """Start connector in daemon thread."""
    def _run():
        asyncio.run(run_connector())
    
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _log("INFO", "Connector thread started")
    return t


if __name__ == "__main__":
    _log("INFO", "Standalone mode (no game import)")
    asyncio.run(run_connector())