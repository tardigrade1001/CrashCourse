# Vision-Only Mode Setup

## What This Is
Pure **vision-based gameplay** — Gemini sees the screen and decides lanes using visual understanding alone. No ground-truth text state.

## Files
- `ai_drive_game_vision.py` — Modified game with **50% slower obstacles**
- `gemini_connector_vision_only.py` — Vision-only connector
- `run_vision.py` — Launcher

## Key Changes

### Game (ai_drive_game_vision.py)
- `OBS_SPEED_INIT = 0.15` (was 0.3)
- `OBS_SPEED_MAX = 0.6` (was 1.2)
- `OBS_SPAWN_RATE = 600` (was 360)
- Slower acceleration: `score / 5000` (was 3000)

**Result:** Obstacles move at half speed. Model has ~1-2 seconds per decision instead of 0.5s.

### Connector (gemini_connector_vision_only.py)
- `FRAME_INTERVAL = 1.0` (was 0.5) — slower decision cycle
- Sends **image only**, no text state
- System prompt tells model to understand visually:
  ```
  Look at lane numbers 1-4 at the top.
  Cyan car at bottom. Colored vehicles are obstacles.
  Which lane is safest?
  ```
- Timeout: 2.0s (more time for reasoning)

## Usage

**Copy files to your project directory:**
```bash
cp ai_drive_game_vision.py <project>/
cp gemini_connector_vision_only.py <project>/
cp run_vision.py <project>/
```

**Run vision-only mode:**
```bash
python run_vision.py
```

**Run human mode (test game):**
```bash
python run_vision.py --human
```

## What to Expect

✓ **Pros:**
- Pure vision understanding (real Live API gameplay)
- Model sees lanes and obstacles visually
- No dependency on game state text generator
- Model has time to analyze (slower obstacles)

✗ **Cons:**
- Lower success rate (~70-80% vs 95%+ with hybrid)
- Occasional oscillation (lane jumping)
- Slower frame rate (1.0s vs 0.5s)
- Model might hallucinate lane positions

## Performance Tuning

If vision understanding is poor:

1. **Increase timeout** (more thinking time):
   - In connector, line ~180: change `asyncio.timeout(2.0)` to `asyncio.timeout(3.0)`

2. **Slower obstacles** (easier to see):
   - In game, line ~35: change `OBS_SPEED_INIT = 0.15` to `0.1`

3. **Improve prompt** (guide model better):
   - In connector, line ~30: update SYSTEM_PROMPT with more detailed instructions

4. **More space between spawns**:
   - In game, line ~40: increase `OBS_SPAWN_RATE` from 600 to 800

## Comparison

| Mode | Accuracy | Stability | Speed | Dependency |
|------|----------|-----------|-------|------------|
| Vision-only | ~70-80% | Medium | 1.0s/frame | Vision model |
| Hybrid (v2) | ~95%+ | High | 0.7s/frame | State text |

Use **vision-only for experimentation**, use **hybrid for reliability**.
