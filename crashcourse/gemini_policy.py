"""A hosted vision-language model steering the car, as a Policy.

This is V1's idea rebuilt on the measurement that ended it. The original
connector paced itself with a fixed 0.7 second sleep and streamed a JPEG frame
per decision, which held the loop to 1.07 Hz. Holding one Live session open,
sending the state as text, and reading the lane from the output transcript
brings the median decision to roughly 0.42 seconds on the same key and model
family.

The model runs far slower than the environment. That gap is the subject here.
``decide_every`` sets how many environment
steps pass between decisions, the action is held in between, and sweeping it
places the trained agent and the hosted model on one axis.

    from crashcourse.gemini_policy import GeminiPolicy
    with GeminiPolicy(cfg, decide_every=25) as policy:
        report = evaluate(policy, cfg, episode_seeds(10))

The Live models reachable on a standard key emit audio, so the lane is read
from the output transcript. V1 read it the same way. The API sets this.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from crashcourse.config import EnvConfig
from crashcourse.policies import latest_frame, occupancy_grid

MODEL = "gemini-3.1-flash-live-preview"

SYSTEM_PROMPT = """You steer a car in a four lane game. Lanes are numbered 0, 1, 2, 3.

Each message reports, for every lane, the distance to the nearest obstacle ahead
in segments. Smaller is closer. A lane marked clear has no obstacle in view.

Choose the lane that keeps the car safe for longest. Prefer a clear lane. Among
clear lanes prefer the one nearest the current lane. Stay in the current lane
when it is clear.

Say only the single digit of the lane you choose. Say nothing else."""

_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "for": "3", "four": "3"}


def lane_digit(text: str) -> int | None:
    """First lane index named in a transcript, reading digits and number words."""
    lowered = text.lower()
    for word, digit in _WORDS.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    match = re.search(r"[0-3]", lowered)
    return int(match.group(0)) if match else None


def describe(obs: np.ndarray, cfg: EnvConfig) -> str:
    """The observation as a compact line of text.

    Reports the nearest occupied segment per lane, which is the whole of what
    the occupancy grid says about safety.
    """
    grid = occupancy_grid(obs, cfg)
    parts = []
    for lane in range(cfg.lanes):
        hits = np.flatnonzero(grid[lane] > 0.5)
        parts.append(f"lane{lane} {'clear' if hits.size == 0 else f'obstacle at {int(hits[0])}'}")
    car_lane = int(round(float(obs[-1]) * (cfg.lanes - 1))) if cfg.position_feature != "none" else 0
    return f"car_lane={car_lane} | " + " | ".join(parts)


def read_api_key(explicit: str | None = None) -> str:
    """Key from the argument, then GEMINI_API_KEY, then legacy/car/API_KEY.txt."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env:
        return env.strip()
    local = Path(__file__).resolve().parent.parent / "legacy" / "car" / "API_KEY.txt"
    if local.exists():
        return local.read_text().strip()
    raise RuntimeError("Set GEMINI_API_KEY to run the hosted policy.")


@dataclass
class CallRecord:
    """One decision, kept so latency is reported from measurement."""

    step: int
    latency: float
    state: str
    transcript: str
    lane: int | None


@dataclass
class GeminiStats:
    calls: list[CallRecord] = field(default_factory=list)
    timeouts: int = 0
    unparsed: int = 0

    @property
    def latencies(self) -> np.ndarray:
        return np.array([c.latency for c in self.calls], dtype=float)

    def summary(self) -> str:
        lat = self.latencies
        if lat.size == 0:
            return "no calls recorded"
        return (f"{lat.size} decisions, median {np.median(lat) * 1000:.0f} ms, "
                f"mean {lat.mean() * 1000:.0f} ms, max {lat.max() * 1000:.0f} ms, "
                f"{1.0 / np.median(lat):.2f} Hz, "
                f"{self.timeouts} timeouts, {self.unparsed} unparsed")


class _LiveSession:
    """A Live session on its own event loop, exposed as a blocking ask()."""

    def __init__(self, api_key: str, model: str, timeout: float):
        self._api_key, self._model, self._timeout = api_key, model, timeout
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._session = None
        self._ctx = None
        self._error: BaseException | None = None
        self._thread.start()
        self._ready.wait(timeout=30)
        if self._error is not None:
            raise self._error

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._open())
        except BaseException as exc:  # surfaced to the constructor
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    async def _open(self) -> None:
        from google import genai

        client = genai.Client(api_key=self._api_key)
        config = {
            "response_modalities": ["AUDIO"],
            "output_audio_transcription": {},
            "system_instruction": SYSTEM_PROMPT,
        }
        self._ctx = client.aio.live.connect(model=self._model, config=config)
        self._session = await self._ctx.__aenter__()

    async def _ask(self, state: str) -> str:
        session = self._session
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": state}]}, turn_complete=True)
        transcript = ""
        async with asyncio.timeout(self._timeout):
            async for response in session.receive():
                content = response.server_content
                if content is None:
                    continue
                caption = content.output_transcription
                if caption is not None and caption.text:
                    transcript += caption.text
                if content.turn_complete:
                    break
        return transcript

    def ask(self, state: str) -> str:
        future = asyncio.run_coroutine_threadsafe(self._ask(state), self._loop)
        return future.result(timeout=self._timeout + 5.0)

    def close(self) -> None:
        async def _shutdown():
            try:
                await self._ctx.__aexit__(None, None, None)
            except BaseException:
                pass
        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(timeout=10)
        except BaseException:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class GeminiPolicy:
    """A hosted model steering the car, holding its action between decisions.

    ``decide_every`` counts environment steps per decision. The environment
    renders at 60 frames per second, so 60 is one decision per second and 25 is
    the 2.4 Hz that the measured latency supports.
    """

    def __init__(self, cfg: EnvConfig, decide_every: int = 25, model: str = MODEL,
                 api_key: str | None = None, timeout: float = 6.0,
                 n_stack: int = 4, name: str | None = None):
        self.cfg = cfg
        self.decide_every = max(1, int(decide_every))
        self.n_stack = n_stack
        self.name = name or f"gemini@{self.decide_every}"
        self.stats = GeminiStats()
        self._session = _LiveSession(read_api_key(api_key), model, timeout)
        self._step = 0
        self._action = 0

    def __enter__(self) -> "GeminiPolicy":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    def reset(self) -> None:
        self._step = 0
        self._action = 0

    def act(self, stacked_obs: np.ndarray) -> int:
        if self._step % self.decide_every == 0:
            self._decide(latest_frame(stacked_obs, self.cfg.obs_size))
        self._step += 1
        return self._action

    def _decide(self, obs: np.ndarray) -> None:
        state = describe(obs, self.cfg)
        start = time.perf_counter()
        try:
            transcript = self._session.ask(state)
        except (TimeoutError, asyncio.TimeoutError):
            self.stats.timeouts += 1
            self.stats.calls.append(
                CallRecord(self._step, time.perf_counter() - start, state, "", None))
            return
        latency = time.perf_counter() - start
        lane = lane_digit(transcript)
        self.stats.calls.append(CallRecord(self._step, latency, state, transcript.strip(), lane))
        if lane is None:
            self.stats.unparsed += 1
            return
        self._action = int(np.clip(lane, 0, self.cfg.lanes - 1))
