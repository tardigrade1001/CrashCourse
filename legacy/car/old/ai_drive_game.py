"""
AI Drive Test v3 — 3-lane snap control, varied obstacle vehicles
Controls (human):  A / S / D  or  LEFT / RIGHT arrow keys (snap between lanes)
AI control:        gemini_connector.py calls inject_lane(1|2|3)
Usage:
    python ai_drive_game.py          # human mode
    python ai_drive_game.py --ai     # AI mode
"""

import pygame
import random
import sys
import csv
import time
import io
import base64
from threading import Lock

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 480, 360

ROAD_LEFT       = 90
ROAD_RIGHT      = 390
ROAD_W          = ROAD_RIGHT - ROAD_LEFT

# 3 lane centres
LANE_COUNT      = 4
LANE_XS         = [
    ROAD_LEFT + int(ROAD_W * (i + 0.5) / 4)
    for i in range(4)
]

CAR_W, CAR_H    = 40, 52
CAR_START_LANE  = 2                  # start in centre
CAR_START_Y     = HEIGHT - 70
CAR_SNAP_SPEED  = 18                 # px/frame when sliding to new lane

OBS_SPEED_INIT  = 0.4
OBS_SPEED_MAX   = 1.2
OBS_SPAWN_RATE  = 360   # easy mode — lots of breathing room
LANE_IDLE_MAX   = 999   # effectively disabled — no pressure spawns
FPS             = 30

PLAYER_COL      = (0, 200, 255)

VEHICLE_TYPES = [
    ("car_red",    40, 50, (220,  50,  50), (180,220,255), False),
    ("car_yellow", 40, 50, (230, 200,  30), (180,220,255), False),
    ("car_green",  40, 50, ( 50, 200,  80), (180,220,255), False),
    ("car_purple", 40, 50, (180,  60, 220), (180,220,255), False),
    ("car_orange", 40, 50, (240, 130,  30), (180,220,255), False),
    ("truck_wht",  54, 68, (210, 210, 210), (150,200,255), True),
    ("truck_blu",  54, 68, ( 40,  80, 200), (180,230,255), True),
    ("moto",       18, 38, (240, 240, 240), ( 80, 80, 80), False),
]

BG          = (15,  15,  20)
ROAD_COL    = (45,  45,  55)
LANE_COL    = (220, 220,  80)
KERB_COL    = (255, 255, 255)
HUD_COL     = (180, 255, 180)
LANE_DIV    = (80,  80,  90)         # subtle lane divider

# ── Shared AI state ───────────────────────────────────────────────────────────
_state_lock  = Lock()
_target_lane = CAR_START_LANE        # 1, 2, or 3
_ai_log      = []
_score_ref   = [0]

def inject_lane(lane: int):
    """AI calls this with 1, 2, or 3."""
    global _target_lane
    lane = max(1, min(4, int(lane)))
    with _state_lock:
        _target_lane = lane
    _ai_log.append((time.time(), f"L{lane}", _score_ref[0]))

def inject_command(cmd: str):
    """Legacy LEFT/RIGHT/FORWARD fallback."""
    cmd = cmd.strip().upper()
    with _state_lock:
        cur = _target_lane
    if cmd == "LEFT"  and cur > 1: inject_lane(cur - 1)
    if cmd == "RIGHT" and cur < 4: inject_lane(cur + 1)

def inject_target_x(x: float):
    """Legacy X-position fallback — snap to nearest lane."""
    dists = [abs(x - lx) for lx in LANE_XS]
    inject_lane(dists.index(min(dists)) + 1)

def get_frame_jpeg(surface, quality=75) -> bytes:
    # Draw lane number labels onto a copy before encoding
    annotated = surface.copy()
    try:
        font = pygame.font.SysFont("monospace", 13, bold=True)
        for i, lx in enumerate(LANE_XS):
            label = font.render(str(i+1), True, (255, 255, 100))
            annotated.blit(label, (lx - 5, 4))
            # faint vertical lane guide line
            pygame.draw.line(annotated, (70, 70, 80), (lx, 0), (lx, HEIGHT), 1)
    except Exception:
        pass

    raw  = pygame.image.tobytes(annotated, "RGB")
    size = annotated.get_size()
    try:
        from PIL import Image
        img = Image.frombytes("RGB", size, raw)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except ImportError:
        buf = io.BytesIO()
        pygame.image.save(annotated, buf)
        return buf.getvalue()

def get_frame_b64(surface, quality=75) -> str:
    return base64.b64encode(get_frame_jpeg(surface, quality)).decode()


def get_game_state_text(car_lane, obstacles):
    """Return a short text description of the current game state."""
    # find obstacles in bottom half only
    threats = []
    for o in obstacles:
        if o.y > HEIGHT * 0.2:   # bottom 60% of screen
            # figure out which lane
            dists = [abs(o.x - lx) for lx in LANE_XS]
            lane_idx = dists.index(min(dists))
            lane_num = lane_idx + 1
            proximity = "VERY CLOSE" if o.y > HEIGHT * 0.65 else "CLOSE"
            threats.append((lane_num, proximity, o.y))

    if not threats:
        return f"Car is in lane {car_lane}. Road ahead is CLEAR. Stay in lane {car_lane}."

    threats.sort(key=lambda t: -t[2])   # closest first
    blocked = set(t[0] for t in threats)
    clear   = [i for i in range(1, len(LANE_XS)+1) if i not in blocked]

    lines = [f"Car is in lane {car_lane}."]
    for lane_num, prox, _ in threats:
        lines.append(f"Lane {lane_num}: obstacle {prox}.")
    if clear:
        lines.append(f"Clear lanes: {', '.join(str(l) for l in clear)}.")
    else:
        lines.append("All lanes have obstacles — pick the one with most space.")
    return " ".join(lines)

# ── Game objects ──────────────────────────────────────────────────────────────
class Car:
    def __init__(self):
        self.lane   = CAR_START_LANE
        self.x      = float(LANE_XS[self.lane - 1])
        self.y      = CAR_START_Y
        self.rect   = pygame.Rect(0, 0, CAR_W, CAR_H)

    def update(self, ai_mode, keys=None):
        if not ai_mode and keys:
            if (keys[pygame.K_LEFT]  or keys[pygame.K_a]) and self.lane > 1:
                self.lane -= 1
            if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and self.lane < 3:
                self.lane += 1
        elif ai_mode:
            with _state_lock:
                self.lane = _target_lane

        # Snap slide toward target lane centre
        tx = float(LANE_XS[self.lane - 1])
        diff = tx - self.x
        if abs(diff) < CAR_SNAP_SPEED:
            self.x = tx
        else:
            self.x += CAR_SNAP_SPEED * (1 if diff > 0 else -1)

    def get_rect(self):
        self.rect.center = (int(self.x), self.y)
        return self.rect

    def draw(self, surf, ai_mode):
        r = self.get_rect()
        pygame.draw.rect(surf, PLAYER_COL, r, border_radius=6)
        pygame.draw.rect(surf, (20,20,30),
                         (r.x+6, r.y+8, r.w-12, 13), border_radius=3)
        for wx, wy in [(r.x-4, r.y+5), (r.x-4, r.bottom-17),
                       (r.right, r.y+5), (r.right, r.bottom-17)]:
            pygame.draw.rect(surf, (30,30,30), (wx, wy, 8, 11), border_radius=2)
        # lane indicator dots above car (AI mode)
        if ai_mode:
            with _state_lock:
                tl = _target_lane
            for i, lx in enumerate(LANE_XS):
                col = (255,220,0) if (i+1)==tl else (60,60,70)
                pygame.draw.circle(surf, col, (lx, r.y - 10), 4)


class Obstacle:
    def __init__(self, speed):
        vtype = random.choice(VEHICLE_TYPES)
        self.label, self.w, self.h, self.body, self.win, self.is_truck = vtype
        # Snap obstacles to lanes too — cleaner gaps
        lane = random.randint(1, 3)
        self.x     = LANE_XS[lane - 1]
        self.y     = float(-self.h)
        self.speed = speed

    def update(self):
        self.y += self.speed

    def off_screen(self):
        return self.y > HEIGHT + self.h

    def get_rect(self):
        return pygame.Rect(self.x - self.w//2, int(self.y), self.w, self.h)

    def draw(self, surf):
        r = self.get_rect()
        pygame.draw.rect(surf, self.body, r, border_radius=5)
        if self.is_truck:
            pygame.draw.rect(surf, self.win,
                             (r.x+4, r.y+4, r.w-8, 22), border_radius=4)
            pygame.draw.rect(surf, (255,255,255),
                             (r.x+6, r.y+30, r.w-12, 5))
        else:
            ws_h = 8 if self.label == "moto" else 12
            pygame.draw.rect(surf, self.win,
                             (r.x+3, r.y+5, r.w-6, ws_h), border_radius=3)
        if self.w > 20:
            for wx, wy in [(r.x-3, r.y+5), (r.x-3, r.bottom-15),
                           (r.right, r.y+5), (r.right, r.bottom-15)]:
                pygame.draw.rect(surf, (20,20,20), (wx, wy, 7, 10), border_radius=2)


# ── Drawing ───────────────────────────────────────────────────────────────────
_lane_font = None

def draw_road(surf, scroll):
    global _lane_font
    if _lane_font is None:
        _lane_font = pygame.font.SysFont("monospace", 13, bold=True)

    surf.fill(BG)
    pygame.draw.rect(surf, ROAD_COL, (ROAD_LEFT, 0, ROAD_W, HEIGHT))
    # subtle lane dividers
    for lx in LANE_XS[:-1]:
        mid = (lx + LANE_XS[LANE_XS.index(lx)+1]) // 2
        dash_h, gap = 20, 16
        total  = dash_h + gap
        offset = int(scroll) % total
        y = -total + offset
        while y < HEIGHT:
            pygame.draw.rect(surf, LANE_DIV, (mid-1, y, 2, dash_h))
            y += total
    # kerbs
    pygame.draw.rect(surf, KERB_COL, (ROAD_LEFT-6, 0, 6, HEIGHT))
    pygame.draw.rect(surf, KERB_COL, (ROAD_RIGHT,  0, 6, HEIGHT))
    # lane numbers — visible on screen AND in Gemini frames
    for i, lx in enumerate(LANE_XS):
        label = _lane_font.render(str(i+1), True, (255, 255, 100))
        surf.blit(label, (lx - 4, 6))


def draw_hud(surf, font, score, speed, ai_mode, lane):
    surf.blit(font.render(f"SCORE {score:05d}", True, HUD_COL), (8, 6))
    surf.blit(font.render(f"SPD  {speed:.1f}",  True, HUD_COL), (8, 26))
    mode_col = (255, 200, 0) if ai_mode else (100, 100, 120)
    surf.blit(font.render("AI" if ai_mode else "HUMAN", True, mode_col),
              (WIDTH - 72, 6))
    if ai_mode:
        surf.blit(font.render(f"TGT  {lane}", True, (255,200,0)),
                  (WIDTH - 72, 26))


def draw_gameover(surf, bfont, font, score, survived):
    ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    ov.fill((0, 0, 0, 170))
    surf.blit(ov, (0, 0))
    go = bfont.render("GAME OVER", True, (255, 60, 60))
    surf.blit(go, go.get_rect(center=(WIDTH//2, HEIGHT//2 - 50)))
    sc = font.render(f"Score: {score}   Time: {survived:.1f}s",
                     True, (220, 220, 220))
    surf.blit(sc, sc.get_rect(center=(WIDTH//2, HEIGHT//2)))
    rs = font.render("R = restart    Q = quit", True, (150, 150, 150))
    surf.blit(rs, rs.get_rect(center=(WIDTH//2, HEIGHT//2 + 36)))


# ── Main loop ─────────────────────────────────────────────────────────────────
game_surface = None
_obstacles   = []

def _reset():
    global _target_lane
    _target_lane = CAR_START_LANE
    return Car(), [], 0, 0.0, float(OBS_SPEED_INIT), 0, time.time(), False, [0]*4


_last_lane_input = [0]   # debounce for human keys

def main(ai_mode=False):
    global game_surface

    pygame.init()
    surf = pygame.display.set_mode((WIDTH, HEIGHT))
    game_surface = surf
    pygame.display.set_caption("AI Drive Test v3 — 4 lanes")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", 16, bold=True)
    bfont = pygame.font.SysFont("monospace", 36, bold=True)

    # key debounce
    key_cooldown = 0
    car, obstacles, score, scroll, obs_speed, frame_n, t0, dead, lane_idle = _reset()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _save_log(); pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if dead:
                    if event.key == pygame.K_r:
                        car, obstacles, score, scroll, obs_speed, frame_n, t0, dead, lane_idle = _reset()
                    if event.key == pygame.K_q:
                        _save_log(); pygame.quit(); sys.exit()
                else:
                    if not ai_mode:
                        if event.key in (pygame.K_LEFT,  pygame.K_a) and car.lane > 1:
                            car.lane -= 1
                            inject_lane(car.lane)
                        if event.key in (pygame.K_RIGHT, pygame.K_d) and car.lane < 4:
                            car.lane += 1
                            inject_lane(car.lane)

        _score_ref[0] = score

        if dead:
            draw_road(surf, scroll)
            for o in obstacles: o.draw(surf)
            car.draw(surf, ai_mode)
            with _state_lock: tl = _target_lane
            draw_hud(surf, font, score, obs_speed, ai_mode, tl)
            draw_gameover(surf, bfont, font, score, time.time() - t0)
            pygame.display.flip()
            continue

        # ── Update car ──
        if ai_mode:
            with _state_lock:
                tl = _target_lane
            car.lane = tl
        car.update(ai_mode)

        # ── Obstacles ──
        frame_n += 1
        scroll  += obs_speed

        # Track which lanes are occupied near top
        occupied = set()
        for o in obstacles:
            if o.y < HEIGHT * 0.6:
                for i, lx in enumerate(LANE_XS):
                    if abs(o.x - lx) < 20:
                        occupied.add(i)

        # Increment idle counter for unoccupied lanes
        n_lanes = len(LANE_XS)
        for i in range(n_lanes):
            if i not in occupied:
                lane_idle[i] += 1
            else:
                lane_idle[i] = 0

        # How many lanes are clear in the top third
        def clear_lanes_count():
            blocked = set()
            for o in obstacles:
                if o.y < HEIGHT // 3:
                    for i, lx in enumerate(LANE_XS):
                        if abs(o.x - lx) < 20:
                            blocked.add(i)
            return len(LANE_XS) - len(blocked)

        # Normal spawn — only if at least 2 lanes will remain clear
        spawn_r   = max(60, OBS_SPAWN_RATE - score // 300)
        top_clear = all(o.y > 60 for o in obstacles)
        if frame_n % spawn_r == 0 and top_clear and clear_lanes_count() > 2:
            obstacles.append(Obstacle(obs_speed))

        # Force spawn in idlest lane — only if at least 3 lanes still clear
        idlest = max(range(len(LANE_XS)), key=lambda i: lane_idle[i])
        if lane_idle[idlest] > LANE_IDLE_MAX and clear_lanes_count() > 2:
            forced = Obstacle(obs_speed)
            forced.x = LANE_XS[idlest]
            obstacles.append(forced)
            lane_idle[idlest] = 0

        for o in obstacles: o.update()
        obstacles = [o for o in obstacles if not o.off_screen()]
        _obstacles = obstacles   # expose for connector

        score    += 1
        obs_speed = min(float(OBS_SPEED_MAX), OBS_SPEED_INIT + score / 3000.0)

        # ── Collision ──
        car_r = car.get_rect()
        for o in obstacles:
            if car_r.colliderect(o.get_rect()):
                dead = True; break

        # ── Draw ──
        draw_road(surf, scroll)
        for o in obstacles: o.draw(surf)
        car.draw(surf, ai_mode)
        with _state_lock: tl = _target_lane
        draw_hud(surf, font, score, obs_speed, ai_mode, tl)
        pygame.display.flip()

    _save_log()


def _save_log():
    if not _ai_log:
        return
    with open("ai_drive_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "lane", "score"])
        w.writerows(_ai_log)
    print(f"Saved {len(_ai_log)} AI decisions to ai_drive_log.csv")


if __name__ == "__main__":
    main(ai_mode="--ai" in sys.argv)