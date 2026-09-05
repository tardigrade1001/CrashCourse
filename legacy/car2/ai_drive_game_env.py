"""
AI Drive Game - Gymnasium Environment Wrapper
Converts the pygame game into a standard RL environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import time

# Import game components
WIDTH, HEIGHT = 480, 360
ROAD_LEFT, ROAD_RIGHT = 90, 390
ROAD_W = ROAD_RIGHT - ROAD_LEFT
LANE_COUNT = 4
LANE_XS = [ROAD_LEFT + int(ROAD_W * (i + 0.5) / 4) for i in range(4)]

CAR_W, CAR_H = 40, 52
CAR_START_LANE = 2
CAR_START_Y = HEIGHT - 70
CAR_SNAP_SPEED = 18

OBS_SPEED_INIT = 0.2
OBS_SPEED_MAX = 0.8
OBS_SPAWN_RATE = 500
LANE_IDLE_MAX = 999
FPS = 30

PLAYER_COL = (0, 200, 255)
BG = (15, 15, 20)
ROAD_COL = (45, 45, 55)
LANE_DIV = (80, 80, 90)
KERB_COL = (255, 255, 255)
HUD_COL = (180, 255, 180)

VEHICLE_TYPES = [
    ("car_red", 40, 50, (220, 50, 50), (180, 220, 255), False),
    ("car_yellow", 40, 50, (230, 200, 30), (180, 220, 255), False),
    ("car_green", 40, 50, (50, 200, 80), (180, 220, 255), False),
    ("car_purple", 40, 50, (180, 60, 220), (180, 220, 255), False),
    ("car_orange", 40, 50, (240, 130, 30), (180, 220, 255), False),
    ("truck_wht", 54, 68, (210, 210, 210), (150, 200, 255), True),
    ("truck_blu", 54, 68, (40, 80, 200), (180, 230, 255), True),
    ("moto", 18, 38, (240, 240, 240), (80, 80, 80), False),
]


class Car:
    def __init__(self):
        self.lane = CAR_START_LANE
        self.x = float(LANE_XS[self.lane - 1])
        self.y = CAR_START_Y
        self.rect = pygame.Rect(0, 0, CAR_W, CAR_H)

    def update(self, target_lane):
        self.lane = target_lane
        tx = float(LANE_XS[self.lane - 1])
        diff = tx - self.x
        if abs(diff) < CAR_SNAP_SPEED:
            self.x = tx
        else:
            self.x += CAR_SNAP_SPEED * (1 if diff > 0 else -1)

    def get_rect(self):
        self.rect.center = (int(self.x), self.y)
        return self.rect

    def draw(self, surf):
        r = self.get_rect()
        pygame.draw.rect(surf, PLAYER_COL, r, border_radius=6)
        pygame.draw.rect(surf, (20, 20, 30), (r.x+6, r.y+8, r.w-12, 13), border_radius=3)


class Obstacle:
    def __init__(self, speed):
        vtype = random.choice(VEHICLE_TYPES)
        self.label, self.w, self.h, self.body, self.win, self.is_truck = vtype
        lane = random.randint(1, 4)
        self.x = LANE_XS[lane - 1]
        self.y = float(-self.h)
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
            pygame.draw.rect(surf, self.win, (r.x+4, r.y+4, r.w-8, 22), border_radius=4)
        else:
            ws_h = 8 if self.label == "moto" else 12
            pygame.draw.rect(surf, self.win, (r.x+3, r.y+5, r.w-6, ws_h), border_radius=3)


_lane_font = None

def draw_road(surf, scroll):
    global _lane_font
    if _lane_font is None:
        _lane_font = pygame.font.SysFont("monospace", 16, bold=True)

    surf.fill(BG)
    pygame.draw.rect(surf, ROAD_COL, (ROAD_LEFT, 0, ROAD_W, HEIGHT))
    
    for lx in LANE_XS[:-1]:
        mid = (lx + LANE_XS[LANE_XS.index(lx)+1]) // 2
        dash_h, gap = 20, 16
        total = dash_h + gap
        offset = int(scroll) % total
        y = -total + offset
        while y < HEIGHT:
            pygame.draw.rect(surf, LANE_DIV, (mid-1, y, 2, dash_h))
            y += total
    
    pygame.draw.rect(surf, KERB_COL, (ROAD_LEFT-6, 0, 6, HEIGHT))
    pygame.draw.rect(surf, KERB_COL, (ROAD_RIGHT, 0, 6, HEIGHT))
    
    for i, lx in enumerate(LANE_XS):
        label = _lane_font.render(str(i+1), True, (255, 255, 100))
        surf.blit(label, (lx - 8, 10))


def draw_hud(surf, font, score, speed):
    surf.blit(font.render(f"SCORE {score:05d}", True, HUD_COL), (8, 6))
    surf.blit(font.render(f"SPD  {speed:.1f}", True, HUD_COL), (8, 26))


class DriveGameEnv(gym.Env):
    """
    Gymnasium environment for the AI Drive Game.
    
    Action space: Discrete(4) — lane 1, 2, 3, or 4
    Observation space: Box(5,) — [car_lane, dist_lane1, dist_lane2, dist_lane3, dist_lane4]
                       where dist_laneX is distance to nearest obstacle (clamped 0-360)
    
    Reward: +1 per frame survived, -100 on crash
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode=None, max_episode_steps=2000):
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # Action: lane choice (1-4)
        self.action_space = spaces.Discrete(4)
        
        # Observation: [car_lane, dist_to_obs_lane1, dist_lane2, dist_lane3, dist_lane4]
        # Each distance is normalized 0-1 (0 = collision, 1 = no obstacle)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16, bold=True)
        
        # Pygame setup
        if render_mode:
            self.surf = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("AI Drive Game - Training")
        else:
            self.surf = pygame.Surface((WIDTH, HEIGHT))
        
        # Game state
        self.car = None
        self.obstacles = []
        self.score = 0
        self.scroll = 0.0
        self.obs_speed = OBS_SPEED_INIT
        self.frame_n = 0
        self.t0 = time.time()
        self.lane_idle = [0] * 4
        self.steps = 0
        self.crashed = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.car = Car()
        self.obstacles = []
        self.score = 0
        self.scroll = 0.0
        self.obs_speed = OBS_SPEED_INIT
        self.frame_n = 0
        self.t0 = time.time()
        self.lane_idle = [0] * 4
        self.steps = 0
        self.crashed = False
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _get_observation(self):
        """
        Return state vector: [car_lane_normalized, dist_lane1, dist_lane2, dist_lane3, dist_lane4]
        where car_lane is 0-1 and dist_laneX is 0 (collision) to 1 (clear)
        """
        # Car lane (normalize 1-4 to 0-1)
        car_lane_norm = (self.car.lane - 1) / 3.0
        
        # Distance to nearest obstacle in each lane
        lane_distances = []
        for lane_idx in range(4):
            lane_x = LANE_XS[lane_idx]
            min_dist = HEIGHT  # If no obstacle, distance is full screen height
            
            for obs in self.obstacles:
                # Check if obstacle is in this lane (within tolerance)
                if abs(obs.x - lane_x) < 25:
                    # Distance is from car to obstacle
                    dist_to_obs = self.car.y - obs.y
                    if dist_to_obs > 0:  # Obstacle is ahead
                        min_dist = min(min_dist, dist_to_obs)
            
            # Normalize: 0 = collision (< 60 pixels), 1 = far away
            normalized = min(1.0, max(0.0, (min_dist - 50) / 250.0))
            lane_distances.append(normalized)
        
        obs = np.array(
            [car_lane_norm] + lane_distances,
            dtype=np.float32
        )
        return obs
    
    def step(self, action):
        """
        Execute one step of the environment.
        action: int in {0, 1, 2, 3} representing lanes {1, 2, 3, 4}
        """
        self.steps += 1
        
        # Convert action (0-3) to lane (1-4)
        target_lane = action + 1
        
        # Update car
        self.car.update(target_lane)
        
        # Spawn obstacles
        self.frame_n += 1
        self.scroll += self.obs_speed
        
        occupied = set()
        for o in self.obstacles:
            if o.y < HEIGHT * 0.6:
                for i, lx in enumerate(LANE_XS):
                    if abs(o.x - lx) < 20:
                        occupied.add(i)
        
        for i in range(4):
            if i not in occupied:
                self.lane_idle[i] += 1
            else:
                self.lane_idle[i] = 0
        
        def clear_lanes_count():
            blocked = set()
            for o in self.obstacles:
                if o.y < HEIGHT // 3:
                    for i, lx in enumerate(LANE_XS):
                        if abs(o.x - lx) < 20:
                            blocked.add(i)
            return 4 - len(blocked)
        
        # Spawn obstacles more frequently for training
        spawn_r = max(30, OBS_SPAWN_RATE // 10 - self.score // 1000)  # Much faster spawning
        top_clear = all(o.y > 60 for o in self.obstacles) if self.obstacles else True
        if self.frame_n % spawn_r == 0 and top_clear and clear_lanes_count() > 2:
            self.obstacles.append(Obstacle(self.obs_speed))
        
        # Update obstacles
        for o in self.obstacles:
            o.update()
        self.obstacles = [o for o in self.obstacles if not o.off_screen()]
        
        # Collision check
        car_r = self.car.get_rect()
        crashed = False
        for o in self.obstacles:
            if car_r.colliderect(o.get_rect()):
                crashed = True
                break
        
        # Reward
        reward = 1.0  # Survive bonus
        if crashed:
            reward = -100.0
            self.crashed = True
        
        # Score increases
        self.score += 1
        self.obs_speed = min(OBS_SPEED_MAX, OBS_SPEED_INIT + self.score / 5000.0)
        
        # Observation
        obs = self._get_observation()
        
        # Termination
        terminated = crashed
        truncated = self.steps >= self.max_episode_steps
        
        info = {
            "score": self.score,
            "obs_speed": self.obs_speed,
            "lane": self.car.lane,
        }
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode is None:
            return
        
        # Handle events so window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        # Draw game
        draw_road(self.surf, self.scroll)
        for o in self.obstacles:
            o.draw(self.surf)
        self.car.draw(self.surf)
        draw_hud(self.surf, self.font, self.score, self.obs_speed)
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def close(self):
        pygame.quit()
