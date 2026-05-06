import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import numpy as np

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
LANE_WIDTH = SCREEN_WIDTH // 4
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class DriveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super(DriveEnv, self).__init__()

        # 4 Discrete actions: Lane 1, 2, 3, 4
        self.action_space = spaces.Discrete(4)

        # Observation space:
        # upgraded: 4 lanes x 20 depth segments + 1 for current lane = 81 features
        self.observation_space = spaces.Box(low=0, high=1, shape=(81,), dtype=np.float32)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.max_frames = 3000
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car_lane = random.randint(0, 3)
        self.car_y = SCREEN_HEIGHT - 100
        self.obstacles = []
        self.score = 0
        self.frame_count = 0

        self.prev_lane = self.car_lane

        if self.render_mode == "human":
            self._init_render()

        return self._get_obs(), {}

    def _get_obs(self):
        # HD Grid: 4 lanes, 20 segments deep
        grid = np.zeros((4, 20), dtype=np.float32)

        for obs_lane, obs_y in self.obstacles:
            dist = self.car_y - (obs_y + 40)
            if 0 <= dist < 1000:
                seg_idx = int(dist // 50)
                if seg_idx < 20:
                    grid[obs_lane, seg_idx] = 1.0

        # Flatten grid and add car lane
        obs = np.concatenate([grid.flatten(), [self.car_lane / 3.0]])
        return obs.astype(np.float32)

    def step(self, action):
        self.prev_lane = self.car_lane
        self.car_lane = action

        # Move obstacles
        for i in range(len(self.obstacles)):
            self.obstacles[i] = (self.obstacles[i][0], self.obstacles[i][1] + 10)

        self.obstacles = [o for o in self.obstacles if o[1] < SCREEN_HEIGHT + 100]

        self.frame_count += 1
        if self.frame_count % 10 == 0:
            lane = random.randint(0, 3)
            self.obstacles.append((lane, -100))

        terminated = False
        truncated = False
        reward = 1.0

        # Encourage center lane
        if self.car_lane == 1 or self.car_lane == 2:
            reward += 0.1

        if self.car_lane != self.prev_lane:
            reward -= 0.1

        for obs_lane, obs_y in self.obstacles:
            if obs_lane == self.car_lane:
                if self.car_y < obs_y + 40 and self.car_y + 60 > obs_y:
                    terminated = True
                    reward = -100.0
                    break

        self.score += 1
        if self.frame_count >= self.max_frames:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {"score": self.score}

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            self._init_render()
            
        # Handle Windows events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # If window was closed, don't try to draw
        if not pygame.get_init() or self.screen is None:
            return

        try:
            self.screen.fill(BLACK)
            for i in range(1, 4):
                pygame.draw.line(self.screen, WHITE, (i * LANE_WIDTH, 0), (i * LANE_WIDTH, SCREEN_HEIGHT), 2)
            car_x = self.car_lane * LANE_WIDTH + (LANE_WIDTH - 40) // 2
            pygame.draw.rect(self.screen, BLUE, (car_x, self.car_y, 40, 60))
            for obs_lane, obs_y in self.obstacles:
                obs_x = obs_lane * LANE_WIDTH + (LANE_WIDTH - 40) // 2
                pygame.draw.rect(self.screen, RED, (obs_x, obs_y, 40, 40))
            
            # Simple score display without complex font handling during possible closure
            if pygame.font.get_init():
                font = pygame.font.SysFont(None, 36)
                img = font.render(f'Score: {self.score}', True, WHITE)
                self.screen.blit(img, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(FPS)
        except pygame.error:
            # Handle cases where display was closed mid-render
            self.screen = None

    def _init_render(self):
        if not pygame.get_init():
            pygame.init()
        if self.screen is None:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("AI Drive Game")
            self.clock = pygame.time.Clock()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
