import sys
import os
sys.path.append(os.getcwd())
from ai_drive_game_env import DriveEnv
import random

def play_random_agent(episodes=3):
    env = DriveEnv(render_mode="human")

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            # Random action
            action = random.randint(0, 3)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()

        final_score = info.get('score', 'N/A')
        print(f"Episode {episode + 1} finished. Score: {final_score}")

    env.close()

if __name__ == "__main__":
    play_random_agent(episodes=3)
