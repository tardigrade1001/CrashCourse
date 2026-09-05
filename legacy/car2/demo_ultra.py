"""
Visualize Ultra Hard Mode
Shows what one environment looks like (packed with traffic, fast cars)
Run this alongside training to see what the agent is learning from
"""

import pygame
from ai_drive_game_env_ultra import DriveGameEnvUltra

def demo_ultra_hard():
    """Show one episode of ultra hard mode"""
    
    print("\n" + "=" * 70)
    print("ULTRA HARD MODE VISUALIZATION")
    print("=" * 70)
    print("Watch the environment - packed traffic, fast cars, chaos!")
    print("Close the window to exit\n")
    
    env = DriveGameEnvUltra(render_mode="human", max_episode_steps=500)
    
    episode = 0
    total_episodes = 0
    
    try:
        while True:
            episode += 1
            obs, _ = env.reset()
            done = False
            steps = 0
            score = 0
            
            while not done:
                # Random action (agent just wandering)
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                score = info["score"]
                done = terminated or truncated
                env.render()
            
            total_episodes += 1
            print(f"Episode {episode}: Score={score}, Steps={steps}, Speed={info['obs_speed']:.2f}")
    
    except KeyboardInterrupt:
        print(f"\n\nStopped. Watched {total_episodes} episodes.")
    finally:
        env.close()


if __name__ == "__main__":
    demo_ultra_hard()
