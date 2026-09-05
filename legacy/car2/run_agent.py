"""
Run a trained PPO agent on the AI Drive Game
"""

import sys
import pygame
from stable_baselines3 import PPO
from ai_drive_game_env import DriveGameEnv

# Initialize pygame
pygame.init()

def run_agent(model_path, n_episodes=5, fps=30):
    """
    Run trained agent and display gameplay.
    
    Args:
        model_path: Path to saved model (with or without .zip)
        n_episodes: Number of episodes to run
        fps: Frames per second (higher = faster)
    """
    
    # Clean up path
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    
    print(f"Loading model: {model_path}")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        print(f"Make sure the model exists at: {model_path}.zip")
        sys.exit(1)
    
    # Create environment with rendering
    env = DriveGameEnv(render_mode="human", max_episode_steps=5000)
    
    print(f"\nRunning {n_episodes} episodes...")
    print("-" * 60)
    
    scores = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Get action from trained agent
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Render
            env.render()
        
        score = info["score"]
        scores.append(score)
        print(f"Episode {episode+1}: Score = {score:5d}, Steps = {steps}")
    
    env.close()
    
    print("-" * 60)
    print(f"Average score: {sum(scores)/len(scores):.0f}")
    print(f"Best score:    {max(scores)}")
    print(f"Worst score:   {min(scores)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trained agent")
    parser.add_argument("model", help="Path to trained model (e.g., models/ppo_drive_final)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    
    args = parser.parse_args()
    
    run_agent(args.model, n_episodes=args.episodes, fps=args.fps)
