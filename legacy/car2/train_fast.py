"""
Train PPO agent - OPTIMIZED FOR CPU (Ryzen 3600)
Uses 12 parallel environments to max out all cores
"""

import os
import sys
import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

pygame.init()

from ai_drive_game_env import DriveGameEnv

def train_agent(
    total_timesteps=500000,
    n_envs=12,  # Ryzen 3600: 6 cores, 12 threads = use all threads
    learning_rate=3e-4,
    batch_size=256,
):
    """
    Fast CPU training with 12 parallel environments.
    """
    
    print("=" * 70)
    print("TRAINING PPO - OPTIMIZED FOR CPU (RYZEN 3600)")
    print("=" * 70)
    print(f"Parallel environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Batch size: {batch_size}")
    print()
    
    os.makedirs("models", exist_ok=True)
    
    # Create parallel environments (CPU threads will handle these)
    print(f"Creating {n_envs} parallel environments...")
    env = make_vec_env(
        lambda: DriveGameEnv(render_mode=None, max_episode_steps=1000),
        n_envs=n_envs,
        seed=42,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/",
        name_prefix="ppo_drive",
    )
    
    # Create agent
    print("Initializing PPO agent...\n")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )
    
    print("Training (watch CPU usage - should be near 100%)...\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            log_interval=5,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
    finally:
        print("\nSaving final model...")
        model.save("models/ppo_drive_final")
        print("✓ Saved to models/ppo_drive_final.zip")
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--envs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    args = parser.parse_args()
    
    train_agent(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        batch_size=args.batch,
        learning_rate=args.lr,
    )