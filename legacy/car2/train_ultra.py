"""
Train PPO on ULTRA HARD MODE
Dense traffic, fast cars, short episodes = fast learning through mistakes
"""

import os
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

pygame.init()

from ai_drive_game_env_ultra import DriveGameEnvUltra

def train_ultra(
    total_timesteps=500000,
    n_envs=16,
    learning_rate=5e-4,  # Slightly higher LR for faster learning
    batch_size=256,
):
    """Train on ULTRA HARD MODE"""
    
    print("=" * 70)
    print("TRAINING PPO - ULTRA HARD MODE")
    print("Dense traffic, fast cars, short episodes = fast learning")
    print("=" * 70)
    print(f"Parallel environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Episode length: 500 steps (crashes = fast learning)")
    print(f"Spawn rate: ULTRA AGGRESSIVE (every 10 frames)")
    print(f"Car speed: 0.5-1.5 (2-3x faster)")
    print()
    
    os.makedirs("models", exist_ok=True)
    
    print(f"Creating {n_envs} parallel environments...")
    
    env = make_vec_env(
        lambda: DriveGameEnvUltra(render_mode=None, max_episode_steps=500),
        n_envs=n_envs,
        seed=42,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/",
        name_prefix="ppo_ultra",
    )
    
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
    
    print("Training ULTRA HARD MODE...\n")
    
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
        model.save("models/ppo_ultra_final")
        print("✓ Saved to models/ppo_ultra_final.zip")
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    
    args = parser.parse_args()
    
    train_ultra(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        batch_size=args.batch,
        learning_rate=args.lr,
    )