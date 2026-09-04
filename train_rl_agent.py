import os
import sys
import argparse
import numpy as np

# Inject path at the very top for main and all potential subprocesses
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

# Helper to ensure subprocesses have the right path
def make_env():
    def _init():
        import sys
        import os
        # Re-inject inside the subprocess
        cdir = os.path.dirname(os.path.abspath(__file__))
        if cdir not in sys.path:
            sys.path.insert(0, cdir)
        from ai_drive_game_env import DriveEnv
        return DriveEnv()
    return _init

class VisualProgressCallback(BaseCallback):
    def __init__(self, eval_freq=40000, verbose=0):
        super(VisualProgressCallback, self).__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"--- Running visual evaluation at step {self.num_timesteps} ---")
            from ai_drive_game_env import DriveEnv
            def make_visual_env():
                return DriveEnv(render_mode="human")
            env = DummyVecEnv([make_visual_env])
            env = VecFrameStack(env, n_stack=4)
            obs = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                done = dones[0]
                env.render()
            env.close()
        return True

def train(timesteps=5000000, n_envs=12, net_size=1024, net_layers=3, n_steps=16384, batch_size=2048):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecMonitor(env, "logs")
    env = VecFrameStack(env, n_stack=4)

    eval_env = SubprocVecEnv([make_env()])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/',
                             log_path='./logs/', eval_freq=20000,
                             deterministic=True, render=False)

    visual_callback = VisualProgressCallback(eval_freq=40000)

    checkpoint_path = "models/ppo_drive_final.zip"
    net_arch = [net_size] * net_layers

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        model = PPO.load(checkpoint_path, env=env, device="cuda")
        print(f"Checkpoint loaded. Continuing training...")
    else:
        print(f"No checkpoint found. Creating new model with network: {net_arch}")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            device="cuda",
            batch_size=batch_size,
            n_steps=n_steps,
            policy_kwargs=dict(net_arch=net_arch),
            learning_rate=5e-5,
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
        )

    print(f"Training: {timesteps:,} timesteps on {n_envs} envs")
    print(f"  Network: {net_layers}x{net_size} neurons")
    print(f"  Buffer size (n_steps): {n_steps:,}")
    print(f"  Batch size: {batch_size:,}")

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, visual_callback],
        progress_bar=True
    )
    model.save("models/ppo_drive_final")
    print("Training complete!")

if __name__ == "__main__":
    # Windows fix for multiprocessing
    if sys.platform == 'win32':
        import multiprocessing
        multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Train PPO agent on driving game")
    parser.add_argument("--timesteps", type=int, default=5000000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument("--net-size", type=int, default=1024, help="Neurons per layer in network")
    parser.add_argument("--net-layers", type=int, default=3, help="Number of layers in network")
    parser.add_argument("--n-steps", type=int, default=16384, help="Rollout buffer size")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for training")
    args = parser.parse_args()
    train(
        timesteps=args.timesteps,
        n_envs=args.envs,
        net_size=args.net_size,
        net_layers=args.net_layers,
        n_steps=args.n_steps,
        batch_size=args.batch_size
    )
