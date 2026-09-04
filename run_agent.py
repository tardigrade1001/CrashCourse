import os
import sys
import argparse
sys.path.append(os.getcwd())
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from ai_drive_game_env import DriveEnv

def run_trained_agent(model_path, episodes=5):
    # Create the environment with human rendering
    def make_env():
        return DriveEnv(render_mode="human")

    # Must use DummyVecEnv + VecFrameStack to match training observations
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    for episode in range(episodes):
        obs = env.reset()
        done = False

        while not done:
            # Predict the action
            action, _states = model.predict(obs, deterministic=True)

            # Execute the action
            obs, rewards, dones, infos = env.step(action)
            done = dones[0]

            # Rendering is handled inside the environment
            env.render()

        # Access the score from the terminal info
        final_score = infos[0].get('score', 'N/A')
        print(f"Episode {episode + 1} finished. Score: {final_score}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the model zip file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to watch")
    args = parser.parse_args()

    run_trained_agent(args.model_path, args.episodes)
