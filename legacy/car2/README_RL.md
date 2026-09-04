# AI Drive Game - Reinforcement Learning

This is an RL implementation of your driving game using **PPO (Proximal Policy Optimization)** from Stable-Baselines3.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Files overview

- **`ai_drive_game_env.py`** — Gymnasium wrapper around your game
  - Converts pygame game into standard RL environment
  - Handles `reset()`, `step(action)`, `render()`
  - Observation: [car_lane, distance_to_obstacle_per_lane]
  - Action space: {0, 1, 2, 3} representing lanes {1, 2, 3, 4}
  - Reward: +1 per frame survived, -100 on crash

- **`train_rl_agent.py`** — Training script
  - Uses 4 parallel environments for faster training
  - Saves checkpoints every 10,000 steps
  - Evaluates every 50,000 steps
  - Logs to tensorboard

- **`run_agent.py`** — Play with trained agent
  - Load a trained model and watch it drive
  - Shows score, FPS, etc.

## Training

### Quick start (default: 500k timesteps, 4 parallel envs)
```bash
python train_rl_agent.py --train
```

### Custom training
```bash
python train_rl_agent.py --train --timesteps 1000000 --envs 8
```

**Expected results:**
- After ~100k steps: Agent avoids basic obstacles, scores ~500-1000
- After ~250k steps: Agent gets better at dodging, scores ~1000-2000
- After ~500k steps: Solid driving, scores ~2000-4000+

**Training will save:**
- `models/ppo_drive_*.zip` — Checkpoints every 10k steps
- `models/best/` — Best model during evaluation
- `models/ppo_drive_final.zip` — Final model
- `logs/` — Tensorboard logs (if you want to monitor training)

## Evaluation

### Watch trained agent
```bash
python run_agent.py models/ppo_drive_final --episodes 5
```

### Evaluate in script
```python
from train_rl_agent import evaluate_agent

evaluate_agent("models/ppo_drive_final", n_episodes=10, render=True)
```

## How it works

### Environment
The Gymnasium wrapper converts your pygame game into a standard RL interface:

```python
obs, info = env.reset()  # Start episode
action = agent.predict(obs)  # Agent chooses lane (0-3)
obs, reward, done, truncated, info = env.step(action)  # Execute
```

**State representation** (simple but effective):
- Normalized car lane position (0-1)
- Distance to nearest obstacle in each lane (0-1)
  - 0 = collision imminent
  - 1 = clear lane

**Reward:**
- +1 per frame alive (encourages surviving)
- -100 on crash (big penalty)

### Algorithm: PPO
- **Stable** — won't diverge like DQN
- **Sample efficient** — learns quickly
- **Works well with discrete actions** — perfect for 4 lanes
- Standard hyperparameters included (learning rate, entropy, etc.)

### Training pipeline
1. **Parallel environments** — 4 games run simultaneously
2. **Rollout collection** — Agent plays 2048 steps across all envs
3. **Advantage computation** — Calculates which actions were good/bad
4. **Policy update** — Updates agent weights using PPO loss
5. **Repeat** — Continues for 500k total steps

## Troubleshooting

**"Module not found" error:**
- Make sure you're in the directory with `ai_drive_game_env.py`
- Or add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**Training is slow:**
- Increase `--envs` (e.g., `--envs 8`) to run more games in parallel
- Reduce `--timesteps` for quick test

**Agent scores 0:**
- Give it more training time — PPO takes ~50k-100k steps to show improvement
- Check if observation space is correct (should see lane + distances)

**Want to monitor training?**
```bash
tensorboard --logdir logs/
```
Then open http://localhost:6006

## Next steps

Once training works, you could:

1. **Try different observation representations**
   - Add speed, distance to crash, time since last dodge
   - Or use raw JPEG frames (requires CNN policy)

2. **Tune hyperparameters**
   - Learning rate, entropy coefficient, GAE lambda
   - See Stable-Baselines3 docs for details

3. **Change reward function**
   - Bonus for staying in clear lanes
   - Penalty for lane changes (smoother driving)
   - Reward based on speed

4. **Test other algorithms**
   - `from stable_baselines3 import A2C, DDPG`
   - Different trade-offs between stability and efficiency

5. **Deploy to production**
   - Export trained agent to ONNX for inference
   - Run on edge devices without numpy/pytorch overhead

Good luck! 🚗
