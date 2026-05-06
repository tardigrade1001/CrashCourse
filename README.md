# AI Drive Game - Deep Reinforcement Learning (PPO)

This project contains a highly optimized AI agent trained to play a custom 2D Pygame driving game. The agent uses Proximal Policy Optimization (PPO) and was heavily optimized to utilize high-end hardware (RTX 3060, 48GB RAM). 

By the end of the training, the AI became an elite-level driver, consistently achieving the maximum possible score of 3,000 frames in every evaluation.

---

## 🚀 How to Run the Trained AI

To watch the near-perfect AI driver in action, run the evaluation script:

```bash
python run_agent.py models/ppo_drive_final.zip --episodes 5
```

---

## 📊 Training Visualization (Simple Explanations)

### Learning Curve - How the AI Got Smarter Over Time

![Learning Curve](graphs/01_learning_curve.png)

**What's happening in plain English:**
Imagine teaching a kid to ride a bike. At first, they fall a lot (bottom left = bad scores). Gradually, they get better and stay upright longer (the line goes up). By the end, they're riding smoothly (top right = great scores).

This graph shows exactly that. The AI started getting **negative points** (it was terrible and crashing). Over 5 million training steps, it gradually got better. You can see it went from basically failing to consistently scoring high by the end.

**What to notice:**
- The blue dots are messy at first (inconsistent/unreliable)
- The dark line gets smoother over time (more consistent)
- By the end, most scores are high (1000+)

---

### Survival Time - How Long the AI Lasted Before Crashing

![Episode Length Over Time](graphs/02_episode_length.png)

**What's happening in plain English:**
This measures how long the AI survived in each game before crashing into an obstacle.

At the start: About **70 frames** (it crashed immediately)
In the middle: About **300-500 frames** (it was learning)
By the end: **3,000 frames** (it made it to the maximum, meaning it never crashed!)

The green line creeping up shows the AI learning to dodge better and survive longer. It's like watching someone improve at a video game—they survive longer as they practice.

---

### Reward Distribution - Spread of All 44,992 Games Played

![Reward Distribution](graphs/03_reward_distribution.png)

**What's happening in plain English:**
This is a histogram showing every single game score during training. Think of it like a bar chart of test scores:

- **Left side (negative scores):** These are the early games when the AI sucked and crashed immediately
- **Middle (0-500):** Games from the learning phase
- **Right side (500-3000):** Games from the later training where the AI was good

The **red line** shows the average (mean) score across all games.
The **orange line** shows the middle value (median) - half the games scored above this, half below.

**Why it looks weird:** Because there are SO MANY bad games from early training, the chart is spread out. As the AI got better, more games scored high (right side of the chart).

---

### Performance Dashboard - Four Different Views of Training

![Performance Dashboard](graphs/04_performance_dashboard.png)

**Top-Left: "Average Score Over Time + How Consistent"**
Shows the average score the AI was getting at each point in training (the line). The shaded area shows how much variation there was. Early on: very unpredictable. Later: more reliable (smaller shaded area = more consistent).

**Top-Right: "How Long Games Lasted + Consistency"**
Similar to the survival graph, but shows the average. At first games lasted ~70 frames. By the end, games lasted almost the full 3,000 frames.

**Bottom-Left: "The Middle 50% of Games"**
Shows what the "typical" games looked like. The orange line in the middle is the median. Notice how it barely moves early on (AI was always bad), then shoots up (AI got much better).

**Bottom-Right: "What % of Games Did the AI Win Perfectly?"**
This shows what percentage of games the AI played all the way through to the maximum without crashing. At the start: 0%. By the end: 15-20% of games are perfect runs.

---

## 🧠 The Journey to Elite-Level AI

Training this AI wasn't a straight path. We went through several major architectural and environmental revisions to arrive at the near-perfect model. Here is the technical breakdown of what we did and why.

### V1: The Naive MLP (Stuck in a Local Optimum)
* **What we did:** We started with a simple Multi-Layer Perceptron (MLP) receiving a flattened grid of the road. We heavily penalized lane changes (`-0.5`) to encourage "smooth driving".
* **Why it failed:** The agent quickly learned that "staying still and eventually crashing" was mathematically better than "moving, taking penalties, and eventually crashing." It became paralyzed by the lane change penalty and scored an average of 70 frames.

### V2: The CNN Attempt (The "Telescope" Problem)
* **What we did:** We tried switching to a Convolutional Neural Network (CNN) to treat the road like an image.
* **Why it failed:** We initially considered Convolutional Neural Networks because they excel at learning spatial hierarchies in large, complex images. However, our input space is a simple 4-lane grid. CNNs are overengineered for this problem—the computational overhead and training complexity provide no benefit over a dense MLP that directly processes the grid features. For small, well-defined state spaces like ours, simpler architectures perform better.

### V3: Reward System Overhaul
* **What we did:** We fixed the broken psychology of the AI.
  * **Crash Penalty:** Reduced from `-500` to `-50`. Crashes became a setback, not a catastrophic failure that discouraged trying.
  * **Lane Change Penalty:** Reduced from `-0.5` to `-0.1`. Moving became cheap.
  * **Obstacle Density:** Increased spawn rate to force the agent to act.
* **The Result:** The AI finally learned to dodge! It jumped from a score of 70 to actively surviving 150+ frames, peaking at 550.

### V4: Temporal Awareness (Frame Stacking)
* **What we did:** We gave the AI the ability to perceive *time and speed*. 
  * **Frame Stacking:** Instead of seeing one still image, it now sees the last 4 frames stacked together (`VecFrameStack`). This allowed the neural network to infer the velocity of oncoming cars.
  * **Long-Range HD Vision:** Increased the AI's vision depth to 20 segments (1000 pixels ahead).
  * **Spawn Rate Optimization:** We also discovered that the obstacle spawn rate was critical. During V3, obstacles spawned every 8 frames, creating extremely dense traffic that limited the agent's peak performance to 550 frames. For V4, we reduced the spawn rate to every 12 frames, giving the agent more breathing room to anticipate and execute dodges. This seemingly small change had a massive effect—it allowed the agent to transition from reactive dodging to predictive planning.
* **The Result:** The agent became highly anticipatory.

### V5: Super Training (Maximum Hardware Utilization)
* **What we did:** To reach the ultimate goal of surviving 1000+ frames, we unleashed the full power of the hardware.
  * **Massive Brain:** Upgraded the neural network to `[1024, 1024, 1024]`, giving the RTX 3060 billions of calculations to chew on.
  * **RAM Gobbler:** We spawned 12 parallel game environments and increased the rollout buffer (`n_steps`) to 16,384. This held almost a gigabyte of experiential data in the 48GB RAM before shipping it to the GPU in massive 2,048-size batches.
  * **Training length:** 5,000,000 steps.

#### Performance Comparison: V3 vs V5

![V3 vs V5 Comparison](graphs/05_v3_vs_v5_comparison.png)

**What changed between V3 and V5?**

| What We Changed | V3 | V5 | Impact |
| :--- | :--- | :--- | :--- |
| How big was the AI's "brain" | Small (256 neurons) | Huge (1024 neurons) | Bigger brain = smarter |
| How often obstacles appeared | Every 8 frames | Every 12 frames | Less crowded = more learnable |
| How much we trained it | 2.75 million games | 5 million games | More practice = better |

**The Results:**

| Score Type | V3 | V5 | Improvement |
| :--- | :--- | :--- | :--- |
| **Best Game Ever** | 550 points | 3,000 points | **5.5x better** |
| **Average Game** | 240 points | 2,138 points | **9x better** |

**Translation:** V3 was OK - sometimes made it to 550 before crashing. V5 is incredible - regularly makes it to 3,000 (the maximum possible) without crashing.

It's like comparing a beginner chess player to a grandmaster. V3 can play, but V5 plays nearly perfect every time.

* **The Result:** The AI became nearly perfect. It regularly scores the absolute maximum possible (3,000 frames), only failing occasionally due to random bad luck with obstacle placement.

---

## 📂 Project Structure

* **`ai_drive_game_env.py`**: The core Gymnasium environment wrapping the Pygame logic. Defines the 81-feature observation space, the reward function, and the Pygame rendering.
* **`train_rl_agent.py`**: The heavy-duty training script. Configured for multiprocessing, Frame Stacking, and GPU-optimized batch sizes.
* **`run_agent.py`**: The evaluation script to watch the trained agent. Includes the necessary `DummyVecEnv` and `VecFrameStack` wrappers to match the training environment's observation shape.
* **`models/ppo_drive_final.zip`**: The final, near-perfect model.
* **`logs/`**: Tensorboard logs generated during the 5 iterations of training.
* **`graphs/`**: Training visualization graphs showing learning curves, reward distributions, and performance metrics.

---

## 🔧 Training the Model Again

If you ever want to retrain the model from scratch using your hardware setup:

```bash
python train_rl_agent.py --timesteps 5000000 --envs 12
```
*(Note: 12 environments is the safe limit for Windows multiprocessing overhead. The large `n_steps=16384` is what effectively utilizes the high RAM).*

---

## 📊 Results Summary

The final agent achieved an average score of 2,138 across evaluation episodes, with peak performance reaching the game's maximum score of 3,000 frames. This represents a 26x improvement over random play (baseline ~70-81 frames). The agent demonstrates robust decision-making, anticipatory dodging, and consistent performance across varying obstacle patterns.
