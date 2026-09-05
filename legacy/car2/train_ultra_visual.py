"""
Train Ultra + Visualize
Just runs training with train_ultra.py
To see what's happening, run demo_ultra.py in another terminal
"""

import subprocess
import sys

print("=" * 70)
print("TRAIN + VISUALIZE")
print("=" * 70)
print("\nRunning training in background and visualization side-by-side\n")

print("Starting demo_ultra.py (visualization)...")
print("This shows what the environment looks like")
print("(Close the window when done)\n")

# Run demo in a subprocess
subprocess.Popen([sys.executable, "demo_ultra.py"])

print("\nStarting training (train_ultra.py)...")
print("Training will run while you watch the visualization\n")

# Run training
subprocess.run([sys.executable, "train_ultra.py", "--timesteps", "500000", "--envs", "16"])