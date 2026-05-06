import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set professional style (British English)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2.5

# Create graphs directory
os.makedirs('graphs', exist_ok=True)

# Read training data
df = pd.read_csv('logs/monitor.csv', skiprows=1)

print("Loading training data...")
print(f"Total episodes: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# Graph 1: Learning Curve - Episode Reward Over Time
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Smooth the rewards with rolling average for clarity
window = 50
df['reward_smoothed'] = df['r'].rolling(window=window, center=True).mean()

# Plot raw + smoothed
ax.scatter(df['t'], df['r'], alpha=0.3, s=10, label='Raw Episode Reward', color='steelblue')
ax.plot(df['t'], df['reward_smoothed'], linewidth=2.5, label=f'Moving Avg (window={window})', color='darkblue')

ax.set_xlabel('Total Timesteps', fontsize=12, fontweight='bold')
ax.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
ax.set_title('Learning Curve: Agent Reward Progression Over 5M Training Steps', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, df['t'].max())

# Add milestones
milestones = [100000, 500000, 1000000, 2500000, 5000000]
for milestone in milestones:
    if milestone <= df['t'].max():
        ax.axvline(x=milestone, color='red', linestyle='--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig('graphs/01_learning_curve.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 01_learning_curve.png")
plt.close()

# ============================================================================
# Graph 2: Episode Length Over Time (Agent Survival)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

df['length_smoothed'] = df['l'].rolling(window=50, center=True).mean()

ax.scatter(df['t'], df['l'], alpha=0.3, s=10, label='Raw Episode Length', color='forestgreen')
ax.plot(df['t'], df['length_smoothed'], linewidth=2.5, label='Moving Avg (window=50)', color='darkgreen')

ax.set_xlabel('Total Timesteps', fontsize=12, fontweight='bold')
ax.set_ylabel('Episode Length (frames)', fontsize=12, fontweight='bold')
ax.set_title('Survival Time: How Long the Agent Lasted Over Training', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

# Add max frames line
max_frames = 3000
ax.axhline(y=max_frames, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Max Game Duration')

plt.tight_layout()
plt.savefig('graphs/02_episode_length.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 02_episode_length.png")
plt.close()

# ============================================================================
# Graph 3: Reward Distribution Histogram
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Create histogram of all rewards
ax.hist(df['r'], bins=100, color='steelblue', edgecolor='black', alpha=0.7)

# Add statistics lines
mean_reward = df['r'].mean()
median_reward = df['r'].median()
ax.axvline(mean_reward, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_reward:.0f}')
ax.axvline(median_reward, color='orange', linestyle='--', linewidth=2.5, label=f'Median: {median_reward:.0f}')

ax.set_xlabel('Episode Reward', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency (Number of Episodes)', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Episode Rewards Across All Training', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graphs/03_reward_distribution.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 03_reward_distribution.png")
plt.close()

# ============================================================================
# Graph 4: Performance Metrics Summary
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Performance Summary Dashboard', fontsize=16, fontweight='bold', y=0.995)

# Subplot 1: Reward Statistics Over Time
ax = axes[0, 0]
rolling_window = 100
df['reward_mean'] = df['r'].rolling(window=rolling_window).mean()
df['reward_std'] = df['r'].rolling(window=rolling_window).std()
ax.plot(df['t'], df['reward_mean'], linewidth=2, label='Mean', color='darkblue')
ax.fill_between(df['t'], df['reward_mean'] - df['reward_std'], df['reward_mean'] + df['reward_std'],
                 alpha=0.2, color='blue', label='±1 Std Dev')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Reward')
ax.set_title('Reward Mean ± Std Dev')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Episode Length Statistics
ax = axes[0, 1]
df['length_mean'] = df['l'].rolling(window=rolling_window).mean()
df['length_std'] = df['l'].rolling(window=rolling_window).std()
ax.plot(df['t'], df['length_mean'], linewidth=2, label='Mean Length', color='darkgreen')
ax.fill_between(df['t'], df['length_mean'] - df['length_std'], df['length_mean'] + df['length_std'],
                 alpha=0.2, color='green', label='±1 Std Dev')
ax.axhline(y=3000, color='red', linestyle='--', alpha=0.5, label='Max Duration')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Length (frames)')
ax.set_title('Episode Duration Mean ± Std Dev')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 3: Reward Percentiles
ax = axes[1, 0]
percentiles = df['r'].rolling(window=rolling_window).quantile(0.25)
median = df['r'].rolling(window=rolling_window).quantile(0.50)
high_percentiles = df['r'].rolling(window=rolling_window).quantile(0.75)
ax.fill_between(df['t'], percentiles, high_percentiles, alpha=0.3, color='orange', label='25th-75th Percentile')
ax.plot(df['t'], median, linewidth=2, color='darkorange', label='Median')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Reward')
ax.set_title('Reward Range (Interquartile)')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 4: Training Progress (% episodes maxing out)
ax = axes[1, 1]
rolling_window = 100
df['max_out'] = (df['l'] >= 2999).astype(int)
df['max_out_pct'] = df['max_out'].rolling(window=rolling_window).mean() * 100
ax.plot(df['t'], df['max_out_pct'], linewidth=2.5, color='purple')
ax.fill_between(df['t'], 0, df['max_out_pct'], alpha=0.3, color='purple')
ax.set_xlabel('Timesteps')
ax.set_ylabel('% Episodes Maxing Out')
ax.set_title('Consistency: % of Episodes Reaching Max Duration')
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/04_performance_dashboard.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 04_performance_dashboard.png")
plt.close()

# ============================================================================
# Graph 5: V3 vs V5 Performance Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

versions = ['V3', 'V5']
peak_scores = [550, 3000]
avg_scores = [240, 2138]

x = np.arange(len(versions))
width = 0.35

bars1 = ax.bar(x - width/2, peak_scores, width, label='Peak Score', color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, avg_scores, width, label='Average Score', color='coral', edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score (frames)', fontsize=13, fontweight='bold')
ax.set_title('V3 vs V5: Performance Improvement Comparison', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=13, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 3500)

# Add score value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 80,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 80,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add improvement percentages with arrow annotations
improvement_peak = ((3000 - 550) / 550) * 100
improvement_avg = ((2138 - 240) / 240) * 100

# Peak score improvement label
ax.annotate('', xy=(1.165, 2900), xytext=(-0.165, 550),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkblue', alpha=0.5))
ax.text(0.5, 1800, f'+{improvement_peak:.0f}%', ha='center', fontsize=14,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=2),
        fontweight='bold')

# Average score improvement label
ax.annotate('', xy=(1.235, 2050), xytext=(-0.235, 240),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkorange', alpha=0.5))
ax.text(0.5, 700, f'+{improvement_avg:.0f}%', ha='center', fontsize=14,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.7, edgecolor='black', linewidth=2),
        fontweight='bold')

plt.tight_layout()
plt.savefig('graphs/05_v3_vs_v5_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 05_v3_vs_v5_comparison.png")
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("TRAINING SUMMARY STATISTICS")
print("="*60)
print(f"Total Episodes: {len(df)}")
print(f"Total Timesteps: {df['t'].max():,}")
print(f"\nReward Statistics:")
print(f"  Minimum: {df['r'].min():.2f}")
print(f"  Maximum: {df['r'].max():.2f}")
print(f"  Mean: {df['r'].mean():.2f}")
print(f"  Median: {df['r'].median():.2f}")
print(f"  Std Dev: {df['r'].std():.2f}")
print(f"\nEpisode Length Statistics:")
print(f"  Minimum: {df['l'].min():.0f} frames")
print(f"  Maximum: {df['l'].max():.0f} frames")
print(f"  Mean: {df['l'].mean():.2f} frames")
print(f"  % at Max (>=2999 frames): {(df['l'] >= 2999).sum() / len(df) * 100:.1f}%")
print(f"\nImprovement from Start to End:")
print(f"  Early Avg (first 1000 episodes): {df['r'].iloc[:1000].mean():.2f}")
print(f"  Late Avg (last 1000 episodes): {df['r'].iloc[-1000:].mean():.2f}")
print(f"  Improvement: {(df['r'].iloc[-1000:].mean() - df['r'].iloc[:1000].mean()):.2f} (+{((df['r'].iloc[-1000:].mean() - df['r'].iloc[:1000].mean()) / abs(df['r'].iloc[:1000].mean()) * 100):.1f}%)")
print("="*60)
print("\nAll graphs saved to: graphs/")
