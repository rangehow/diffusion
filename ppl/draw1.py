import matplotlib.pyplot as plt
import numpy as np

# 数据
settings = ['llama', 'bs16_st16', 'bs16_st8', 'bs32_st8', 'bs64_st8']
models = ['Qwen3-8B', 'SmolLM3-3B', 'Gemma-27B', 'GPT2-large']

throughput = [50.65, 68.69, 130.70, 252.43, 478.67]

ppl = np.array([
    [6.32, 6.67, 15.32, 16.45, 18.95],
    [6.50, 6.85, 17.82, 19.44, 22.51],
    [7.88, 7.97, 18.47, 20.90, 23.92],
    [8.17, 8.39, 26.83, 30.11, 35.72]
])

ppl_std = np.array([
    [2.90, 2.65, 11.01, 18.08, 19.36],
    [2.96, 2.64, 13.76, 22.39, 24.04],
    [5.03, 3.14, 15.59, 30.52, 28.70],
    [4.05, 3.41, 19.13, 31.11, 35.22]
])

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 子图1: Throughput
ax1 = axes[0]
bars = ax1.bar(settings, throughput, color='#59A14F', alpha=0.8, edgecolor='black')
ax1.set_ylabel('Throughput (tokens/s)', fontweight='bold')
ax1.set_title('① Throughput Comparison', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, throughput):
    ax1.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom', fontsize=9)

# 子图2: PPL (分组柱状图)
ax2 = axes[1]
x = np.arange(len(settings))
width = 0.2
for i, (model, color) in enumerate(zip(models, colors)):
    ax2.bar(x + (i - 1.5) * width, ppl[i], width, label=model, color=color, alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(settings, rotation=45)
ax2.set_ylabel('Generation PPL', fontweight='bold')
ax2.set_title('② PPL by Model & Config', fontweight='bold')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(axis='y', alpha=0.3)

# 子图3: PPL STD (折线图)
ax3 = axes[2]
for i, (model, color) in enumerate(zip(models, colors)):
    ax3.plot(settings, ppl_std[i], marker='o', label=model, 
             color=color, linewidth=2, markersize=7)
ax3.set_ylabel('PPL Standard Deviation', fontweight='bold')
ax3.set_title('③ PPL Variance Analysis', fontweight='bold')
ax3.legend(fontsize=8)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('three_subplots.png', dpi=150)
plt.show()