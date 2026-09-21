import matplotlib.pyplot as plt
import numpy as np

# 数据准备
settings = ['llama\n(baseline)', 'block16\nstep16', 'block16\nstep8', 'block32\nstep8', 'block64\nstep8']
models = ['Qwen3-8B-Base', 'SmolLM3-3B-Base', 'gemma-3-27b-pt', 'gpt2-large']

# Throughput
throughput = [3771.3286131914056, 6091.050870835777, 10701.55792919795, 15064.407949534721]

# GEN PPL
ppl = np.array([
    [6.32, 6.67, 15.32, 16.45, ],
    [6.50, 6.85, 17.82, 19.44, ],
    [7.88, 7.97, 18.47, 20.90],
    [8.17, 8.39, 26.83, 30.11]
])

# 创建图表
fig, ax1 = plt.subplots(figsize=(14, 8))

x = np.arange(len(settings))
width = 0.18
colors = ['#4E79A7', '#F28E2B', '#76B7B2', '#E15759']

# 绘制PPL柱状图（无误差条）
for i, (model, color) in enumerate(zip(models, colors)):
    ax1.bar(x + (i - 1.5) * width, ppl[i], width, 
            label=model, color=color, alpha=0.8)

ax1.set_xlabel('Configuration', fontsize=18, fontweight='bold')
ax1.set_ylabel('Generation PPL (↓ better)', fontsize=18, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(settings, fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelsize=18)
ax1.legend(loc='upper left', framealpha=0.9, fontsize=14)
ax1.set_ylim(0, 70)
ax1.grid(axis='y', alpha=0.3)

# 第二个Y轴绘制Throughput（无阴影）
ax2 = ax1.twinx()
ax2.plot(x, throughput, '--', marker='s', markersize=12, 
         linewidth=3, label='Throughput', color='#59A14F')
ax2.set_ylabel('Throughput (tokens/s, ↑ better)', fontsize=18, 
               fontweight='bold', color='#59A14F')
ax2.set_ylim(0, 800)
ax2.tick_params(axis='y', labelsize=18, colors='#59A14F')
ax2.legend(loc='upper right', fontsize=14)

# 标注throughput数值
for i, val in enumerate(throughput):
    ax2.annotate(f'{val:.0f}', (x[i], val), textcoords="offset points", 
                 xytext=(0, 10), ha='center', fontsize=14, 
                 fontweight='bold', color='#59A14F')

plt.title('Speculative Decoding: Quality vs Speed Trade-off', 
          fontsize=22, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('combined_plot_v6.png', dpi=150, bbox_inches='tight')
plt.show()