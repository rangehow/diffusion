import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

# 左侧：PPL热力图
settings = ['llama', 'bs16_st16', 'bs16_st8', 'bs32_st8', 'bs64_st8']
models = ['Qwen3-8B', 'SmolLM3-3B', 'Gemma-27B', 'GPT2-large']

ppl = np.array([
    [6.32, 6.67, 15.32, 16.45, 18.95],
    [6.50, 6.85, 17.82, 19.44, 22.51],
    [7.88, 7.97, 18.47, 20.90, 23.92],
    [8.17, 8.39, 26.83, 30.11, 35.72]
])

sns.heatmap(ppl, annot=True, fmt='.2f', cmap='RdYlGn_r',
            xticklabels=settings, yticklabels=models, ax=axes[0],
            cbar_kws={'label': 'PPL (lower is better)'})
axes[0].set_title('Generation PPL Heatmap', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Configuration')
axes[0].set_ylabel('Evaluation Model')

# 右侧：Throughput条形图（横向）
throughput = [50.65, 68.69, 130.70, 252.43, 478.67]
colors_bar = plt.cm.Greens(np.linspace(0.3, 0.9, len(throughput)))
axes[1].barh(settings, throughput, color=colors_bar, edgecolor='black')
axes[1].set_xlabel('Throughput (tokens/s)', fontweight='bold')
axes[1].set_title('Speed Comparison', fontweight='bold', fontsize=12)
for i, v in enumerate(throughput):
    axes[1].text(v + 5, i, f'{v:.1f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('heatmap_throughput.png', dpi=150)
plt.show()