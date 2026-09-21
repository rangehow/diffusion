import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
import seaborn as sns

# 设置绘图风格
sns.set_style("ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

def plot_corrected_distribution():
    # --- 1. 数据生成配置 ---
    x = np.linspace(0, 1, 1000)
    
    true_loc = 0.5
    true_scale = 0.4
    true_pdf = norm.pdf(x, loc=true_loc, scale=true_scale)
    
    np.random.seed(10)
    samples = np.concatenate([
        np.random.normal(0.35, 0.1, 4),
        np.random.normal(0.65, 0.1, 4),
        [0.1, 0.9]
    ])
    samples = np.clip(samples, 0.05, 0.95)
    
    kde = gaussian_kde(samples, bw_method=0.7)
    teacher_pdf = kde(x)

    # --- 2. 绘图 ---
    # 【修改点 1】设置画布大小：figsize=(宽, 高)，单位是英寸
    # 如果觉得原来的 (5, 5) 太方，可以改成 (6, 4) 或者其他比例
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=600)

    max_val = max(np.max(teacher_pdf), np.max(true_pdf))
    arrow_top = max_val * 1.25 
    
    # 画线和箭头
    ax.plot(x, teacher_pdf, color='#1f77b4', linestyle='--', linewidth=3.5, alpha=0.9,
            label=r'$p_{\mathrm{statistic}}$')
    ax.fill_between(x, teacher_pdf, color='#1f77b4', alpha=0.15)
    
    ax.plot(x, true_pdf, color='#74c476', linewidth=3, alpha=0.8, 
            label=r'$p_{\mathrm{true}}$')

    ax.vlines(samples, ymin=0, ymax=arrow_top, color='#d62728', 
              linewidth=1.5, alpha=0.8, zorder=5)
    
    # ax.scatter(samples, [arrow_top]*len(samples), color='#d62728', 
    #            s=50, marker='-', zorder=6, label=r'$p_{\mathrm{data}}$')

    # --- 3. 坐标轴修饰 ---
    ax.set_xlim(0, 1)
    
    # 【修改点 2】减少顶部留白
    # 原来是 arrow_top * 1.25，导致上面空太多。
    # 改成 * 1.15 左右，只要刚好能放下图例且不遮挡箭头即可。
    ax.set_ylim(0, arrow_top * 1.15) 
    
    ax.set_xticks([]) 
    ax.set_yticks([])
    
    ax.set_xlabel('Semantic Space', fontsize=16, weight='bold')
    ax.set_ylabel('Probability Density', fontsize=16, weight='bold')
    
    # 【修改点 3】调整图例位置
    # bbox_to_anchor=(0.5, 1.0) 表示图例的顶部中心点对齐到绘图框的上边缘
    # 这样图例就会紧贴着上面，消除了多余的空隙
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), 
              ncol=3, frameon=False, fontsize=14, handlelength=1.5)
    
    sns.despine(left=True, bottom=False)
    
    # tight_layout 会自动切除整个图片周围多余的白边
    plt.tight_layout()
    plt.savefig('distribution_fixed_legend.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_corrected_distribution()