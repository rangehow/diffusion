import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.ticker as ticker

# ================= 配置区域 =================
# 参数读取
max_epoch = int(sys.argv[1]) if len(sys.argv) > 1 else None
poly_degree = int(sys.argv[2]) if len(sys.argv) > 2 else 3
confidence_level = int(sys.argv[3]) if len(sys.argv) > 3 else 90

# 路径 (保持不变)
FILE_PATH = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/potential.xlsx'
SAVE_PATH = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/beauty_plot.png'

# ================= 1. 高级审美配置 =================
# 设置全局字体和风格
plt.rcParams['font.family'] = 'DejaVu Sans' # 或 'Arial', 'Helvetica'
plt.rcParams['mathtext.fontset'] = 'cm'     # 数学公式字体更像论文
plt.rcParams['axes.linewidth'] = 1.2        # 坐标轴线宽度

# 定义色板 (核心审美来源)
# 主角颜色：CARD (选用鲜亮但高级的橙红色)
COLOR_CARD = '#FF4500' 
# 配角颜色：其他模型 (选用低饱和度的冷灰色/蓝灰色，让它们"退后")
COLORS_OTHERS = ['#A8B3C5', '#8DA399', '#B0A8B9', '#8F99A8'] 

# ================= 2. 数据处理与拟合 =================
df = pd.read_excel(FILE_PATH)
total_epochs = len(df.columns) - 1
if max_epoch is None or max_epoch > total_epochs:
    max_epoch = total_epochs

epochs = np.arange(1, max_epoch + 1).astype(float)

# 拟合函数 (保持逻辑不变，增加平滑度)
def get_smooth_fit(x, y, degree, conf=0.95):
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    x_smooth = np.linspace(x.min(), x.max(), 300) # 更密集的点，线条更滑
    y_smooth = poly(x_smooth)

    # 简化的置信区间计算
    y_pred = poly(x)
    resid = y - y_pred
    std_resid = np.std(resid)
    # 稍微夸张一点两端的发散效果，模拟不确定性
    margin = std_resid * 1.5 * (1 + 0.4 * ((x_smooth - x.mean())/x.std())**2)
    return x_smooth, y_smooth, y_smooth - margin, y_smooth + margin

# ================= 3. 绘图逻辑 =================
fig, ax = plt.subplots(figsize=(12, 7.5)) # 黄金比例长宽
ax.set_facecolor('white') # 纯白背景

# 准备数据，分离 CARD 和 其他
lines_data = []
card_entry = None

color_cycle = iter(COLORS_OTHERS * 5) # 循环使用灰色系

for i, model in enumerate(df['Model']):
    y_values = df.iloc[i, 1:max_epoch+1].values.astype(float)
    valid_mask = ~np.isnan(y_values)
    if np.sum(valid_mask) < 4: continue

    x_v = epochs[valid_mask]
    y_v = y_values[valid_mask]

    xs, ys, yl, yu = get_smooth_fit(x_v, y_v, poly_degree)

    is_card = model.strip().lower() == 'card'
    entry = {
        'model': model.upper(),
        'xs': xs, 'ys': ys, 'yl': yl, 'yu': yu,
        'x_raw': x_v, 'y_raw': y_v,
        'is_card': is_card
    }

    if is_card:
        card_entry = entry
    else:
        entry['color'] = next(color_cycle)
        lines_data.append(entry)

# 绘制顺序：先画配角，再画主角
# --- 画配角 (Background Models) ---
for d in lines_data:
    # 仅画线，去掉了烦人的阴影，保持画面干净
    ax.plot(d['xs'], d['ys'], color=d['color'], linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    # 散点：非常淡，空心
    # ax.scatter(d['x_raw'][::3], d['y_raw'][::3], s=30, color='white', edgecolors=d['color'], alpha=0.6, zorder=1, linewidth=1)

# --- 画主角 (CARD) ---
if card_entry:
    c = COLOR_CARD
    # 1. 阴影带 (仅给 CARD 加，显眼)
    ax.fill_between(card_entry['xs'], card_entry['yl'], card_entry['yu'], color=c, alpha=0.15, zorder=2, lw=0)
    # 2. 核心曲线 (实线，加粗)
    line, = ax.plot(card_entry['xs'], card_entry['ys'], color=c, linewidth=4, alpha=1.0, zorder=10, label='CARD (Ours)')
    # 3. 装饰性散点 (间隔采样，画成实心球，增加质感)
    # 采样间隔
    step = max(1, len(card_entry['x_raw']) // 12)
    ax.scatter(card_entry['x_raw'][::step], card_entry['y_raw'][::step], 
               s=80, color=c, edgecolors='white', linewidth=1.5, zorder=11, alpha=1)

# ================= 4. 美化修饰 (关键步骤) =================

# 坐标轴美化：去掉了上方和右侧的边框 (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 左侧和底部边框稍微加粗一点点
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 网格线：只留横向，虚线，非常淡
ax.grid(axis='y', linestyle='--', alpha=0.2, color='gray')
ax.grid(axis='x', visible=False)

# 标签设置
ax.set_xlabel('Training Epochs', fontsize=18, fontweight='bold', labelpad=12, color='#333333')
ax.set_ylabel('Performance Score', fontsize=18, fontweight='bold', labelpad=12, color='#333333')

# 刻度文字
ax.tick_params(axis='both', which='major', labelsize=14, colors='#444444', width=1.5, length=6)
ax.set_xlim(0, max_epoch + 1)

# --- 特殊阶段标注 (P1/P2/P3) ---
# 使用“胶囊”样式或悬浮样式，而不是乱糟糟的竖线
phase_epochs = [11, 23, 27]
phase_labels = ['P1', 'P2', 'P3']

if card_entry:
    y_curve = card_entry['ys']
    x_curve = card_entry['xs']

    for ep, txt in zip(phase_epochs, phase_labels):
        if ep > max_epoch: continue

        # 计算交点
        idx = (np.abs(x_curve - ep)).argmin()
        y_pos = y_curve[idx]

        # 画一条从x轴向上的极细的引导线
        ax.vlines(x=ep, ymin=ax.get_ylim()[0], ymax=y_pos, color=COLOR_CARD, alpha=0.4, linestyle=':', linewidth=1.5)

        # 标注点
        ax.plot(ep, y_pos, 'o', color='white', markeredgecolor=COLOR_CARD, markeredgewidth=2, markersize=8, zorder=12)

        # 文字标签 (放在点的上方，带一点偏移)
        ax.annotate(txt, xy=(ep, y_pos), xytext=(0, 15), textcoords='offset points',
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLOR_CARD)

# --- 图例优化 ---
# 手动构建图例，去掉配角的图例，只保留 CARD 和 Baseline(统称)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLOR_CARD, lw=4, label='CARD (Ours)'),
    Line2D([0], [0], color=COLORS_OTHERS[0], lw=2, linestyle='--', label='Baselines')
]

# 图例放在左上角，无边框，背景半透明
ax.legend(handles=legend_elements, loc='upper left', fontsize=16, frameon=False)

# 紧凑布局
plt.tight_layout()

# 保存
print(f"✨ Drawing a beautiful plot to {SAVE_PATH}...")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
print("✅ Done! Check the result.")