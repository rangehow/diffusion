# ... (保留之前的代码，直到 plt.tight_layout() 附近) ...
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 参数：控制画到多少个 epoch（默认全部），多项式拟合阶数（默认3），置信水平（默认90），是否显示阴影（默认false）
max_epoch = int(sys.argv[1]) if len(sys.argv) > 1 else None
poly_degree = int(sys.argv[2]) if len(sys.argv) > 2 else 3
confidence_level = int(sys.argv[3]) if len(sys.argv) > 3 else 90
show_shadow = sys.argv[4].lower() in ('true', '1', 'yes') if len(sys.argv) > 4 else False

# 读取数据 (修改路径)
df = pd.read_excel('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/potential.xlsx')

# 总 epoch 数
total_epochs = len(df.columns) - 1  # 减去 Model 列

# 如果指定了 max_epoch，则截取
if max_epoch is None or max_epoch > total_epochs:
    max_epoch = total_epochs

print(f"📊 总 epoch 数: {total_epochs}, 画到: {max_epoch}, 拟合阶数: {poly_degree}, 置信度: {confidence_level}%, 阴影: {show_shadow}")

# X 轴数据（epoch，从1开始）
epochs = np.arange(1, max_epoch + 1).astype(float)

# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 7))

# 颜色和标记
colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']
markers = ['o', 'o', 'o', 'o']  # 统一用圆形，靠颜色区分

def poly_fit_with_confidence(x, y, degree, confidence=0.90, num_points=200):
    """多项式拟合并计算置信区间"""
    # 拟合多项式
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    
    # 生成平滑的 x 值
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    y_smooth = poly(x_smooth)
    
    # 计算残差
    y_pred = poly(x)
    residuals = y - y_pred
    
    # 计算残差的标准误差
    n = len(x)
    p = degree + 1  # 参数个数
    mse = np.sum(residuals**2) / (n - p)  # 均方误差
    
    # 计算每个预测点的标准误差（使用杠杆值近似）
    # 简化版：使用残差标准差作为置信带宽度
    se = np.sqrt(mse)
    
    # t 分布临界值
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, n - p)
    
    # 置信区间（这里使用简化的方法）
    # 对于多项式拟合，更精确的方法需要计算设计矩阵
    margin = t_crit * se
    
    # 为了让置信带在两端更宽（更真实），可以加一个杠杆效应
    x_normalized = (x_smooth - x.mean()) / x.std()
    leverage_factor = 1 + 0.3 * x_normalized**2  # 简单的杠杆近似
    margin_smooth = margin * leverage_factor
    
    y_upper = y_smooth + margin_smooth
    y_lower = y_smooth - margin_smooth
    
    return x_smooth, y_smooth, y_lower, y_upper

# 存储拟合结果
fit_results = []
for i, model in enumerate(df['Model']):
    y_values = df.iloc[i, 1:max_epoch+1].values.astype(float)
    
    # 过滤掉 NaN 值用于拟合
    valid_mask = ~np.isnan(y_values)
    x_valid = epochs[valid_mask]
    y_valid = y_values[valid_mask]
    
    if len(x_valid) < poly_degree + 1:
        print(f"⚠️ {model} 有效数据点不足，跳过拟合")
        continue
    
    x_smooth, y_smooth, y_lower, y_upper = poly_fit_with_confidence(
        x_valid, y_valid, poly_degree, confidence=confidence_level/100
    )
    # 扩展 x_smooth 到完整范围
    x_smooth = np.linspace(1, max_epoch, 200)
    coeffs = np.polyfit(x_valid, y_valid, poly_degree)
    poly = np.poly1d(coeffs)
    y_smooth = poly(x_smooth)
    
    # 重新计算置信区间
    y_pred = poly(x_valid)
    residuals = y_valid - y_pred
    n = len(x_valid)
    p = poly_degree + 1
    mse = np.sum(residuals**2) / (n - p)
    se = np.sqrt(mse)
    alpha = 1 - confidence_level/100
    t_crit = stats.t.ppf(1 - alpha/2, n - p)
    margin = t_crit * se
    x_normalized = (x_smooth - x_valid.mean()) / x_valid.std()
    leverage_factor = 1 + 0.3 * x_normalized**2
    margin_smooth = margin * leverage_factor
    y_upper = y_smooth + margin_smooth
    y_lower = y_smooth - margin_smooth
    
    fit_results.append((model, y_values, epochs, x_smooth, y_smooth, y_lower, y_upper))

# 重新排序：CARD (我们的方法) 放最后画，确保在最上层
draw_order = [i for i, (m, *_) in enumerate(fit_results) if m.lower() != 'card']
draw_order += [i for i, (m, *_) in enumerate(fit_results) if m.lower() == 'card']

# 第一遍：画阴影（可选）
if show_shadow:
    for i in draw_order:
        model, y_values, x_epochs, x_smooth, y_smooth, y_lower, y_upper = fit_results[i]
        alpha = 0.25 if model.lower() == 'card' else 0.12
        zorder = 5 if model.lower() == 'card' else 1
        ax.fill_between(x_smooth, y_lower, y_upper, color=colors[i], alpha=alpha, zorder=zorder)

# 第二遍：画所有拟合曲线（CARD 加粗）- 不加 label
for i in draw_order:
    model, y_values, x_epochs, x_smooth, y_smooth, y_lower, y_upper = fit_results[i]
    lw = 3.5 if model.lower() == 'card' else 2.5
    ax.plot(x_smooth, y_smooth, color=colors[i], linewidth=lw, alpha=0.95)

# 第三遍：画所有散点（最上层，跳过 NaN，控制密度）- 加 label 用于图例
scatter_every = max(1, max_epoch // 15)  # 控制散点密度
for i in draw_order:
    model, y_values, x_epochs, x_smooth, y_smooth, y_lower, y_upper = fit_results[i]
    valid_mask = ~np.isnan(y_values)
    # 采样
    sample_idx = np.arange(0, len(x_epochs[valid_mask]), scatter_every)
    x_sampled = x_epochs[valid_mask][sample_idx]
    y_sampled = y_values[valid_mask][sample_idx]
    sz = 90 if model.lower() == 'card' else 70
    ax.scatter(x_sampled, y_sampled, color=colors[i], marker=markers[i], 
               s=sz, alpha=0.9, edgecolors=colors[i], linewidths=0.5, zorder=10, label=model.upper())

ax.set_xlabel('Training Epoch', fontsize=34)
ax.set_ylabel('Performance', fontsize=34)
ax.legend(loc='upper left', fontsize=24, handletextpad=0.3)
# 刻度设置：y轴正常显示刻度线，x轴不显示刻度线
ax.tick_params(axis='y', labelsize=24, length=7, direction='in')
ax.tick_params(axis='x', labelsize=24, length=7, direction='in')  # x轴只显示数字，不显示刻度线
# 黑色边框
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.5)
# 关闭网格（只保留 P1 P2 P3 的垂直虚线）
# 只显示 y 轴网格线
ax.grid(False)
ax.yaxis.grid(True, alpha=0.6, linestyle='--', linewidth=1.2, dashes=(12, 5))

# 设置 x 轴刻度
tick_step = max(1, max_epoch // 10)
ax.set_xticks(np.arange(0, max_epoch + 1, tick_step))
ax.set_xlim(0, max_epoch + 1)

# 在特定 epoch 画垂直虚线并标注 P1 P2 P3（标签放在 CARD 曲线上方）
phase_epochs = [11, 23, 27]
phase_labels = ['P1', 'P2', 'P3']
y_min, y_max = ax.get_ylim()
# 固定 y 轴范围，确保虚线从边框开始
ax.set_ylim(y_min, y_max)

# 找到 CARD 的拟合曲线用于获取 y 值
card_result = None
for model, y_values, x_epochs, x_smooth, y_smooth, y_lower, y_upper in fit_results:
    if model.lower() == 'card':
        card_result = (x_smooth, y_smooth)
        break

for ep, label in zip(phase_epochs, phase_labels):
    if ep <= max_epoch:
        # 获取该 epoch 对应的 CARD 曲线 y 值
        if card_result:
            idx = np.argmin(np.abs(card_result[0] - ep))
            y_at_ep = card_result[1][idx]
        else:
            y_at_ep = (y_min + y_max) / 2
        # 从 x 轴（底部边框）到交点的虚线
        ax.vlines(x=ep, ymin=y_min, ymax=y_at_ep, color='gray', linestyle='--', linewidth=2.5, alpha=0.8, clip_on=False)
        # 标签放在该点上方（P2 稍微往下一点）
        offset = 0.01 if label == 'P2' else 0.045
        ax.text(ep, y_at_ep + (y_max - y_min) * offset, label, fontsize=20, fontweight='bold',
                ha='center', va='bottom', color='black')

# ==========================================
# 🟢 新增：导出数据给 LaTeX (pgfplots) 使用
# ==========================================
import os
output_dir = 'Figure/data' # 确保这个文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 1. 导出拟合曲线 (含置信区间)
# 格式: ModelName_x, ModelName_y, ModelName_lower, ModelName_upper
dfs_fit = []
for model, y_values, x_epochs, x_smooth, y_smooth, y_lower, y_upper in fit_results:
    safe_name = model.strip().replace(" ", "_").replace("-", "_") # 避免Latex不喜欢的字符
    dfs_fit.append(pd.DataFrame({
        f'{safe_name}_x': x_smooth,
        f'{safe_name}_y': y_smooth,
        f'{safe_name}_lower': y_lower,
        f'{safe_name}_upper': y_upper
    }))
pd.concat(dfs_fit, axis=1).to_csv(f'{output_dir}/fit_data.csv', index=False)
print(f"拟合数据已保存: {output_dir}/fit_data.csv")

# 2. 导出散点数据 (原始点)
# 格式: ModelName_x, ModelName_y
dfs_scatter = []
for model, y_values, x_epochs, x_smooth, y_smooth, y_lower, y_upper in fit_results:
    safe_name = model.strip().replace(" ", "_").replace("-", "_")
    # 过滤无效值
    mask = ~np.isnan(y_values)
    # 为了对齐长度，这里单独存，或者简单处理：
    # 直接存一个长格式，或者多文件。为了方便Latex读取，我们这里简单存为同一文件，长度不足补NaN
    temp_df = pd.DataFrame({
        f'{safe_name}_x': x_epochs[mask],
        f'{safe_name}_y': y_values[mask]
    })
    dfs_scatter.append(temp_df)
pd.concat(dfs_scatter, axis=1).to_csv(f'{output_dir}/scatter_data.csv', index=False)
print(f"散点数据已保存: {output_dir}/scatter_data.csv")

# 打印一下你的模型名字，方便你填入 Latex
print("\n👇 请将以下名字填入 Latex 的列名中:")
for m in [f[0].strip().replace(" ", "_").replace("-", "_") for f in fit_results]:
    print(m)