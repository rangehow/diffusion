import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

# 启用手绘/漫画风格上下文
# 注意：如果你的系统没有 Comic Sans MS 等字体，Matplotlib 会回退到默认字体，但线条依然会弯曲
plt.xkcd(scale=1, length=100, randomness=2)

def draw_comic_style_matrix(matrix_type, filename):
    """
    绘制类似 Excalidraw/漫画风格的注意力矩阵
    """
    # 设置画布
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 隐藏默认坐标轴边框，我们将自己画
    ax.set_axis_off()
    
    # 网格尺寸 (统一为 6x6)
    N = 6
    
    # 定义间距和盒子大小
    box_size = 0.85
    spacing = 1.0
    
    # --- 颜色定义 (参考第二张图的配色) ---
    # 蓝色 (Input style)
    color_blue = '#DAE8FC' 
    # 灰色 (Masked/Ignored)
    color_grey = '#F5F5F5'
    # 混合掩码用的额外颜色 (保持 pastel 色调)
    color_orange = '#FFE6CC' # 对应左上 Block Diagonal
    color_yellow = '#FFF2CC' # 对应右下 Block Causal
    
    # 标签定义
    if matrix_type == 'hybrid':
        # 使用简单的文本代替复杂的 Latex，保证漫画风格的一致性
        labels = ['t1', 't2', 't3', 'p1', 'p2', 'p3'] # t=time, p=prompt
    else:
        labels = ['1', '2', '3', '4', '5', '6']

    # --- 绘制网格循环 ---
    for i in range(N): # Row (从上到下)
        for j in range(N): # Col (从左到右)
            
            # 计算绘制位置 (Matplotlib y轴向上，所以要做反转处理)
            x_pos = j * spacing
            y_pos = (N - 1 - i) * spacing
            
            # --- 确定颜色逻辑 ---
            fill_color = color_grey # 默认背景色
            is_active = False
            
            if matrix_type == 'causal':
                # 下三角为蓝色
                if j <= i:
                    fill_color = color_blue
                    is_active = True
                    
            elif matrix_type == 'full':
                # 全部为蓝色
                fill_color = color_blue
                is_active = True
                
            elif matrix_type == 'hybrid':
                # 复杂的混合逻辑 (6x6 Block Level)
                
                # 1. 左上 (0-2, 0-2): 对角线 (Orange)
                if i < 3 and j < 3:
                    if i == j:
                        fill_color = color_orange
                        is_active = True
                    else:
                        fill_color = color_grey # 背景
                
                # 2. 右上 (0-2, 3-5): 偏移因果 (Blue)
                elif i < 3 and j >= 3:
                    # Row 0 (idx 0) sees nothing
                    # Row 1 (idx 1) sees Col 3 (idx 3) -> j < i + 3 ? No.
                    # 逻辑: Row i 关注 Col j (where j-3 < i)
                    # i=1 (t2) sees j=3 (p1) -> 3-3 < 1 -> 0 < 1 (True)
                    target_col_idx = j - 3
                    if target_col_idx < i: 
                        fill_color = color_blue
                        is_active = True
                    else:
                        fill_color = color_grey
                
                # 3. 左下 (3-5, 0-2): 空 (Grey)
                elif i >= 3 and j < 3:
                    fill_color = color_grey
                    
                # 4. 右下 (3-5, 3-5): 因果 (Yellow)
                elif i >= 3 and j >= 3:
                    if j <= i:
                        fill_color = color_yellow
                        is_active = True

            # --- 绘制圆角矩形 (FancyBboxPatch) ---
            # 这种 BoxStyle="Round" 最接近你给的图的风格
            rect = patches.FancyBboxPatch(
                (x_pos, y_pos), 
                box_size, box_size,
                boxstyle="Round,pad=0.02,rounding_size=0.1",
                facecolor=fill_color,
                edgecolor='black',
                linewidth=1.5, # 粗边框
                mutation_scale=1 # 保持形状比例
            )
            ax.add_patch(rect)
            
            # (可选) 如果你想在格子里填数字，可以解开下面这行
            # if is_active:
            #     ax.text(x_pos + box_size/2, y_pos + box_size/2, "1", 
            #             ha='center', va='center', fontsize=12, fontname='Comic Sans MS')

    # --- 添加手绘风格的标签 ---
    # 调整标签位置使其居中对齐方块
    center_offset = box_size / 2
    
    # X轴标签 (底部)
    for idx, label in enumerate(labels):
        ax.text(idx * spacing + center_offset, -0.6, label, 
                ha='center', va='center', fontsize=20, fontname='Comic Sans MS')
        
    # Y轴标签 (左侧)
    for idx, label in enumerate(labels):
        ax.text(-0.6, (N - 1 - idx) * spacing + center_offset, label, 
                ha='center', va='center', fontsize=20, fontname='Comic Sans MS')

    # 添加轴标题 (类似 Input / Label 的指示)
    ax.text(-1.2, N/2 * spacing, "Target Sequence", ha='center', va='center', rotation=90, fontsize=16)
    ax.text(N/2 * spacing, -1.2, "Source Sequence", ha='center', va='center', fontsize=16)

    # 调整显示范围
    ax.set_xlim(-1.5, N * spacing + 0.5)
    ax.set_ylim(-1.5, N * spacing + 0.5)
    
    # 保存
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight') # 降低dpi一点点增加手绘感
    print(f"Saved: {filename}")
    plt.close()

# --- 生成三张图 ---

# 1. Causal Mask (6x6)
draw_comic_style_matrix('causal', 'comic_mask_causal.png')

# 2. Full Mask (6x6)
draw_comic_style_matrix('full', 'comic_mask_full.png')

# 3. Hybrid Mask (6x6 blocks)
draw_comic_style_matrix('hybrid', 'comic_mask_hybrid.png')