import pandas as pd
import matplotlib.pyplot as plt

def export_excel_to_scatter(file_path, output_png='scatter_plot.png'):
    # 1. 读取 Excel 文件
    # 假设第一列是 'Model'，将其设置为索引
    df = pd.read_excel(file_path, index_col=0)

    # 2. 预处理数据
    # 将列名（横坐标 948, 1896...）转换为数值类型，防止绘图时坐标轴错乱
    df.columns = pd.to_numeric(df.columns)

    # 3. 创建画布
    plt.figure(figsize=(16, 8))
    
    # 定义一些颜色和标记，方便区分不同的 Model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    # 4. 遍历每一行（每一个 Model）进行绘图
    for i, (model_name, row) in enumerate(df.iterrows()):
        # 去除缺失值 (NaN)，防止绘制空点
        valid_data = row.dropna()
        x_vals = valid_data.index
        y_vals = valid_data.values
        
        plt.scatter(x_vals, y_vals, 
                    label=model_name, 
                    alpha=0.7, 
                    s=20,  # 点的大小
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)])
        
        # 如果你想把点连成线，可以取消下面这行的注释：
        # plt.plot(x_vals, y_vals, alpha=0.3, color=colors[i % len(colors)])

    # 5. 设置图表细节
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Step / Parameter (X)', fontsize=12)
    plt.ylabel('Value (Y)', fontsize=12)
    plt.legend(title='Models', loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 自动调整布局，防止标签被遮挡
    plt.tight_layout()

    # 6. 保存为 PNG
    plt.savefig(output_png, dpi=300)
    print(f"图像已保存至: {output_png}")
    
    # 显示图像
    plt.show()

# 使用示例
# 请确保 your_data.xlsx 的格式与图片一致
export_excel_to_scatter('potential.xlsx')