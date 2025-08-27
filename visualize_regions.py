import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.colors import ListedColormap
from pdf_handler import pdf_to_image, find_regions, count_regions_below_threshold


def create_colored_regions_image(image, labels, num_regions, is_black_list=None):
    """
    创建彩色区域图像，每个区域使用不同的随机颜色
    
    Args:
        image: 原始图像
        labels: 标记的区域图像
        num_regions: 区域数量
        is_black_list: 标记每个区域是否为黑色的列表
        
    Returns:
        colored_regions: 彩色区域图像
    """
    # 创建随机颜色映射（跳过黑色作为背景）
    np.random.seed(42)  # 设置随机种子以获得一致的颜色
    colors = np.random.randint(50, 256, size=(num_regions + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # 背景为黑色
    
    # 创建彩色区域图像
    colored_regions = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # 如果提供了黑色区域标记，则使用特殊颜色标记黑色区域
    if is_black_list is not None:
        for i in range(1, num_regions + 1):
            if is_black_list[i]:
                # 黑色区域使用深灰色表示
                colored_regions[labels == i] = [50, 50, 50]
            else:
                colored_regions[labels == i] = colors[i]
    else:
        # 没有提供黑色区域标记，使用常规着色
        for i in range(num_regions + 1):
            colored_regions[labels == i] = colors[i]
    
    return colored_regions

def create_original_color_regions(image, labels, region_colors, num_regions, is_black_list=None):
    """
    创建使用原始颜色的区域图像
    
    Args:
        image: 原始图像
        labels: 标记的区域图像
        region_colors: 每个区域的颜色
        num_regions: 区域数量
        is_black_list: 标记每个区域是否为黑色的列表
        
    Returns:
        original_color_regions: 使用原始颜色的区域图像
    """
    # 创建使用原始颜色的区域图像
    original_color_regions = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    original_color_regions[:] = [0, 0, 0]  # 背景为白色
    
    for i in range(1, num_regions + 1):
        # 如果提供了黑色区域标记且当前区域是黑色，则使用深灰色表示
        if is_black_list is not None and is_black_list[i]:
            original_color_regions[labels == i] = [50, 50, 50]
        else:
            original_color_regions[labels == i] = region_colors[i]
    
    return original_color_regions


def visualize_regions(image, labels, num_regions, region_colors, is_black_list, pdf_path, output_dir):
    """
    可视化PDF文件中的区域
    
    Args:
        image: pdf文件对应的图片
        labels: 每个像素的region标签
        num_regions: 一共多少个色块
        region_colors: 每个色块的颜色信息
        is_black_list: 是否是黑色背景区域
        pdf_path: PDF文件路径
        output_dir: 输出目录
        
    Returns:
        None
    """
    # 统计非黑色区域的数量
    non_black_regions = sum(1 for i in range(1, num_regions + 1) if not is_black_list[i])
    
    # 创建彩色区域图像
    colored_regions = create_colored_regions_image(image, labels, num_regions, is_black_list)
    
    # 创建使用原始颜色的区域图像
    original_color_regions = create_original_color_regions(image, labels, region_colors, num_regions, is_black_list)
    # 保存使用原始颜色的图像
    # 获取原始文件名和扩展名
    base_name = os.path.basename(pdf_path)
    name, ext = os.path.splitext(base_name)
    new_file_name = f"{name}_oriColor.png"
    # 构建完整的输出路径
    output_path = os.path.join(output_dir, new_file_name)
    cv2.imwrite(output_path,original_color_regions)
    # 原始图像的输出;
    output_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(output_path, image)
    
    # 创建可视化图像
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 所有区域图像（包括黑色区域）
    axes[0, 1].imshow(cv2.cvtColor(colored_regions, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'所有区域 (共 {num_regions} 个区域，其中非黑色区域 {non_black_regions} 个)')
    axes[0, 1].axis('off')
    
    # 非黑色区域图像
    non_black_colored = colored_regions.copy()
    for i in range(1, num_regions + 1):
        if is_black_list[i]:
            # 将黑色区域设为白色（背景色）
            non_black_colored[labels == i] = [0, 0, 0]
    
    axes[1, 0].imshow(cv2.cvtColor(non_black_colored, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'非黑色区域 (共 {non_black_regions} 个区域)')
    axes[1, 0].axis('off')
    
    # 原始颜色区域图像
    axes[1, 1].imshow(cv2.cvtColor(original_color_regions, cv2.COLOR_BGR2RGB))
    
    # 计算非黑色区域的不同颜色数量
    non_black_colors = set()
    for i in range(1, num_regions + 1):
        if not is_black_list[i] and i in region_colors:
            non_black_colors.add(region_colors[i])
    
    axes[1, 1].set_title(f'原始颜色区域 (共 {len(non_black_colors)} 种非黑色颜色)')
    axes[1, 1].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    pdf_name = os.path.basename(pdf_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{pdf_name}_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到: {output_path}")


def visualize_color_distribution(region_colors, output_dir, pdf_name, is_black_list=None):
    """
    可视化颜色分布，只显示区域数量最多的10个和最少的10个颜色
    
    Args:
        region_colors: 每个区域的颜色
        output_dir: 输出目录
        pdf_name: PDF文件名
        is_black_list: 标记每个区域是否为黑色的列表
        
    Returns:
        None
    """
    # 统计每种颜色的区域数量（排除黑色区域）
    color_counts = {}
    
    for region_id, color in enumerate(region_colors):
        # 如果提供了黑色区域标记，则跳过黑色区域
        if is_black_list is not None and is_black_list[region_id]:
            continue
            
        if color in color_counts:
            color_counts[color] += 1
        else:
            color_counts[color] = 1
    
    # 按区域数量排序
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
    
    # 获取区域数量最少的10个和最多的10个颜色
    min_colors = sorted_colors[:min(10, len(sorted_colors)//2)]
    max_colors = sorted_colors[-min(10, len(sorted_colors)//2):]
    
    # 合并最少和最多的颜色
    selected_colors = min_colors + max_colors
    
    # 创建颜色分布图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 准备数据
    colors = [item[0] for item in selected_colors]
    counts = [item[1] for item in selected_colors]
    labels = [f"最少{i+1}" for i in range(len(min_colors))] + [f"最多{i+1}" for i in range(len(max_colors))]
    
    # 将BGR颜色转换为RGB用于显示
    rgb_colors = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
    
    # 创建条形图
    bars = ax.bar(range(len(colors)), counts, color=rgb_colors)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(count), ha='center', va='bottom')
    
    # 设置标题和标签
    ax.set_title(f'颜色分布 (显示区域数量最多和最少的各10个颜色，共 {len(color_counts)} 种非黑色颜色)')
    ax.set_xlabel('颜色')
    ax.set_ylabel('区域数量')
    ax.set_xticks(range(len(colors)))
    ax.set_xticklabels(labels)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f"{pdf_name}_color_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印区域数量最多和最少的10个颜色
    print(f"颜色分布 (共 {len(color_counts)} 种非黑色颜色)")
    print("区域数量最少的10个颜色:")
    for i, (color, count) in enumerate(min_colors):
        print(f"  颜色 {i+1}: BGR={color}, 区域数量={count}")
    
    print("区域数量最多的10个颜色:")
    for i, (color, count) in enumerate(max_colors):
        print(f"  颜色 {i+1}: BGR={color}, 区域数量={count}")
    
    print(f"颜色分布图已保存到: {output_path}")

