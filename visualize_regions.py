import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pdf_handler import pdf_to_image, find_regions, get_region_colors


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
        elif i in region_colors:
            original_color_regions[labels == i] = region_colors[i]
    
    return original_color_regions


def visualize_regions(pdf_path, output_dir):
    """
    可视化PDF文件中的区域
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        
    Returns:
        None
    """
    # 读取图像（支持PDF和PNG）
    image, (_, _, imgw, imgh) = pdf_to_image(pdf_path)
    image_1k, (_, _, imgw, imgh) = pdf_to_image(pdf_path, image_size=1024)
    
    # 查找区域（不再区分边界和内部区域）
    labels, stats, centroids, num_regions, is_black_list = find_regions(image, image_1k)
    
    # 获取每个区域的颜色
    region_colors = get_region_colors(image, labels, num_regions)
    
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
    
    return {
        "image": image,
        "labels": labels,
        "num_regions": num_regions,
        "non_black_regions": non_black_regions,
        "region_colors": region_colors,
        "colored_regions": colored_regions,
        "original_color_regions": original_color_regions,
        "is_black_list": is_black_list
    }

def visualize_color_distribution(region_colors, output_dir, pdf_name, is_black_list=None):
    """
    可视化颜色分布
    
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
    
    for region_id, color in region_colors.items():
        # 如果提供了黑色区域标记，则跳过黑色区域
        if is_black_list is not None and is_black_list[region_id]:
            continue
            
        if color in color_counts:
            color_counts[color] += 1
        else:
            color_counts[color] = 1
    
    # 创建颜色分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 准备数据
    colors = list(color_counts.keys())
    counts = list(color_counts.values())
    
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
    ax.set_title(f'颜色分布 (共 {len(colors)} 种非黑色颜色)')
    ax.set_xlabel('颜色')
    ax.set_ylabel('区域数量')
    ax.set_xticks(range(len(colors)))
    ax.set_xticklabels([f"颜色 {i+1}" for i in range(len(colors))])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, f"{pdf_name}_color_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"颜色分布图已保存到: {output_path}")

def main():
    # 处理data目录中的所有PDF文件
    data_dir = "data\\pdf"
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    files = ["472.pdf"]
    
    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 可视化每个PDF文件
    for f in files:
        print(f"\n 开始处理文件{f}")
        file_path = os.path.join(data_dir, f)
        file_name = f.split('.')[0]
        
        try:
            # 可视化区域
            vis_results = visualize_regions(file_path, output_dir)
            
            # 可视化颜色分布
            # visualize_color_distribution(vis_results["region_colors"], output_dir, file_name, vis_results["is_black_list"])
            
        except Exception as e:
            print(f"处理文件 {f} 时出错: {str(e)}")

if __name__ == "__main__":
    main()