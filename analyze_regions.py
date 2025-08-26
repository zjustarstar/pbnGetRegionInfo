import os
import numpy as np
import cv2
from main import load_image, find_regions, get_region_colors

def analyze_pdf(pdf_path):
    """
    分析PDF文件中的区域
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        分析结果字典
    """
    # 读取图像（支持PDF和PNG）
    image, _ = load_image(pdf_path)
    
    # 查找区域（不再区分边界和内部区域）
    labels, stats, centroids, num_regions, is_black_list = find_regions(image)
    
    # 获取每个区域的颜色
    region_colors = get_region_colors(image, labels, num_regions)
    
    # 统计非黑色区域的数量
    non_black_regions = sum(1 for i in range(1, num_regions + 1) if not is_black_list[i])
    
    # 计算每个区域的面积（像素数）
    region_areas = {}
    for i in range(1, num_regions + 1):
        # 只统计非黑色区域
        if not is_black_list[i]:
            region_areas[i] = stats[i, cv2.CC_STAT_AREA]
    
    # 统计每种颜色的区域数量（排除黑色区域）
    color_regions = {}
    for region_id, color in region_colors.items():
        # 跳过黑色区域
        if is_black_list[region_id]:
            continue
            
        color_key = tuple(color)  # 将颜色转换为元组以便用作字典键
        if color_key in color_regions:
            color_regions[color_key] += 1
        else:
            color_regions[color_key] = 1
    
    # 计算统计数据
    if region_areas:
        max_region_area = max(region_areas.values())
        min_region_area = min(region_areas.values())
    else:
        max_region_area = 0
        min_region_area = 0
    
    if color_regions:
        max_color_regions = max(color_regions.values())
        min_color_regions = min(color_regions.values())
        num_colors = len(color_regions)
    else:
        max_color_regions = 0
        min_color_regions = 0
        num_colors = 0
    
    # 返回分析结果
    return {
        "num_regions": non_black_regions,  # 只计算非黑色区域
        "num_colors": num_colors,
        "max_region_area": max_region_area,
        "min_region_area": min_region_area,
        "max_color_regions": max_color_regions,
        "min_color_regions": min_color_regions,
        "region_areas": region_areas,
        "color_regions": color_regions
    }

def main():
    # 处理data目录中的所有PDF文件
    data_dir = "data"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    # 分析每个PDF文件
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"\n分析文件: {pdf_file}")
        
        try:
            # 分析PDF文件
            results = analyze_pdf(pdf_path)
            
            # 输出分析结果
            print(f"闭合连通区域数量: {results['num_regions']}")
            print(f"不同颜色数量: {results['num_colors']}")
            print(f"最大区域像素数: {results['max_region_area']}")
            print(f"最小区域像素数: {results['min_region_area']}")
            print(f"包含同一种颜色的最多区域数量: {results['max_color_regions']}")
            print(f"包含同一种颜色的最少区域数量: {results['min_color_regions']}")
            
        except Exception as e:
            print(f"处理文件 {pdf_file} 时出错: {str(e)}")

if __name__ == "__main__":
    main()