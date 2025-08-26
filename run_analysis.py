import os
import argparse
import pandas as pd
from analyze_regions import analyze_pdf
from visualize_regions import visualize_regions, visualize_color_distribution

def analyze_all_images(data_dir, output_dir, visualize=True):
    """
    分析目录中的所有图像文件（支持PDF和PNG等格式）
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        visualize: 是否生成可视化结果
        
    Returns:
        所有图像文件的分析结果
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有支持的图像文件
    supported_extensions = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    image_files = [f for f in os.listdir(data_dir) if os.path.splitext(f.lower())[1] in supported_extensions]
    
    if not image_files:
        print(f"在 {data_dir} 目录中未找到支持的图像文件")
        return {}
    
    # 存储所有图像文件的分析结果
    all_results = {}
    
    # 分析每个图像文件
    for image_file in image_files:
        image_path = os.path.join(data_dir, image_file)
        image_name = os.path.splitext(image_file)[0]
        print(f"\n分析文件: {image_file}")
        
        try:
            # 分析图像文件
            results = analyze_pdf(image_path)  # 函数名保持不变，但实际上可以处理多种格式
            all_results[image_name] = results
            
            # 输出分析结果
            print(f"闭合连通区域数量: {results['num_regions']}")
            print(f"不同颜色数量: {results['num_colors']}")
            print(f"最大区域像素数: {results['max_region_area']}")
            print(f"最小区域像素数: {results['min_region_area']}")
            print(f"包含同一种颜色的最多区域数量: {results['max_color_regions']}")
            print(f"包含同一种颜色的最少区域数量: {results['min_color_regions']}")
            
            # 生成可视化结果
            if visualize:
                vis_results = visualize_regions(image_path, output_dir)
                visualize_color_distribution(vis_results["region_colors"], output_dir, image_name, vis_results["is_black_list"])
            
        except Exception as e:
            print(f"处理文件 {image_file} 时出错: {str(e)}")
    
    return all_results

def save_results_to_csv(all_results, output_dir):
    """
    将分析结果保存为CSV文件
    
    Args:
        all_results: 所有PDF文件的分析结果
        output_dir: 输出目录
        
    Returns:
        None
    """
    if not all_results:
        print("没有结果可以保存")
        return
    
    # 准备数据
    data = []
    for pdf_name, results in all_results.items():
        data.append({
            "文件名": pdf_name,
            "闭合连通区域数量": results["num_regions"],
            "不同颜色数量": results["num_colors"],
            "最大区域像素数": results["max_region_area"],
            "最小区域像素数": results["min_region_area"],
            "包含同一种颜色的最多区域数量": results["max_color_regions"],
            "包含同一种颜色的最少区域数量": results["min_color_regions"]
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存为CSV文件
    csv_path = os.path.join(output_dir, "analysis_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n分析结果已保存到: {csv_path}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="分析图像文件中的区域（支持PDF和PNG等格式）")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--no_visualize", action="store_true", help="不生成可视化结果")
    args = parser.parse_args()
    
    # 分析所有图像文件
    all_results = analyze_all_images(args.data_dir, args.output_dir, not args.no_visualize)
    
    # 保存结果到CSV文件
    save_results_to_csv(all_results, args.output_dir)
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()