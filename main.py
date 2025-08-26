import os
from pdf_handler import analyze_pdf


if __name__ == "__main__":
    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理data目录中的所有PDF文件
    data_dir = "data\\pdf"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"\n分析文件: {pdf_file}")
        
        try:
            results = analyze_pdf(pdf_path)
            
            # 输出分析结果
            print(f"闭合连通区域数量（不含黑色区域）: {results['num_regions']}")
            print(f"不同颜色数量: {results['num_colors']}")
            print(f"最大区域像素数: {results['max_region_area']}")
            print(f"最小区域像素数: {results['min_region_area']}")
            print(f"包含同一种颜色的最多区域数量: {results['max_color_regions']}")
            print(f"包含同一种颜色的最少区域数量: {results['min_color_regions']}")
            
        except Exception as e:
            print(f"处理文件 {pdf_file} 时出错: {str(e)}")