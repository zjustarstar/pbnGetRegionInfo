import os
import time
from pdf_handler import analyze_pdf, pdf_to_image
import visualize_regions as visreg
import config


if __name__ == "__main__":
    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    # 处理data目录中的所有PDF文件
    data_dir = "data\\pdf"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        file_name = pdf_file.split('.')[0]
        print(f"\n分析文件: {pdf_file}")

        try:
            image, _ = pdf_to_image(pdf_path)

            # labels=全图每个像素的region标签; region_colors表示每个region的颜色;
            labels, num_regions, region_colors, is_black_list = analyze_pdf(image, pdf_path)

            # 可视化
            if config.SaveVisualResult:
                visreg.visualize_regions(image, labels, num_regions, region_colors, is_black_list, pdf_path, output_dir)
                visreg.visualize_color_distribution(region_colors, output_dir, file_name, is_black_list)

            # 输出分析结果
            #
            # # 返回分析结果
            # return {
            #     "num_regions": non_black_count,  # 只计算非黑色区域
            #     "num_colors": len(color_counts),
            #     "max_region_area": max_region[1],
            #     "min_region_area": min_region[1],
            #     "max_color_regions": max_color_regions[1],
            #     "min_color_regions": min_color_regions[1],
            #     "labels": labels,  # 添加标签信息供可视化使用
            #     "is_black_list": is_black_list,  # 添加黑色区域标记供可视化使用
            #     "black_regions": black_regions,  # 添加黑色区域列表
            #     "non_black_regions": non_black_regions  # 添加非黑色区域列表
            # }

            # print(f"闭合连通区域数量（不含黑色区域）: {results['num_regions']}")
            # print(f"不同颜色数量: {results['num_colors']}")
            # print(f"包含同一种颜色的最多区域数量: {results['max_color_regions']}")
            # print(f"包含同一种颜色的最少区域数量: {results['min_color_regions']}")
            
        except Exception as e:
            print(f"处理文件 {pdf_file} 时出错: {str(e)}")

  # 计算运行时长（秒）
    file_size = len(pdf_files)
    duration = (time.time() - start_time) // file_size

    # 转换为更友好的格式（可选）
    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)

    # 输出结果
    print(f"每张图运行时长: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")