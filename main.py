import os
import time
from pdf_handler import analyze_pdf, pdf_to_image
import visualize_regions as visreg
import config


# 输入单张pdf图像，返回色块信息;
def get_blockinfo_from_image(pdf_path):
    image, _ = pdf_to_image(pdf_path)

    # labels=全图每个像素的region标签; region_colors表示每个region的颜色;
    labels, num_regions, region_colors, is_black_list, final_num_regions = analyze_pdf(image, pdf_path)
    min_colors, max_colors = visreg.visualize_color_distribution(region_colors, output_dir, file_name, is_black_list)

    # 可视化
    if config.SaveVisualResult:
        visreg.visualize_regions(image, labels, num_regions, region_colors, is_black_list, pdf_path, output_dir)

    # 返回最终色块数, 最少的10种颜色BGR,最多的10种颜色BGR
    return final_num_regions, min_colors, max_colors


if __name__ == "__main__":
    # 如果需要保存一些数据用于debug，需要创建输出目录
    if config.SaveColorDistribution or config.SaveVisualResult:
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    start_time = time.time()
    # 处理data目录中的所有PDF文件
    data_dir = "data"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        file_name = pdf_file.split('.')[0]
        print(f"\n分析文件: {pdf_file}")

        try:
            # 单个pdf文件的色块信息提取
            final_num_regions, min_colors, max_colors = get_blockinfo_from_image(pdf_path)

            # 输出返回结果
            print(f"最终色块数量（不含黑色区域）: {final_num_regions}")
            print("色块数量最少的10个颜色:")
            for i, (color, count) in enumerate(min_colors):
                print(f"  颜色 {i + 1}: BGR={color}, 区域数量={count}")

            print("色块数量最多的10个颜色:")
            for i, (color, count) in enumerate(max_colors):
                print(f"  颜色 {i + 1}: BGR={color}, 区域数量={count}")
            
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