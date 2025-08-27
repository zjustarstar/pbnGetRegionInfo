import cv2
import os
import numpy as np
import time
import fitz  # PyMuPDF
from PIL import Image
from datetime import datetime
from scipy.spatial.distance import cdist
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import config


def count_regions_below_threshold(region_size, thre_size):
    """
    统计region_size中小于thre_size中每个阈值的区域个数

    参数:
    region_size: 区域面积数组
    thre_size: 阈值数组

    返回:
    results: 包含统计结果的字典
    """
    # 确保输入是numpy数组以便于计算
    region_arr = np.array(region_size)
    thre_arr = np.array(thre_size)

    results = {}

    # 对每个阈值进行统计
    for i, threshold in enumerate(thre_arr):
        # 计算大于当前阈值的区域数量
        count = np.sum(region_arr > threshold)
        results[threshold] = count

        # 打印结果
        print(f"阈值 {threshold}: 有 {count} 个区域的面积大于此值")

    return results


def pdf_to_image(pdf_path, image_size=2048, page_num=0):
    """
    将PDF文件的指定页面转换为图像，并缩放到最长边为2048像素

    Args:
        pdf_path: PDF文件路径
        page_num: 页码（从0开始）

    Returns:
        OpenCV格式的图像对象和页面尺寸信息
    """
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        raise ValueError(f"PDF只有{len(doc)}页，无法访问第{page_num + 1}页")

    # 获取指定页面
    page = doc[page_num]

    # 获取页面尺寸（PDF坐标系）
    pdf_width = page.rect.width
    pdf_height = page.rect.height
    print(f"pdf width={pdf_width}, pdf height={pdf_height}")

    # 计算缩放比例，使图像最长边为固定大小（保持宽高比）
    max_dimension = max(pdf_width, pdf_height)
    scale = image_size / max_dimension

    # 渲染页面为图像
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))

    # 转换为PIL图像
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 转换为OpenCV格式（BGR）
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if doc:
        doc.close()

    print(f'pdf_to_image, width:{pix.width}, height:{pix.height}')

    # 返回图像和页面尺寸信息
    return img_cv, (pdf_width, pdf_height, pix.width, pix.height)


def png_to_image(png_path):
    """
    读取PNG图像文件，并缩放到最长边为2048像素
    支持8位和24位PNG图像

    Args:
        png_path: PNG文件路径

    Returns:
        OpenCV格式的图像对象和图像尺寸信息
    """
    # 读取PNG图像
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    
    # 检查图像是否成功加载
    if img is None:
        raise ValueError(f"无法读取图像文件: {png_path}")
    
    # 获取原始图像尺寸
    orig_height, orig_width = img.shape[:2]
    print(f"原始图像尺寸: 宽={orig_width}, 高={orig_height}")
    
    # 处理不同位深度的PNG图像
    if len(img.shape) == 2:  # 灰度图像
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:  # 带透明通道的图像
        # 创建白色背景
        background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
        # 提取RGB通道
        rgb = img[:, :, :3]
        # 提取Alpha通道并归一化到0-1
        alpha = img[:, :, 3] / 255.0
        # 将Alpha通道扩展为3通道
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        # 混合前景和背景
        img = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
    
    # 计算缩放比例，使图像最长边为2048像素
    max_dimension = max(orig_width, orig_height)
    if max_dimension > 2048:
        scale = 2048 / max_dimension
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"图像已缩放至: 宽={new_width}, 高={new_height}")
    else:
        new_width = orig_width
        new_height = orig_height
    
    # 返回图像和尺寸信息
    return img, (orig_width, orig_height, new_width, new_height)


def load_image(file_path):
    """
    通用图像加载函数，支持PDF和PNG格式
    自动根据文件扩展名选择适当的处理方法
    
    Args:
        file_path: 文件路径
        
    Returns:
        OpenCV格式的图像对象和尺寸信息
    """
    # 获取文件扩展名
    _, ext = os.path.splitext(file_path.lower())
    
    # 根据扩展名选择处理方法
    if ext == '.pdf':
        return pdf_to_image(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        return png_to_image(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def is_black_color(color, threshold=config.BinaryThreshold):
    """
    判断颜色是否为黑色
    
    Args:
        color: BGR颜色值
        threshold: 阈值，小于此值视为黑色
        
    Returns:
        是否为黑色
    """
    # 将BGR颜色转换为灰度值
    gray_value = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    # 灰度值小于阈值时视为黑色
    return gray_value < threshold


def cluster_colors(image1k, image2k_hsv, image2k_non_border_mask):
    """
    对图像进行颜色聚类
    在缩小的图像上进行聚类以提高效率
    
    Args:
        image1k: 输入的图像,用于颜色聚类
        image2k_hsv: 在2k分辨率上判断像素归属
        image2k_no_border_mask: 2k分辨率上的非边界mask
        
    Returns:
        centers: 聚类中心
        temp_labels: 聚类标签图像
        n_colors: 聚类数量
    """
    # 记录开始时间
    start_time = time.time()

    borders, non_border_mask = extract_black_borders(image1k)
    # 减少区域, 防止边界可能得噪音
    kernel = np.ones((3, 3), np.uint8)
    non_border_mask = cv2.morphologyEx(non_border_mask, cv2.MORPH_ERODE, kernel)

    # 转换图像为HSV颜色空间，便于颜色分割
    hsv_img = cv2.cvtColor(image1k, cv2.COLOR_BGR2HSV)
    
    # 获取图像尺寸
    height, width = image1k.shape[:2]
    
    # 获取非边界区域的像素hsv值
    non_border_pixels = hsv_img[non_border_mask > 0]
    
    # 确定聚类数量（根据图像复杂度动态调整）
    n_colors = min(config.ColorNumForCluster, max(5, len(non_border_pixels) // 20000 + 5))  # 根据图像大小动态调整聚类数量
    
    # 将样本转换为double类型
    sample_pixels = non_border_pixels.astype(np.double)
    
    # 使用scikit-learn的KMeans进行聚类
    np.random.seed(42)  # 设置随机种子以获得一致的结果
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=20)
    
    # 执行K-means聚类
    kmeans_start_time = time.time()
    kmeans.fit(sample_pixels)
    print(f"KMeans拟合完成，用时: {time.time() - kmeans_start_time:.2f}秒")
    
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    
    # 将非边界区域的所有像素分配到最近的聚类中心（在原始尺寸图像上）
    # 创建非边界区域的掩码图像（在原始尺寸图像上）
    image2k_non_border_hsv = cv2.bitwise_and(image2k_hsv, image2k_hsv, mask=image2k_non_border_mask)
    
    # 获取非边界区域的像素
    non_border_pixels = image2k_non_border_hsv[image2k_non_border_mask > 0]
    
    # 将非边界区域的所有像素分配到最近的聚类中心
    non_border_labels = kmeans.predict(non_border_pixels.astype(np.double))
    print(f"颜色聚类完成，总用时: {time.time() - start_time:.2f}秒，聚类数量: {n_colors}")
    
    # 创建一个临时的标签图像，用于存储非边界区域的聚类标签
    height, width = image2k_hsv.shape[:2]
    temp_labels = np.zeros((height, width), dtype=np.int32)    # 储存每个像素属于哪一种颜色;
    temp_labels_flat = temp_labels.reshape(-1)
    
    # 获取非边界掩码的索引
    non_border_indices = np.where(image2k_non_border_mask.reshape(-1) > 0)[0]
    
    # 将聚类标签分配给相应的像素
    temp_labels_flat[non_border_indices] = non_border_labels + 1  # +1 是为了避免与背景标签0冲突
    
    return centers, temp_labels, n_colors


def find_regions(image_2k, image_1k):
    """
    查找闭合连通区域，基于颜色进行分割
    首先提取黑色边界，然后只在非边界区域进行颜色聚类和区域识别
    使用scikit-learn的KMeans进行颜色聚类，提高性能
    在缩小的图像上进行聚类以提高效率
    
    Args:
        image_1k用于提取颜色, image_2k用于查找区域;
        
    Returns:
        regions: 标记的区域图像
        stats: 区域统计信息
        centroids: 区域中心点
        num_regions: 区域数量
        is_black: 标记每个区域是否为黑色
    """
    # 记录开始时间
    start_time = time.time()
    
    # 首先提取黑色边界
    borders, non_border_mask = extract_black_borders(image_2k)
    if config.SaveBorderImage:
        cv2.imwrite("output\\borders_2k.png", borders)
    
    # 转换图像为HSV颜色空间，便于颜色分割
    hsv_img = cv2.cvtColor(image_2k, cv2.COLOR_BGR2HSV)
    
    # 获取图像尺寸
    height, width = image_2k.shape[:2]
    
    # 创建最终标签图像和统计信息
    final_labels = np.zeros((height, width), dtype=np.int32)
    stats_list = [np.array([0, 0, 0, 0, 0])]  # 背景统计信息
    centroids_list = [np.array([0, 0])]  # 背景中心点
    is_black_list = [True]  # 黑色区域
    next_label = 0  # 从1开始，0是背景
    
    # 将黑色边界区域标记为特殊标签（使用next_label值）
    border_regions_mask = (borders > 0)
    if np.any(border_regions_mask):
        final_labels[border_regions_mask] = next_label
        next_label += 1

    # 保存每个区域的大小和对应的颜色;下标0是背景信息;
    region_size, region_color = [0], [(0,0,0)]
    # 只处理非边界区域
    if np.any(non_border_mask > 0):
        # 在缩小的图像上进行颜色聚类
        centers, temp_labels, n_colors = cluster_colors(image_1k, hsv_img, non_border_mask)
        
        # 为每个颜色聚类创建掩码并查找连通区域
        for cluster_idx in range(n_colors):
            # 创建当前颜色聚类的掩码
            color_mask = (temp_labels == (cluster_idx + 1)).astype(np.uint8) * 255
            # # 去掉一些非常细的边界
            # kernel = np.ones((3, 3), np.uint8)
            # color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            # 获取当前聚类的BGR颜色
            cluster_hsv = centers[cluster_idx]
            cluster_bgr = cv2.cvtColor(np.uint8([[cluster_hsv]]), cv2.COLOR_HSV2BGR)[0][0]

            # 判断是否为黑色
            is_black = is_black_color(cluster_bgr)
            
            # 使用连通组件分析查找当前颜色内的连通区域
            num_color_labels, color_labels, color_stats, color_centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
            
            # 添加每个连通区域作为新区域
            for i in range(1, num_color_labels):  # 跳过背景
                area = color_stats[i, cv2.CC_STAT_AREA]
                if area > config.MinAreaOfRegion:  # 忽略太小的区域
                    # 将当前连通区域添加到最终标签图像
                    mask = (color_labels == i)
                    final_labels[mask] = next_label

                    # 当前连通区域信息:大小，颜色; next_label刚好就是下标值;
                    region_size.append(area)
                    region_color.append(tuple(cluster_bgr.tolist()))
                    
                    # 添加统计信息
                    stats_list.append(color_stats[i])
                    centroids_list.append(color_centroids[i])
                    is_black_list.append(is_black)
                    
                    next_label += 1

    
    # 区域数量是next_label - 1（减去黑色）
    num_regions = next_label - 1
    
    print(f"区域识别完成，总用时: {time.time() - start_time:.2f}秒，识别到 {num_regions} 个区域")

    return final_labels, region_color, region_size, num_regions, is_black_list

#
# def get_region_info(image, labels, num_regions):
#     """
#     获取每个区域的主要颜色和数量
#
#     Args:
#         image: 原始图像
#         labels: 标记的区域图像
#         num_regions: 区域数量
#
#     Returns:
#         region_colors: 每个区域的主要颜色
#     """
#     region_colors = {}
#     region_size = []
#
#     for i in range(1, num_regions + 1):  # 跳过背景（标签0）
#         # 创建当前区域的掩码
#         region_mask = (labels == i).astype(np.uint8)
#
#         # 应用掩码到原始图像
#         masked_img = cv2.bitwise_and(image, image, mask=region_mask)
#
#         # 获取非零像素（即区域内的像素）
#         non_zero_pixels = masked_img[region_mask > 0]
#
#         if len(non_zero_pixels) > 0:
#             # 计算区域内像素的平均颜色
#             avg_color = np.mean(non_zero_pixels, axis=0).astype(int)
#             # 将颜色转换为元组以便作为字典键
#             color_tuple = tuple(avg_color)
#             region_colors[i] = color_tuple
#             region_size.append(len(non_zero_pixels))
#
#     return region_colors, region_size


def analyze_pdf(image, pdf_path):
    """
    分析PDF文件中的区域和颜色
    首先提取黑色边界，然后只在非边界区域进行颜色聚类和区域识别
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        分析结果的字典
    """
    # 从PDF读取图像, 默认生成的事2k的图, 这里生成一个1k的用于聚类
    image_1k, _ = pdf_to_image(pdf_path, image_size=config.ImgSizeForCluster)
    
    # 查找区域（先识别黑色边界，然后在非边界区域进行处理）
    labels, region_colors, region_size, num_regions, is_black_list = find_regions(image, image_1k)
    
    #统计不同阈值下的色块数量
    count_results = count_regions_below_threshold(region_size, [20, 30, 50])

    # # 统计非黑色区域的数量
    # non_black_regions = sum(1 for i in range(1, num_regions + 1) if not is_black_list[i])

    return labels, num_regions, region_colors, is_black_list


def extract_black_borders(image, threshold=config.BinaryThreshold):
    """
    提取图像中的黑色边界，使用简单的二值化方法
    
    Args:
        image: 输入图像
        threshold: 黑色阈值，小于此值视为黑色
        
    Returns:
        borders: 黑色边界掩码（255表示边界，0表示非边界）
        non_border_mask: 非边界区域掩码（255表示非边界区域，0表示边界）
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用cv2.threshold进行二值化，小于阈值的像素被视为黑色
    _, borders = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 使用形态学操作改善黑色区域的连通性
    # kernel = np.ones((3, 3), np.uint8)
    # borders = cv2.morphologyEx(borders, cv2.MORPH_CLOSE, kernel)
    
    # 创建非边界区域掩码（边界的反转）
    non_border_mask = cv2.bitwise_not(borders)
    print(f"黑色边界提取完成")
    
    return borders, non_border_mask
