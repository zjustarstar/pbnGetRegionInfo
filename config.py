# algorithm parameter
ImgSizeForCluster = 1200  # 用于聚类的图像大小;一般是缩放到更小，再聚类的：
BinaryThreshold = 30      # 二值化阈值
MinAreaOfRegion = 20      # 最小区域大小
ColorNumForCluster = 100   # 聚类时的颜色数量。一般50-200种颜色;

# debug parameter
SaveBorderImage = False        # 保存黑色边界图;仅保存最新的一张;
SaveVisualResult = False       # 保存色块可视化结果
SaveColorDistribution = False  # 保存颜色分布可视化结果

# blocks parameter
NumThre1 = 2000    # 第一级阈值;

