# pbnGetRegionInfo
用于color产品。基于模型生成的图，统计基础信息包括色块数量等

## 输入输出
- 输入：pdf文件
- 输出：该pdf文件中的色块数量，以及拥有最多色块和最少色块的10种颜色

## 示例
config.py文件用于配置，请勿改动，保持默认值即可。
参见main.py中的get_blockinfo_from_image函数。正常情况下，处理data\\test.pdf返回的色块数量是1775。




