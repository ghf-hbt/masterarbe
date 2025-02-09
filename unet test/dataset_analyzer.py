import os
import numpy as np
from PIL import Image
from collections import defaultdict6
import matplotlib.pyplot as plt

def analyze_dataset(label_folder, tolerance=2, background_ranges=None):
    """
    分析标签数据集统计信息
    参数:
        label_folder: 包含标签PNG文件的文件夹路径
        tolerance: 自动发现类别间隔的容差参数
        background_ranges: 背景范围列表
    返回:
        class_info: 包含类别统计信息的字典
        histogram: 灰度值分布直方图数据
    """
    # 在函数开头添加背景处理参数
    background_ranges = background_ranges or [(0, 0)]  # 默认认为0是背景
    bg_values = set()
    for start, end in background_ranges:
        bg_values.update(range(start, end+1))
    
    # 初始化统计存储
    value_counts = defaultdict(int)
    total_pixels = 0
    
    # 遍历所有标签文件
    for filename in os.listdir(label_folder):
        if not filename.endswith('.png'):
            continue
            
        filepath = os.path.join(label_folder, filename)
        try:
            # 直接提取红色通道（假设三通道值相同）
            img = Image.open(filepath)
            red_channel = img.getchannel(0)  # 获取红色通道（索引0）
            pixels = np.array(red_channel).flatten()  # 直接转换为numpy数组
            
            # 统计像素值
            unique, counts = np.unique(pixels, return_counts=True)
            for val, cnt in zip(unique, counts):
                value_counts[int(val)] += cnt
                total_pixels += cnt
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue
    
    # 分析类别信息
    sorted_values = sorted(value_counts.keys())
    class_info = {}
    current_class = 0
    
    # 自动发现类别间隔
    prev = sorted_values[0]
    class_start = prev
    
    for val in sorted_values[1:]:
        if val - prev > tolerance:
            # 记录当前类别范围
            class_info[current_class] = {
                'range': (class_start, prev),
                'count': sum(value_counts[v] for v in range(class_start, prev+1)),
                'values': [v for v in range(class_start, prev+1)]
            }
            current_class += 1
            class_start = val
        prev = val
    
    # 添加最后一个类别
    class_info[current_class] = {
        'range': (class_start, sorted_values[-1]),
        'count': sum(value_counts[v] for v in range(class_start, sorted_values[-1]+1)),
        'values': [v for v in range(class_start, sorted_values[-1]+1)]
    }
    
    # 生成直方图数据
    histogram = {
        'bins': np.arange(0, 256),
        'counts': [value_counts.get(i, 0) for i in range(256)]
    }
    
    # 在返回前添加背景标识
    for cid in list(class_info.keys()):
        values = class_info[cid]['values']
        if any(v in bg_values for v in values):
            class_info[cid]['is_background'] = True
            # 将背景类移到最前面
            class_info = {cid: class_info.pop(cid), **class_info}
        else:
            class_info[cid]['is_background'] = False
    
    return class_info, histogram

def visualize_analysis(class_info, histogram):
    """可视化分析结果"""
    plt.figure(figsize=(15, 6))
    
    # 直方图
    plt.subplot(1, 2, 1)
    plt.bar(histogram['bins'], histogram['counts'], width=1.0)
    plt.title('灰度值分布直方图')
    plt.xlabel('灰度值')
    plt.ylabel('出现次数')
    
    # 类别信息
    plt.subplot(1, 2, 2)
    class_ids = list(class_info.keys())
    percentages = [info['count']/sum(info['count'] for info in class_info.values())*100 
                   for info in class_info.values()]
    
    # 修改类别分布图的颜色
    colors = ['#FF6B6B' if info['is_background'] else '#4ECDC4' 
             for info in class_info.values()]
    
    plt.barh(class_ids, percentages, color=colors)
    plt.title('类别分布比例（红色为背景）')
    plt.xlabel('占比 (%)')
    plt.ylabel('类别 ID')
    plt.yticks(class_ids, [f"Class {cid}\n({info['range'][0]}-{info['range'][1]})" 
                          for cid, info in class_info.items()])
    
    plt.tight_layout()
    plt.show()

def detect_label_type(class_info):
    """
    根据分析结果判断标签类型
    返回类型：
        'index' - 直接使用像素值作为类别索引
        'range' - 使用灰度值范围表示类别
    """
    # 检查是否所有类别都是单个值
    single_value = all(info['range'][0] == info['range'][1] for info in class_info.values())
    
    # 检查值是否连续
    values = sorted([v for info in class_info.values() for v in info['values']])
    is_continuous = all(values[i+1] - values[i] == 1 for i in range(len(values)-1))
    
    if single_value and is_continuous and max(values) < 256:
        return 'index'
    else:
        return 'range'

def create_class_mapping(class_info, label_type='auto'):
    """
    创建类别映射表
    参数:
        label_type: 
            'auto' - 自动检测类型
            'index' - 直接索引模式
            'range' - 范围模式
    """
    if label_type == 'auto':
        label_type = detect_label_type(class_info)
    
    mapping = {}
    if label_type == 'index':
        # 直接使用灰度值作为类别ID
        for cid, info in class_info.items():
            for val in info['values']:
                mapping[val] = cid
    else:
        # 范围模式保留现有逻辑
        for cid, info in class_info.items():
            mapping[cid] = info['range']
    
    return mapping, label_type

if __name__ == "__main__":
    # 使用示例
    label_folder = "C:\\Users\\sm1508\\Desktop\\dataset\\train"  # 修改为实际路径
    
    # 指定背景范围（例如0和255都是背景）
    bg_ranges = [(0,0), (255,255)]
    
    print("正在分析数据集...")
    class_info, histogram = analyze_dataset(
        label_folder, 
        tolerance=2,
        background_ranges=bg_ranges
    )
    
    print("\n统计结果：")
    print(f"总类别数：{len(class_info)}")
    print(f"总像素数：{sum(histogram['counts']):,}")
    print("\n详细类别信息：")
    
    for cid, info in class_info.items():
        print(f"类别 {cid}:")
        print(f"  灰度范围：{info['range'][0]} - {info['range'][1]}")
        print(f"  包含像素数：{info['count']:,} ({info['count']/sum(histogram['counts'])*100:.2f}%)")
        print(f"  实际使用灰度值：{info['values']}\n")
    
    # 可视化结果
    visualize_analysis(class_info, histogram)

    print("\n类别映射关系：")
    mapping, label_type = create_class_mapping(class_info)
    print(f"检测到标签类型：{label_type}")
    for val, cid in sorted(mapping.items()):
        print(f"灰度值 {val} → 类别 {cid}")