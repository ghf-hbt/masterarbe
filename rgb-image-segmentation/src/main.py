import os
import yaml
from utils import preprocess_image, visualize_results
from data.dataset import ImageDataset
from models.segmentation_model import SegmentationModel

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = load_config('configs/config.yaml')

    # 加载数据集
    dataset = ImageDataset(config['dataset']['path'])
    
    # 初始化模型
    model = SegmentationModel(config['model'])

    # 训练模型
    model.train(dataset)

    # 进行预测
    results = model.predict(dataset)

    # 可视化结果
    visualize_results(results)

if __name__ == "__main__":
    main()