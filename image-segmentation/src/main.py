import os
import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset
from model import UNet
from utils import save_model, visualize

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    dataset = ImageDataset(root_dir='path/to/dataset', transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 初始化模型
    model = UNet(in_channels=3, out_channels=1).to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    save_model(model, 'unet_model.pth')

    # 可视化结果
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            outputs = model(images)
            visualize(images, outputs, masks)

if __name__ == "__main__":
    main()