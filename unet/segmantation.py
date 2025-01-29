import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# ================== 配置参数 ==================
CFG = {
    "img_size": 50,  # 保持原始尺寸
    "n_classes": 2,
    "batch_size": 4,
    "input_channels": 3,  # 固定为RGB三通道
    "lr": 1e-4,
    "epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_paths": {
        "train": "train_data",  # 包含训练集npy和png的文件夹
        "val": "val_data",      # 验证集文件夹
        "test": "test_data"     # 测试集文件夹
    },
    "input_size": (3, 50, 50),  # 明确输入尺寸
    "rgb_channels": [0, 1, 2],  # 根据实际观察结果设置
    "scheduler": True,        # 添加学习率调度
    "amp": True              # 启用混合精度训练
}

# ================== 加载数据集类 ==================
class SegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 获取匹配的文件对
        self.samples = []
        for f in os.listdir(data_dir):
            if f.endswith('.npy'):
                base_name = os.path.splitext(f)[0]
                png_file = f"{base_name}.png"
                if os.path.exists(os.path.join(data_dir, png_file)):
                    self.samples.append((f, png_file))
        
        if not self.samples:
            raise FileNotFoundError("未找到匹配的npy和png文件对")

    def __getitem__(self, idx):
        # 加载712通道的numpy图像
        npy_file, png_file = self.samples[idx]
        image = np.load(os.path.join(self.data_dir, npy_file))  # 原始形状 (50, 50, 712)
        
        # 提取前三个通道并转换维度
        image = image[:, :, :3].astype(np.float32)  # 裁剪为 (50, 50, 3)
        image = torch.from_numpy(image).permute(2, 0, 1)  # 转为 (3, 50, 50)
        
        # 应用标准化（假设数据范围0-1）
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                    std=[0.5, 0.5, 0.5])(image)
        
        # 加载对应的PNG标签
        mask = Image.open(os.path.join(self.data_dir, png_file)).convert('L')
        mask = transforms.ToTensor()(mask).squeeze(0).long()
        
        return image, mask

    def show_sample(self, idx=0):
        image, mask = self[idx]
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(image[0].cpu().numpy(), cmap='Reds')  # 红色通道
        plt.subplot(1,3,2)
        plt.imshow(image[1].cpu().numpy(), cmap='Greens')# 绿色通道
        plt.subplot(1,3,3)
        plt.imshow(image[2].cpu().numpy(), cmap='Blues') # 蓝色通道
        plt.show()

# ================== 新增通道选择工具函数 ==================
def find_rgb_channels(data_dir, sample_index=0):
    """
    可视化帮助选择RGB通道的工具函数
    返回建议的通道索引列表
    """
    # 加载示例数据
    sample_npy = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])[sample_index]
    image = np.load(os.path.join(data_dir, sample_npy))
    
    print(f"数据形状：{image.shape} (H, W, C)")
    print("通道统计信息：")
    for c in range(image.shape[-1]):
        print(f"通道 {c}: 均值={image[...,c].mean():.2f}, 方差={image[...,c].std():.2f}")
    
    # 可视化前N个通道
    plt.figure(figsize=(15, 5))
    for i in range(6):  # 显示前6个通道
        plt.subplot(2, 3, i+1)
        plt.imshow(image[...,i], cmap='gray')
        plt.title(f"通道 {i}")
        plt.axis('off')
    plt.show()
    
    # 返回建议通道（示例逻辑，需根据实际情况调整）
    return [0, 1, 2]  # 根据可视化结果修改实际返回值

# ================== U-Net模型 ==================
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()
        
        # 编码器部分
        self.enc1 = self.ConvBlock(in_ch, 64)
        self.enc2 = self.ConvBlock(64, 128)
        self.enc3 = self.ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # 桥接层调整通道数以匹配像素洗牌要求
        self.bridge = self.ConvBlock(256, 1024)  # 1024 = 256*4
        
        # 方案3解码器（像素洗牌+注意力）
        self.dec1 = self.AttentionBlock(256, 256)  # 输入通道256=1024//4
        self.dec2 = self.AttentionBlock(128, 128)
        self.dec3 = self.AttentionBlock(64, 64)
        
        self.out = nn.Sequential(
            nn.Conv2d(64, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.Softmax(dim=1) if out_ch > 1 else nn.Sigmoid()
        )

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            
        def forward(self, x):
            return self.conv(x) + self.skip(x)

    class AttentionBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            # 像素洗牌上采样（输入通道需为out_ch*4）
            self.up = nn.PixelShuffle(upscale_factor=2)
            
            # 空间注意力机制
            self.att = nn.Sequential(
                nn.Conv2d(out_ch*2, out_ch, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_ch, 1, 1),
                nn.Sigmoid()
            )
            
            # 特征转换
            self.conv = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )
            
            # 通道调整
            self.ch_adj = nn.Conv2d(in_ch, out_ch*4, 1) if in_ch != out_ch*4 else nn.Identity()
            
        def forward(self, x, skip):
            # 调整通道数
            x = self.ch_adj(x)
            # 像素洗牌上采样
            x = self.up(x)  # (N, C*4, H, W) -> (N, C, H*2, W*2)
            
            # 空间注意力
            att_map = self.att(torch.cat([x, skip], dim=1))
            x = x * att_map + skip * (1 - att_map)
            
            # 在AttentionBlock中保存注意力图
            self.att_map = att_map.detach().cpu()
            
            return self.conv(x)

    def forward(self, x):
        # 编码阶段
        e1 = self.enc1(x)       # [64, 50,50]
        e2 = self.enc2(self.pool(e1))  # [128,25,25]
        e3 = self.enc3(self.pool(e2))  # [256,12,12]
        
        # 桥接层
        bridge = self.bridge(self.pool(e3))  # [1024,6,6]
        
        # 解码阶段
        d1 = self.dec1(bridge, e3)  # [256,12,12]
        d2 = self.dec2(d1, e2)      # [128,25,25]
        d3 = self.dec3(d2, e1)      # [64,50,50]
        
        return self.out(d3)

# ================== 训练流程 ==================
def train():
    # 修改数据路径配置
    CFG["data_paths"] = {
        "train": "train_data",  # 包含训练集npy和png的文件夹
        "val": "val_data",      # 验证集文件夹
        "test": "test_data"     # 测试集文件夹
    }
    
    # 创建数据集实例
    train_set = SegDataset(CFG["data_paths"]["train"], 
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()
                         ]))
    val_set = SegDataset(CFG["data_paths"]["val"])
    
    train_loader = DataLoader(train_set, CFG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, CFG["batch_size"], shuffle=False)
    
    # 初始化模型
    model = UNet(in_channels=CFG["input_channels"], out_channels=CFG["n_classes"]).to(CFG["device"])
    criterion = nn.CrossEntropyLoss()
    
    # 初始化混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["amp"])
    
    # 添加学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    
    # 修改后的训练循环（添加验证步骤）
    best_val_loss = float('inf')
    for epoch in range(CFG["epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(CFG["device"]), masks.to(CFG["device"])
            
            # 混合精度训练上下文
            with torch.cuda.amp.autocast(enabled=CFG["amp"]):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # 梯度缩放反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 学习率更新
            scheduler.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(CFG["device"]), masks.to(CFG["device"])
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "unet_model.pth")
    print("Training complete! Model saved.")

    # 训练后可视化
    plt.imshow(model.dec1.att_map[0,0].numpy(), cmap='jet')

# ================== 推理与可视化 ==================
def predict(img_path, model_path="unet_model.pth"):
    # 加载模型
    model = UNet(in_channels=CFG["input_channels"], out_channels=CFG["n_classes"]).to(CFG["device"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor()
    ])
    
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(CFG["device"])
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # 可视化
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(pred, cmap='jet')
    plt.title("Prediction")
    plt.axis('off')
    plt.show()

# ================== 执行训练 ==================
if __name__ == "__main__":
    train()
    # 使用示例：
    # predict("your_image.jpg")