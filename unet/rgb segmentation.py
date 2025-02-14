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
    "n_classes": 13,  # 包含12类物体+背景
    "class_interval": 20,  # 每个类别间隔20个灰度值
    "batch_size": 4,
    "input_channels": 3,  # 固定为RGB三通道
    "lr": 1e-4,
    "epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_paths": {
        "train": r"F:\BaiduNetdiskDownload\dataset (2)\dataset\dataset\train",
        "val": r"F:\BaiduNetdiskDownload\dataset (2)\dataset\dataset\val",
        "test": r"F:\BaiduNetdiskDownload\dataset (2)\dataset\dataset\test"
    },
    "scheduler": True,        # 添加学习率调度
    "amp": True,              # 启用混合精度训练
    "resume_training": True,  # 新增配置项：是否从上次训练继续
}
# ================== 加载数据集类 ==================
class SegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 获取并配对.npy和.png文件
        self.samples = []
        for f in os.listdir(data_dir):
            if f.endswith('.npy'):
                prefix = f.rsplit('.', 1)[0]
                png_file = f"{prefix}.png"
                if os.path.exists(os.path.join(data_dir, png_file)):
                    self.samples.append((f, png_file))
                else:
                    raise FileNotFoundError(f"找不到对应的PNG标签文件: {png_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_file, png_file = self.samples[idx]
        
        # 加载图像数据
        npy_path = os.path.join(self.data_dir, npy_file)
        image = np.load(npy_path)[:, :, :3].astype(np.float32)  # 取前3个通道
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        
        # 加载标签数据
        png_path = os.path.join(self.data_dir, png_file)
        mask = Image.open(png_path).convert('RGB')
        mask = transforms.ToTensor()(mask)  # (3, H, W)
        
        # 转换标签为单通道类别索引
        mask = torch.round(mask * 255).long()  # 转换为0-255整数
        mask = mask[0]  # 取第一个通道（假设三通道值相同）
        mask = mask // 20  # 将0-255映射到0-12（13个类别）
        mask = torch.clamp(mask, 0, 12)  # 确保不超过12

        # 数据增强（保持同步）
        if self.transform:
            seed = torch.randint(0, 2**32, size=(1,)).item()
            
            torch.manual_seed(seed)
            image = self.transform(image)
            
            torch.manual_seed(seed)
            mask = self.transform(mask.float()).long()

        return image, mask
# ================== 标准U-Net模型 ==================
class UNet(nn.Module):
    def __init__(self, in_channels=CFG["input_channels"], out_channels=CFG["n_classes"]):
        super().__init__()
        
        # 基础卷积块
        class ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            def forward(self, x):
                return self.conv(x)
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # 桥接层
        self.bridge = ConvBlock(256, 512)
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.LogSoftmax(dim=1)  # 适用于NLLLoss
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)        # [64, 50,50]
        e2 = self.enc2(self.pool(e1))  # [128,25,25]
        e3 = self.enc3(self.pool(e2))  # [256,12,12]
        
        # 桥接
        bridge = self.bridge(self.pool(e3))  # [512,6,6]
        
        # 解码（添加尺寸对齐）
        d1 = self.up1(bridge)
        # 计算尺寸差异并填充
        diffY = e3.size()[2] - d1.size()[2]
        diffX = e3.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        diffY = e2.size()[2] - d2.size()[2]
        diffX = e2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        diffY = e1.size()[2] - d3.size()[2]
        diffX = e1.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)
        
        return self.out(d3)

# ================== 训练流程 ==================
def train():
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
    
    # 加载已有模型
    if CFG["resume_training"] and os.path.exists("best_unet_model.pth"):
        model.load_state_dict(torch.load("best_unet_model.pth", map_location=CFG["device"]))
        print("已加载预训练模型，继续训练...")
    
    criterion = nn.NLLLoss()
    
    # 混合精度训练初始化
    scaler = torch.amp.GradScaler(enabled=CFG["amp"])
    
    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CFG["lr"],          # 初始学习率 1e-4
        weight_decay=1e-4      # L2正则化强度
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(CFG["epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(CFG["device"]), masks.to(CFG["device"])
            optimizer.zero_grad()
            
            # 参数device_type
            with torch.amp.autocast(device_type=CFG["device"].split(':')[0], enabled=CFG["amp"]):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # 梯度缩放反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # 每个epoch结束后更新学习率
        scheduler.step()
        
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
            print(f"Epoch {epoch+1} | 发现新的最优模型，已保存！")
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "unet_model.pth")
    print("Training complete! Model saved.")

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

# ================== 测试集评估函数 ==================
def evaluate_testset(model_path="unet_model.pth"):
    # 初始化模型
    model = UNet(in_channels=CFG["input_channels"], 
               out_channels=CFG["n_classes"]).to(CFG["device"])
    model.load_state_dict(torch.load(model_path, map_location=CFG["device"], weights_only=True))
    model.eval()
    
    # 创建测试集
    test_dataset = SegDataset(CFG["data_paths"]["test"])
    test_loader = DataLoader(test_dataset, 
                           batch_size=CFG["batch_size"], 
                           shuffle=False)
    
    # 初始化评估指标
    total_correct = 0
    total_pixels = 0
    iou_scores = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(CFG["device"])
            masks = masks.to(CFG["device"])
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 计算准确率
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
            
            # 计算IoU
            for pred, mask in zip(preds, masks):
                iou = calculate_iou(pred.cpu().numpy(), mask.cpu().numpy())
                iou_scores.append(iou)
    
    # 输出结果
    accuracy = total_correct / total_pixels
    mean_iou = np.mean(iou_scores)
    print(f"测试集评估结果（样本数：{len(test_dataset)}）:")
    print(f"整体准确率：{accuracy*100:.2f}%")
    print(f"平均IoU：{mean_iou:.4f}")
    
    # 可视化3个随机样本
    visualize_samples(test_dataset, model, n_samples=3)

# ================== 辅助函数 ==================
def calculate_iou(pred, target):
    """支持多类别IoU计算"""
    ious = []
    # 遍历所有类别（包括背景）
    for cls in range(CFG["n_classes"]):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:  # 没有该类别时跳过
            continue
        ious.append(intersection / union)
    return np.nanmean(ious) if ious else 0.0

def visualize_samples(dataset, model, n_samples=3):
    """可视化预测对比"""
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    plt.figure(figsize=(15, 5*n_samples))
    
    for i, idx in enumerate(indices):
        image, true_mask = dataset[idx]
        image_tensor = image.unsqueeze(0).to(CFG["device"])
        
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # 显示原始图像
        plt.subplot(n_samples, 3, i*3+1)
        
        # 修改后的图像处理方式
        display_image = image.permute(1, 2, 0).cpu().numpy()  # 直接转换通道顺序
        display_image = np.clip(display_image, 0, 1)  # 确保数值在合理范围
        
        plt.imshow(display_image)
        plt.title("Input Image")
        plt.axis('off')
        
        # 显示真实标签
        plt.subplot(n_samples, 3, i*3+2)
        plt.imshow(true_mask.cpu().numpy(), cmap='gray', vmin=0, vmax=CFG["n_classes"]-1)
        plt.title("Ground Truth")
        plt.axis('off')
        
        # 显示预测结果
        plt.subplot(n_samples, 3, i*3+3)
        plt.imshow(pred_mask, cmap='gray', vmin=0, vmax=CFG["n_classes"]-1)
        plt.title("Prediction")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ================== 使用方式 ==================
if __name__ == "__main__":
    train()          # 训练模型
    evaluate_testset()  # 评估测试集
    # predict("your_image.jpg")  # 单张图像预测