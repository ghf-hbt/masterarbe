def predict(model, image):
    import numpy as np
    import cv2
    import torch

    # 将图像转换为模型输入格式
    image = cv2.resize(image, (256, 256))  # 假设模型输入大小为256x256
    image = image / 255.0  # 归一化
    image = np.transpose(image, (2, 0, 1))  # 转换为C x H x W
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # 添加batch维度

    # 加载模型并进行预测
    model.eval()
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)  # 应用sigmoid激活函数
        output = output.squeeze().numpy()  # 移除多余的维度

    # 将输出转换为二值图像
    prediction = (output > 0.5).astype(np.uint8)  # 阈值处理

    return prediction