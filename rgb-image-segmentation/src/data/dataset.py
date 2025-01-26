class ImageDataset:
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)

        if self.transform:
            image = self.transform(image)

        return image

    def load_image(self, path):
        # 这里可以使用PIL或OpenCV等库加载图像
        pass