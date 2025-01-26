class SegmentationModel:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, dataloader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for images, masks in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
        return output

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, masks in dataloader:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        print(f'Average Loss: {average_loss:.4f}')