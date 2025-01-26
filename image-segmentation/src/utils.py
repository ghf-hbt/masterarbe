def visualize_predictions(images, masks, predictions, num_images=5):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5 * num_images))
    for i in range(num_images):
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(masks[i])
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(predictions[i])
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_model(model, filepath):
    import torch

    torch.save(model.state_dict(), filepath)


def load_model(model, filepath):
    import torch

    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model