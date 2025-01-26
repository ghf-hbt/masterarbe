def visualize(original_image, segmented_image):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image)
    plt.axis('off')

    plt.show()

def calculate_metrics(true_mask, predicted_mask):
    from sklearn.metrics import jaccard_score, accuracy_score

    # Flatten the masks
    true_mask_flat = true_mask.flatten()
    predicted_mask_flat = predicted_mask.flatten()

    # Calculate Jaccard index (IoU)
    jaccard_index = jaccard_score(true_mask_flat, predicted_mask_flat, average='weighted')

    # Calculate accuracy
    accuracy = accuracy_score(true_mask_flat, predicted_mask_flat)

    return {
        'jaccard_index': jaccard_index,
        'accuracy': accuracy
    }