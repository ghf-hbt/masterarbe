import os
import cv2
import numpy as np
from collections import Counter

def load_data_with_matching(folder):
    """
    Loads .npy images and .png label masks from a single folder, ensuring they match based on filenames.

    Args:
        folder (str): Path to the folder containing images and labels.

    Returns:
        tuple: A tuple containing two lists - images and labels (matched by filename).
    """
    images = []
    labels = []

    # List all files in the folder
    files = os.listdir(folder)

    # Group files by their base names (without extensions)
    file_dict = {}
    for filename in files:
        if filename.endswith(".npy") or filename.endswith(".png"):
            base_name = os.path.splitext(filename)[0]  # Remove the extension
            if base_name not in file_dict:
                file_dict[base_name] = {}
            if filename.endswith(".npy"):
                file_dict[base_name]['image'] = os.path.join(folder, filename)
            elif filename.endswith(".png"):
                file_dict[base_name]['label'] = os.path.join(folder, filename)

    # Ensure pairs are complete (both .npy and .png exist for a base name)
    for base_name, paths in file_dict.items():
        if 'image' in paths and 'label' in paths:  # Check if both image and label exist
            # Load the image (.npy)
            image = np.load(paths['image'])
            # Load the label (.png)
            label = cv2.imread(paths['label'], cv2.IMREAD_GRAYSCALE)

            # Append to lists if both were successfully loaded
            if image is not None and label is not None:
                images.append(image)
                labels.append(label)

    return images, labels

def analyze_dataset(images, labels):
    """
    Analyzes a dataset of images and corresponding label masks.

    Args:
        images (list of np.ndarray): List of images as NumPy arrays.
        labels (list of np.ndarray): List of label masks as NumPy arrays.

    Returns:
        dict: A dictionary containing analysis results.
    """
    image_shapes = []
    label_shapes = []
    all_classes = Counter()

    for img, label in zip(images, labels):
        image_shapes.append(img.shape)
        label_shapes.append(label.shape)
        unique_classes = np.unique(label)
        all_classes.update(unique_classes)

    unique_image_shapes = Counter(image_shapes)
    unique_label_shapes = Counter(label_shapes)

    result = {
        "total_images": len(images),
        "unique_image_shapes": unique_image_shapes,
        "unique_label_shapes": unique_label_shapes,
        "total_classes": len(all_classes),
        "class_distribution": dict(all_classes)
    }

    return result

# Example usage
if __name__ == "__main__":
    # Specify the folder containing both images and labels
    folder = r"C:\Users\sm1508\Desktop\test"

    # Load the dataset
    images, labels = load_data_with_matching(folder)

    # Analyze the dataset
    analysis = analyze_dataset(images, labels)

    # Print the results
    print("Dataset Analysis:")
    print(f"Total images: {analysis['total_images']}")
    print(f"Unique image shapes: {analysis['unique_image_shapes']}")
    print(f"Unique label shapes: {analysis['unique_label_shapes']}")
    print(f"Total classes: {analysis['total_classes']}")
    print(f"Class distribution: {analysis['class_distribution']}")
