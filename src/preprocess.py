"""
Data preprocessing and augmentation functions for posture recognition.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image.

    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (height, width)

    Returns:
        numpy.ndarray: Preprocessed image array
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image, target_size)

        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0

        return image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def load_dataset_from_folders(data_dir, target_size=(224, 224)):
    """
    Load dataset from folder structure:
    data_dir/
        ├── correct/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── incorrect/
            ├── img3.jpg
            └── img4.jpg

    Args:
        data_dir (str): Root directory containing class folders
        target_size (tuple): Target size for images

    Returns:
        tuple: (images, labels, class_names)
    """
    images = []
    labels = []
    class_names = ['correct', 'incorrect']

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist")
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_dir, filename)
                image = load_and_preprocess_image(image_path, target_size)

                if image is not None:
                    images.append(image)
                    labels.append(class_idx)

    if not images:
        print("No images found. Creating synthetic dataset...")
        return create_synthetic_dataset(target_size)

    return np.array(images), np.array(labels), class_names


def create_synthetic_dataset(target_size=(224, 224), num_samples=1000):
    """
    Create a synthetic dataset for demonstration purposes.
    This simulates posture data with different patterns.

    Args:
        target_size (tuple): Image dimensions
        num_samples (int): Number of samples to generate

    Returns:
        tuple: (images, labels, class_names)
    """
    print("Creating synthetic posture dataset...")

    images = []
    labels = []
    class_names = ['correct', 'incorrect']

    for i in range(num_samples):
        # Create synthetic image
        image = np.random.rand(*target_size, 3).astype(np.float32)

        # Add some structure to simulate posture differences
        if i % 2 == 0:  # "Correct" posture
            # Add vertical lines to simulate good alignment
            image[:, target_size[1] // 2 - 5:target_size[1] // 2 + 5, :] += 0.3
            label = 0
        else:  # "Incorrect" posture
            # Add diagonal pattern to simulate poor alignment
            for j in range(target_size[0]):
                col = int(j * 0.5) % target_size[1]
                if col < target_size[1]:
                    image[j, col, :] += 0.3
            label = 1

        # Normalize to [0, 1]
        image = np.clip(image, 0, 1)

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels), class_names


def create_data_generators(X_train, X_val, y_train, y_val, batch_size=32):
    """
    Create data generators with augmentation for training.

    Args:
        X_train, X_val: Training and validation images
        y_train, y_val: Training and validation labels
        batch_size (int): Batch size for training

    Returns:
        tuple: (train_generator, val_generator)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # No augmentation for validation
    val_datagen = ImageDataGenerator()

    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )

    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, val_generator


def prepare_data(data_dir=None, target_size=(224, 224), test_size=0.2, random_state=42):
    """
    Complete data preparation pipeline.

    Args:
        data_dir (str): Directory containing the dataset
        target_size (tuple): Target image size
        test_size (float): Fraction of data for validation
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_val, y_train, y_val, class_names)
    """
    print("Loading and preparing dataset...")

    # Load dataset
    if data_dir and os.path.exists(data_dir):
        images, labels, class_names = load_dataset_from_folders(data_dir, target_size)
    else:
        images, labels, class_names = create_synthetic_dataset(target_size)

    print(f"Dataset loaded: {len(images)} images, {len(set(labels))} classes")

    # Convert labels to categorical
    labels_categorical = to_categorical(labels, num_classes=len(class_names))

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_categorical,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    return X_train, X_val, y_train, y_val, class_names


def visualize_samples(images, labels, class_names, num_samples=8):
    """
    Visualize sample images from the dataset.

    Args:
        images: Array of images
        labels: Array of labels
        class_names: List of class names
        num_samples: Number of samples to display
    """
    plt.figure(figsize=(12, 8))

    for i in range(min(num_samples, len(images))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])

        if len(labels.shape) > 1:  # One-hot encoded
            label_idx = np.argmax(labels[i])
        else:  # Integer labels
            label_idx = labels[i]

        plt.title(f"Class: {class_names[label_idx]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing preprocessing functions...")

    # Prepare data (will create synthetic dataset if no real data found)
    X_train, X_val, y_train, y_val, class_names = prepare_data()

    # Visualize some samples
    visualize_samples(X_train, y_train, class_names)

    # Create data generators
    train_gen, val_gen = create_data_generators(X_train, X_val, y_train, y_val)

    print("Preprocessing setup complete!")