import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

class DataPreprocessor:
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1)
        ])

    def preprocess_images(self, images, labels, augment=True):
        processed_images = []
        processed_labels = []

        # Convert to tensorflow tensors and normalize
        images = tf.convert_to_tensor(images, dtype=tf.float32) / 255.0
        
        # Original images
        processed_images.extend(images.numpy())
        processed_labels.extend(labels)  
        
        # Augmented images if requested
        if augment:
            augmented_images = self.data_augmentation(images)
            # Ensure values are clipped to [0, 1] range after augmentation
            augmented_images = tf.clip_by_value(augmented_images, 0.0, 1.0)
            processed_images.extend(augmented_images.numpy())
            processed_labels.extend(labels)  
        
        # Final safety check - clip all values to [0, 1]
        processed_images = np.clip(np.array(processed_images), 0.0, 1.0)
        
        return processed_images, np.array(processed_labels)

    def create_train_val_test_split(self, images, labels, val_size=0.15, test_size=0.15):
        """Split data into train/validation/test sets"""
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, 
            test_size=(val_size + test_size), 
            random_state=42,
            stratify=labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=(test_size/(val_size + test_size)), 
            random_state=42,
            stratify=y_temp
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }