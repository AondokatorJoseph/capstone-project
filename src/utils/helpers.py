import os
import cv2
import numpy as np
from datetime import datetime
import json

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image for prediction
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def save_model_metadata(model_path, training_history, metadata):
    """
    Save model training metadata and history
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = os.path.join(model_path, f'model_metadata_{timestamp}.json')
    
    metadata_dict = {
        'timestamp': timestamp,
        'training_history': {
            'loss': training_history.history['loss'],
            'accuracy': training_history.history['accuracy'],
            'val_loss': training_history.history['val_loss'],
            'val_accuracy': training_history.history['val_accuracy']
        },
        'model_info': metadata
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=4)

def get_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset
    """
    class_counts = np.bincount(labels)
    total = len(labels)
    class_weights = {i: total / (len(class_counts) * count) 
                    for i, count in enumerate(class_counts)}
    return class_weights

def create_experiment_folder():
    """
    Create a timestamped folder for experiment results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join('experiments', f'experiment_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def log_prediction(prediction, confidence, image_path, log_dir='prediction_logs'):
    """
    Log prediction results
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, 'predictions.log')
    
    with open(log_path, 'a') as f:
        log_entry = (f"{timestamp},"
                    f"{os.path.basename(image_path)},"
                    f"{prediction},"
                    f"{confidence:.4f}\n")
        f.write(log_entry)