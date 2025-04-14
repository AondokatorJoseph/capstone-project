# ♻️ Waste Classification System

A comprehensive machine learning system for classifying waste materials into categories — **glass, paper, cardboard, plastic, metal, and trash** — using computer vision and deep learning techniques.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Loading and Preprocessing Data](#loading-and-preprocessing-data)
  - [Training a Model](#training-a-model)
  - [Evaluating a Model](#evaluating-a-model)
  - [Running Tests](#running-tests)
  - [Running the Streamlit Application](#running-the-streamlit-application)
- [Dataset Information](#dataset-information)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## 📝 Overview

This project implements a waste classification system using convolutional neural networks (CNNs). Trained on the **TrashNet** dataset, with optional integration of **COCO** dataset images, the system aims to accurately classify waste items to encourage better recycling and waste management.

---

## 📁 Project Structure


---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd main_capstone

python -m venv myenv

3. Activate the virtual environment
myenv\Scripts\activate

4. Install dependencies:
pip install -r requirements.txt
5. Install the project in development mode:
pip install -e .

🚀 Usage
🔄 Loading and Preprocessing Data
from src.data.dataset_loader import WasteDatasetLoader
from src.data.preprocessing import DataPreprocessor

loader = WasteDatasetLoader()
preprocessor = DataPreprocessor()

# Load datasets
images, labels = loader.load_trashnet()
# Optional: images, labels = loader.load_combined_datasets()

# Preprocess images
processed_images, processed_labels = preprocessor.preprocess_images(images, labels)

# Split into train/val/test
data_splits = preprocessor.create_train_val_test_split(processed_images, processed_labels)


🧠 Training a Model
from src.models.model import WasteClassifier
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    classifier = WasteClassifier()
    model = classifier.build_model()

    history = classifier.train(
        train_data=data_splits['train'],
        validation_data=data_splits['val'],
        epochs=40,
        batch_size=32
    )
📊 Evaluating a Model
test_loss, test_acc = classifier.model.evaluate(
    data_splits['test'][0], 
    data_splits['test'][1]
)

y_pred = classifier.model.predict(data_splits['test'][0])
y_pred_classes = np.argmax(y_pred, axis=1)

🧪 Running Tests
# Run all unit tests
python -m unittest discover tests/unit

# Run a specific test
python tests/unit/test_dataset.py

📺 Running the Streamlit Application
cd src/demo
streamlit run app.py
