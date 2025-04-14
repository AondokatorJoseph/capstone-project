Waste Classification System
A comprehensive machine learning system for classifying waste materials into different categories (glass, paper, cardboard, plastic, metal, and trash) using computer vision and deep learning techniques.

Table of Contents
Overview
Project Structure
Installation
Usage
Dataset Information
Model Architecture
Training Process
Evaluation
Contributing
License
Overview
This project implements a waste classification system using convolutional neural networks. The system is trained on the TrashNet dataset and can optionally incorporate COCO dataset images. The goal is to accurately classify waste items to promote proper recycling and waste management.

Project Structure
main_capstone/
├── src/                      # Source code
│   ├── data/                 # Data handling modules
│   │   ├── dataset_loader.py # Dataset loading functionality
│   │   └── preprocessing.py  # Image preprocessing
│   ├── models/               # Model definition and training
│   │   ├── model.py          # Model architecture
│   │   └── training.ipynb    # Training notebook
│   ├── demo/                 # Demo applications
│   └── utils/                # Utility functions
├── tests/                    # Tests
│   ├── unit/                 # Unit tests
│   └── test_notebooks/       # Test notebooks
├── datasets/                 # Dataset storage (not in repo)
│   ├── custom_waste/         
│   │   └── trashnet/         # TrashNet dataset
│   └── coco/                 # COCO dataset
├── models/                   # Saved models
│   └── saved_models/         
├── docs/                     # Documentation
├── requirements.txt          # Project dependencies
├── setup.py                  # Project setup script
└── README.md                 # This file

Installation
1. Clone the repository:
git clone <repository-url>
cd main_capstone

2. Create a virtual environment (optional but recommended):
python -m venv myenv

3. Activate the virtual environment:
    Windows
myenv\Scripts\activate

4. Install dependencies
pip install -r requirements.txt

5. Install the project in development mode
pip install -e .


Usage
Loading and Preprocessing Data

from src.data.dataset_loader import WasteDatasetLoader
from src.data.preprocessing import DataPreprocessor

# Initialize loaders
loader = WasteDatasetLoader()
preprocessor = DataPreprocessor()

# Load datasets
images, labels = loader.load_trashnet()
# Or use combined datasets
# images, labels = loader.load_combined_datasets()

# Preprocess images
processed_images, processed_labels = preprocessor.preprocess_images(images, labels)

# Create train/val/test splits
data_splits = preprocessor.create_train_val_test_split(processed_images, processed_labels)

Training a Model
from src.models.model import WasteClassifier
import tensorflow as tf

# Initialize training strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Initialize model
    classifier = WasteClassifier()
    model = classifier.build_model()
    
    # Train model
    history = classifier.train(
        train_data=data_splits['train'],
        validation_data=data_splits['val'],
        epochs=40,
        batch_size=32
    )

Evaluating a Model

# Evaluate on test set
test_loss, test_acc = classifier.model.evaluate(
    data_splits['test'][0], 
    data_splits['test'][1]
)

# Get predictions
y_pred = classifier.model.predict(data_splits['test'][0])
y_pred_classes = np.argmax(y_pred, axis=1)

Running Tests
# Run unit tests
python -m unittest discover tests/unit

# Run a specific test
python tests/unit/test_dataset.py

Running the StreamLit Application 
1. Navigate to src/demo folder
in the command prompt run Streamlit run app.py

Dataset Information
The project uses the following datasets:

TrashNet: A dataset containing images of waste items in six categories (glass, paper, cardboard, plastic, metal, trash).
COCO: Common Objects in Context dataset, filtered to include relevant waste-related categories.
Model Architecture
The waste classification model is built using a convolutional neural network architecture, implemented in the WasteClassifier class.

Training Process
Training is performed with data augmentation to improve model generalization. The process includes:

Loading and preprocessing images
Data augmentation (random rotations, flips, zoom, brightness)
Train/validation/test splitting
Model training with early stopping and learning rate scheduling
Model evaluation on test data
Evaluation
The model is evaluated using:

Accuracy
Loss
Classification report (precision, recall, F1-score)
Confusion matrix
Contributing
Contributions are welcome. Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License.