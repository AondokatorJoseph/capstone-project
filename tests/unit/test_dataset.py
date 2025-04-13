import unittest
import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data.dataset_loader import WasteDatasetLoader
from src.data.preprocessing import DataPreprocessor

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.loader = WasteDatasetLoader()
        cls.preprocessor = DataPreprocessor()
    
    def test_trashnet_loading(self):
        """Test TrashNet dataset loading"""
        images, labels = self.loader.load_trashnet()
        self.assertIsNotNone(images)
        self.assertIsNotNone(labels)
        self.assertEqual(len(images), len(labels))
        self.assertTrue(isinstance(images, np.ndarray))
        
    def test_preprocessing(self):
        """Test image preprocessing"""
        images, labels = self.loader.load_trashnet()
        processed_images, processed_labels = self.preprocessor.preprocess_images(
            images[:5], labels[:5]
        )
        self.assertEqual(processed_images.shape[1:], (224, 224, 3))
        self.assertTrue(np.max(processed_images) <= 1.0)
        
    def test_data_split(self):
        """Test train/val/test split"""
        images, labels = self.loader.load_trashnet()
        data_splits = self.preprocessor.create_train_val_test_split(
            images[:100], labels[:100]
        )
        self.assertIn('train', data_splits)
        self.assertIn('val', data_splits)
        self.assertIn('test', data_splits)

if __name__ == '__main__':
    unittest.main()