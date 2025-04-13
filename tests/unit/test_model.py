import unittest
import sys
import os
import numpy as np
import tensorflow as tf

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.model import WasteClassifier

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.model = WasteClassifier()
        
    def test_model_creation(self):
        """Test model architecture creation"""
        model = self.model.build_model()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 8)  # Check number of layers
        
    def test_model_output_shape(self):
        """Test model output shape"""
        model = self.model.build_model()
        test_input = np.random.random((1, 224, 224, 3))
        prediction = model.predict(test_input)
        self.assertEqual(prediction.shape, (1, 6))  # 6 classes
        
    def test_model_training(self):
        """Test model training"""
        # Create dummy data
        X = np.random.random((10, 224, 224, 3))
        y = np.random.randint(0, 6, 10)
        
        history = self.model.train(
            train_data=(X, y),
            validation_data=(X, y),
            epochs=1
        )
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)

if __name__ == '__main__':
    unittest.main()