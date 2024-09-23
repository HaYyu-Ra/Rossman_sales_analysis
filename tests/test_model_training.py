import unittest
import numpy as np
import sys
import os

# Adjust the import path to include the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_training import train_model, evaluate_model, load_saved_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Generate synthetic classification data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Define path for saving the model
        self.model_path = 'test_model.h5'

    def test_train_model(self):
        """Test training of the model."""
        history = train_model(self.X_train, self.y_train, epochs=1, batch_size=32)
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertGreater(len(history.history['loss']), 0)

    def test_evaluate_model(self):
        """Test evaluation of the model."""
        model = train_model(self.X_train, self.y_train, epochs=1, batch_size=32, save_path=self.model_path)
        evaluation = evaluate_model(model, self.X_test, self.y_test)
        self.assertIsNotNone(evaluation)
        self.assertIn('accuracy', evaluation)
        self.assertGreater(evaluation['accuracy'], 0)

    def test_load_saved_model(self):
        """Test loading of a saved model."""
        model = train_model(self.X_train, self.y_train, epochs=1, batch_size=32, save_path=self.model_path)
        loaded_model = load_saved_model(self.model_path)
        self.assertIsNotNone(loaded_model)
        self.assertEqual(model.get_config(), loaded_model.get_config())  # Compare model configurations

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
