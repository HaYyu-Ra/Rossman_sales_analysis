import unittest
import numpy as np
from tensorflow.keras.models import load_model
from lstm_model import create_lstm_model, train_lstm_model, predict_with_lstm_model

class TestLSTMModel(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Generate synthetic data for testing
        self.X_train = np.random.random((100, 10, 1))  # 100 samples, 10 time steps, 1 feature
        self.y_train = np.random.random((100, 1))
        self.X_test = np.random.random((10, 10, 1))  # 10 samples, 10 time steps, 1 feature
        self.y_test = np.random.random((10, 1))
        
        # Define paths for model saving
        self.model_path = 'test_lstm_model.h5'

    def test_create_lstm_model(self):
        """Test the creation of the LSTM model."""
        model = create_lstm_model(input_shape=(10, 1))  # Adjust input_shape as needed
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 4)  # Adjust based on your model architecture

    def test_train_lstm_model(self):
        """Test training of the LSTM model."""
        model = create_lstm_model(input_shape=(10, 1))
        history = train_lstm_model(model, self.X_train, self.y_train, epochs=1, batch_size=32)
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertGreater(len(history.history['loss']), 0)

    def test_predict_with_lstm_model(self):
        """Test prediction with the LSTM model."""
        model = create_lstm_model(input_shape=(10, 1))
        train_lstm_model(model, self.X_train, self.y_train, epochs=1, batch_size=32)
        predictions = predict_with_lstm_model(model, self.X_test)
        self.assertEqual(predictions.shape, (10, 1))  # Should match the shape of y_test

    def tearDown(self):
        """Clean up after tests."""
        # Remove the model file if it exists
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
