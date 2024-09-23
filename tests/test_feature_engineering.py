import unittest
import pandas as pd
from feature_engineering import feature_engineer  # Import the function you want to test

class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Load sample data for testing
        self.raw_data_path = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/train.csv'
        self.expected_features_path = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/expected_features.csv'
        self.raw_data = pd.read_csv(self.raw_data_path)
        self.expected_features = pd.read_csv(self.expected_features_path)

    def test_feature_engineering(self):
        """Test the feature engineering function."""
        engineered_features = feature_engineer(self.raw_data)
        
        # Check if the engineered features data is a DataFrame
        self.assertIsInstance(engineered_features, pd.DataFrame)
        
        # Check if the engineered features match the expected features
        pd.testing.assert_frame_equal(engineered_features, self.expected_features)

if __name__ == '__main__':
    unittest.main()
