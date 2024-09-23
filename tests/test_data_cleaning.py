import unittest
import pandas as pd
from data_cleaning import clean_data  # Import the function you want to test

class TestDataCleaning(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Load sample data for testing
        self.raw_data_path = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/sample_submission.csv'
        self.clean_data_path = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/clean_data.csv'
        self.raw_data = pd.read_csv(self.raw_data_path)
        self.expected_clean_data = pd.read_csv(self.clean_data_path)

    def test_clean_data(self):
        """Test the data cleaning function."""
        cleaned_data = clean_data(self.raw_data)
        
        # Check if the cleaned data is a DataFrame
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        
        # Check if the cleaned data matches the expected clean data
        pd.testing.assert_frame_equal(cleaned_data, self.expected_clean_data)

if __name__ == '__main__':
    unittest.main()
