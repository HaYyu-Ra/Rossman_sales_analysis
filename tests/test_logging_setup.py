import unittest
import logging
import os
from logging_setup import setup_logging  # Import the logging setup function

class TestLoggingSetup(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Define a path for the log file to test
        self.log_file = 'test_log.log'
        
        # Remove the log file if it already exists
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        
        # Set up logging configuration
        setup_logging(log_file=self.log_file)

    def test_logging(self):
        """Test the logging configuration."""
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)
        
        # Test logging at different levels
        logger.debug('This is a debug message')
        logger.info('This is an info message')
        logger.warning('This is a warning message')
        logger.error('This is an error message')
        logger.critical('This is a critical message')
        
        # Check if the log file is created and has log messages
        self.assertTrue(os.path.exists(self.log_file))
        
        # Read the log file and check its content
        with open(self.log_file, 'r') as file:
            log_content = file.read()
        
        # Check if all log messages are present in the log file
        self.assertIn('This is a debug message', log_content)
        self.assertIn('This is an info message', log_content)
        self.assertIn('This is a warning message', log_content)
        self.assertIn('This is an error message', log_content)
        self.assertIn('This is a critical message', log_content)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the log file after test
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

if __name__ == '__main__':
    unittest.main()
