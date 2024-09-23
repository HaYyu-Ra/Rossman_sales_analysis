import logging
import os

def setup_logging(log_file_path='app.log'):
    """
    Sets up the logging configuration.
    :param log_file_path: Path to the log file where logs will be stored.
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure the logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),  # Log to a file
            logging.StreamHandler()  # Also log to the console
        ]
    )

def main():
    """
    Main function to test logging setup.
    """
    # Setup logging
    setup_logging('C:\\Users\\hayyu.ragea\\AppData\\Local\\Programs\\Python\\Python312\\Rossmann_Sales_Forecasting_Project\\logs\\app.log')
    
    # Test logging
    logger = logging.getLogger(__name__)
    
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')

if __name__ == "__main__":
    main()
