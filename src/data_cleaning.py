import pandas as pd
import numpy as np

def load_data(file_paths):
    """
    Load data from multiple CSV files into a dictionary of DataFrames.
    :param file_paths: List of file paths to CSV files.
    :return: Dictionary of DataFrames keyed by file names.
    """
    data_frames = {}
    for file_path in file_paths:
        file_name = file_path.split('\\')[-1]
        data_frames[file_name] = pd.read_csv(file_path)
    return data_frames

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    :param df: DataFrame with missing values.
    :return: DataFrame with missing values handled.
    """
    # Example: Fill missing values with mean for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Example: Fill missing values with mode for categorical columns
    categorical_cols = df.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    :param df: DataFrame with potential duplicates.
    :return: DataFrame with duplicates removed.
    """
    return df.drop_duplicates()

def standardize_data_types(df):
    """
    Standardize data types in the DataFrame.
    :param df: DataFrame with mixed data types.
    :return: DataFrame with standardized data types.
    """
    # Example: Convert date columns to datetime type
    date_cols = ['Date']  # Replace with your date columns
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Example: Convert categorical columns to category type
    categorical_cols = df.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables into numeric values.
    :param df: DataFrame with categorical variables.
    :return: DataFrame with categorical variables encoded.
    """
    # Example: One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def handle_outliers(df, threshold=1.5):
    """
    Handle outliers in numeric columns.
    :param df: DataFrame with numeric columns.
    :param threshold: Threshold for detecting outliers.
    :return: DataFrame with outliers handled.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outliers
        outliers = (df[col] < (Q1 - threshold * IQR)) | (df[col] > (Q3 + threshold * IQR))
        
        # Option: Replace outliers with median
        df.loc[outliers, col] = df[col].median()
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    :param df: Cleaned DataFrame.
    :param output_path: Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)

def clean_all_data(file_paths, output_paths):
    """
    Clean multiple datasets and save them to new CSV files.
    :param file_paths: List of input file paths.
    :param output_paths: List of output file paths.
    """
    data_frames = load_data(file_paths)
    
    for file_name, df in data_frames.items():
        print(f"Cleaning {file_name}...")
        df = handle_missing_values(df)
        df = remove_duplicates(df)
        df = standardize_data_types(df)
        df = encode_categorical_variables(df)
        df = handle_outliers(df)
        
        # Determine the corresponding output path
        output_path = output_paths[file_name]
        save_cleaned_data(df, output_path)
        print(f"Saved cleaned data to {output_path}")

def main():
    input_paths = {
        'clean_data.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\clean_data.csv',
        'sample_submission.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\sample_submission.csv',
        'store.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\store.csv',
        'synthetic_sales_data.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\synthetic_sales_data.csv',
        'test.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\test.csv',
        'train.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train.csv',
    }
    
    output_paths = {
        'clean_data.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\clean_data_cleaned.csv',
        'sample_submission.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\sample_submission_cleaned.csv',
        'store.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\store_cleaned.csv',
        'synthetic_sales_data.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\synthetic_sales_data_cleaned.csv',
        'test.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\test_cleaned.csv',
        'train.csv': r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train_cleaned.csv',
    }

    clean_all_data(list(input_paths.values()), output_paths)

if __name__ == "__main__":
    main()
