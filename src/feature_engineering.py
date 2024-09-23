import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """
    Load data from a CSV file into a DataFrame.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data.
    """
    return pd.read_csv(file_path)

def create_features(df):
    """
    Create new features from the existing data.
    :param df: DataFrame with raw data.
    :return: DataFrame with new features added.
    """
    # Example: Extract features from the 'Date' column if it exists
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday
    
    # Example: Create interaction features
    if 'Open' in df.columns and 'Promo' in df.columns:
        df['Open_Promo'] = df['Open'] * df['Promo']
    
    return df

def encode_features(df):
    """
    Encode categorical variables into numeric values.
    :param df: DataFrame with categorical variables.
    :return: DataFrame with categorical variables encoded.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    if 'Date' in categorical_cols:
        categorical_cols = categorical_cols.drop('Date')  # Drop 'Date' if it's in categorical columns
    
    # One-Hot Encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded

def scale_features(df):
    """
    Scale numerical features in the DataFrame.
    :param df: DataFrame with numerical features.
    :return: DataFrame with scaled numerical features.
    """
    # Identify numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # StandardScaler for numerical features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def feature_engineering(file_path, output_path):
    """
    Perform feature engineering on the dataset and save the results.
    :param file_path: Path to the input CSV file.
    :param output_path: Path to save the processed CSV file.
    """
    # Load data
    df = load_data(file_path)
    
    # Create new features
    df = create_features(df)
    
    # Encode categorical features
    df = encode_features(df)
    
    # Scale numerical features
    df = scale_features(df)
    
    # Save the processed data
    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Processed data saved to {output_path}")

def main():
    input_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train.csv'
    output_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train_feature_engineered.csv'
    
    feature_engineering(input_path, output_path)

if __name__ == "__main__":
    main()
