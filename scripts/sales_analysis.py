import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
clean_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/clean_data.csv"
sample_submission_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/sample_submission.csv"
store_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/store.csv"
synthetic_sales_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/synthetic_sales_data.csv"
test_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/test.csv"
train_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/data/train.csv"

# Function to load and clean sales data
def load_and_clean_data(filepath):
    """
    Loads sales data from the provided CSV file and cleans it.

    Args:
    - filepath (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(filepath)

    # Handle missing values
    df.fillna(0, inplace=True)

    # Convert 'Date' to datetime format if it exists in the dataset
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    return df

# Load specific datasets using their paths
def load_all_datasets():
    """
    Loads all datasets into separate DataFrames.
    
    Returns:
    A tuple of DataFrames for clean_data, store_data, train_data, test_data, sample_submission, and synthetic_sales_data.
    """
    clean_data = load_and_clean_data(clean_data_path)
    store_data = load_and_clean_data(store_data_path)
    train_data = load_and_clean_data(train_data_path)
    test_data = load_and_clean_data(test_data_path)
    sample_submission = load_and_clean_data(sample_submission_path)
    synthetic_sales_data = load_and_clean_data(synthetic_sales_data_path)
    
    return clean_data, store_data, train_data, test_data, sample_submission, synthetic_sales_data

# Example function to analyze sales trend in training data
def sales_trend_analysis(df):
    """
    Analyzes the trend of sales over time from the training data.

    Args:
    - df (pd.DataFrame): Sales data.

    Returns:
    - None
    """
    if 'Date' in df.columns and 'Sales' in df.columns:
        # Group by date and calculate total sales for each date
        sales_trend = df.groupby('Date')['Sales'].sum()

        # Plot sales trend over time
        plt.figure(figsize=(12, 6))
        sales_trend.plot()
        plt.title('Sales Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("The dataset does not contain 'Date' or 'Sales' columns.")

# Function to plot weekly sales
def plot_weekly_sales(df):
    """
    Plots the total sales on a weekly basis.

    Args:
    - df (pd.DataFrame): Sales data containing 'Date' and 'Sales' columns.

    Returns:
    - None
    """
    if 'Date' in df.columns and 'Sales' in df.columns:
        # Convert 'Date' to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Set 'Date' as index
        df.set_index('Date', inplace=True)

        # Resample to get weekly sales
        weekly_sales = df['Sales'].resample('W').sum()

        # Plot weekly sales
        plt.figure(figsize=(12, 6))
        weekly_sales.plot()
        plt.title('Weekly Sales')
        plt.xlabel('Week')
        plt.ylabel('Total Sales')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("The dataset does not contain 'Date' or 'Sales' columns.")

# Function to merge store and train data
def merge_store_train(store_df, train_df):
    """
    Merges the store data with the training data.

    Args:
    - store_df (pd.DataFrame): Store information.
    - train_df (pd.DataFrame): Sales training data.

    Returns:
    - pd.DataFrame: Merged DataFrame.
    """
    merged_df = pd.merge(train_df, store_df, on='Store', how='left')
    return merged_df

# Function to calculate and display correlations
def calculate_correlations(df):
    """
    Calculates and displays the top 10 features that are most correlated with Sales.

    Args:
    - df (pd.DataFrame): Sales DataFrame containing numerical columns and 'Sales'.

    Returns:
    - None
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate correlations
    correlations = df[numeric_columns].corr()['Sales'].abs().sort_values(ascending=False)

    # Select top 10 correlated features (excluding 'Sales' itself)
    top_features = correlations[1:11].index.tolist()

    # Create correlation matrix for these features
    f_correlation = df[top_features].corr()

    # Generate a mask for the upper triangle
    f_mask = np.triu(np.ones_like(f_correlation, dtype=bool))

    # Set up the matplotlib figure
    f_fig, f_ax = plt.subplots(figsize=(12, 10))

    # Create the heatmap
    sns.heatmap(f_correlation, mask=f_mask, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Top 10 Features Correlated with Sales', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Print the correlation values
    print("Top 10 Correlations with Sales:")
    print(correlations[top_features])

# Main execution function
if __name__ == "__main__":
    # Load all datasets
    clean_data, store_data, train_data, test_data, sample_submission, synthetic_sales_data = load_all_datasets()

    # Perform sales trend analysis on training data
    print("Analyzing sales trends in training data...")
    sales_trend_analysis(train_data)

    # Plot weekly sales trend
    print("Plotting weekly sales trend...")
    plot_weekly_sales(train_data)

    # Merge store data with train data and analyze
    merged_train_data = merge_store_train(store_data, train_data)
    print(f"Merged Data Shape: {merged_train_data.shape}")
    print(merged_train_data.head())

    # Calculate correlations in the merged data
    calculate_correlations(merged_train_data)
