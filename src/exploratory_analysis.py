import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename='exploratory_analysis.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load data paths
DATA_PATH = {
    "train": r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\train.csv",
    "test": r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\test.csv",
    "store": r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\store.csv"
}

# Load data
def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path, low_memory=False)

# Clean data
def clean_data(df):
    logging.info("Cleaning data")
    # Handle missing values
    df.ffill(inplace=True)  # Forward fill for missing values
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    return df

# Exploratory Data Analysis
def exploratory_analysis(train_df, test_df, store_df):
    logging.info("Starting exploratory analysis")

    # Check distributions of sales in train and test sets
    if 'Sales' not in train_df.columns or 'Sales' not in test_df.columns:
        logging.error("Sales column not found in train or test dataset")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(train_df['Sales'], bins=50, kde=True, color='blue', label='Train', stat="density")
    sns.histplot(test_df['Sales'], bins=50, kde=True, color='orange', label='Test', stat="density")
    plt.title('Sales Distribution in Train and Test Sets')
    plt.xlabel('Sales')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Sales behavior around holidays
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    train_df['is_holiday'] = train_df['StateHoliday'] != '0'
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='is_holiday', y='Sales', data=train_df)
    plt.title('Sales Behavior Before, During, and After Holidays')
    plt.xlabel('Is Holiday')
    plt.ylabel('Sales')
    plt.show()

    # Seasonal purchase behaviors
    train_df['Month'] = train_df['Date'].dt.month
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Month', y='Sales', data=train_df)
    plt.title('Seasonal Purchase Behaviors')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.show()

    # Correlation between sales and number of customers
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Customers', y='Sales', data=train_df)
    plt.title('Correlation between Sales and Number of Customers')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

    # Effect of promo on sales
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Promo', y='Sales', data=train_df)
    plt.title('Effect of Promotions on Sales')
    plt.xlabel('Promo')
    plt.ylabel('Sales')
    plt.show()

    # Stores open on all weekdays and their weekend sales
    weekday_sales = train_df[train_df['Open'] == 1].groupby('Store')['Sales'].agg(['mean', 'std'])
    plt.figure(figsize=(12, 6))
    sns.histplot(weekday_sales['mean'], bins=50, color='green')
    plt.title('Stores Open on All Weekdays Sales Distribution')
    plt.xlabel('Mean Sales')
    plt.ylabel('Frequency')
    plt.show()

    # Assortment type effect on sales
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Assortment', y='Sales', data=train_df)
    plt.title('Assortment Type Effect on Sales')
    plt.xlabel('Assortment Type')
    plt.ylabel('Sales')
    plt.show()

    # Distance to the next competitor effect on sales
    store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].median(), inplace=True)
    train_df = train_df.merge(store_df[['Store', 'CompetitionDistance']], on='Store', how='left')
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=train_df)
    plt.title('Effect of Distance to Competitor on Sales')
    plt.xlabel('Distance to Competitor')
    plt.ylabel('Sales')
    plt.show()

    # Analyze reopening of new competitors
    competitor_reopening = store_df[store_df['CompetitionDistance'].isnull() | (store_df['CompetitionDistance'] != 0)]
    plt.figure(figsize=(12, 6))
    sns.countplot(data=competitor_reopening, x='Store')
    plt.title('Reopening Competitors Impact on Sales')
    plt.xlabel('Store')
    plt.ylabel('Count')
    plt.show()

    logging.info("Exploratory analysis completed")

if __name__ == "__main__":
    # Load datasets
    train_data = load_data(DATA_PATH['train'])
    test_data = load_data(DATA_PATH['test'])
    store_data = load_data(DATA_PATH['store'])

    # Clean datasets
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)
    store_data = clean_data(store_data)

    # Perform exploratory analysis
    exploratory_analysis(train_data, test_data, store_data)
