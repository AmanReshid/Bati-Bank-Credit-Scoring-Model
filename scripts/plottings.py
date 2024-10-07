import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import os

# Logging configuration
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)

def univariate_analysis(numerical_value):
    try:
        logger.info("Starting univariate analysis")
        plt.figure(figsize=(20, 5))
        
        logger.info("Creating boxplot")
        plt.subplot(1, 4, 1)
        sns.boxplot(x=numerical_value)
        
        logger.info("Creating histogram without KDE")
        plt.subplot(1, 4, 2)
        sns.histplot(numerical_value, bins=20)
        
        logger.info("Creating histogram with KDE")
        plt.subplot(1, 4, 3)
        sns.histplot(numerical_value, bins=20, kde=True)
        
        logger.info("Creating histogram")
        plt.subplot(1, 4, 4)
        plt.hist(numerical_value, bins=20)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Univariate analysis completed successfully")
    except Exception as e:
        logger.error(f"Error during univariate analysis: {e}")
        raise


def univariate_analysis_categorical(categorical_value):
    try:
        logger.info("Starting univariate analysis for categorical data")
        plt.figure(figsize=(20, 5))
        
        logger.info("Creating count plot")
        plt.subplot(1, 2, 1)
        sns.countplot(x=categorical_value)
        
        logger.info("Creating pie chart")
        plt.subplot(1, 2, 2)
        categorical_value.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Univariate analysis for categorical data completed successfully")
    except Exception as e:
        logger.error(f"Error during univariate analysis for categorical data: {e}")
        raise





def plot_transactions_per_day(df, date_column, transaction_column):
    """
    Plots the number of transactions per day over time.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    date_column (str): The name of the column with the datetime information.
    transaction_column (str): The name of the column with the transaction IDs.
    
    Returns:
    None: Displays a plot of the transactions per day.
    """
    # Ensure the date_column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set the date_column as the index
    df.set_index(date_column, inplace=True)
    
    # Resample the data by day and count the transactions
    transactions_per_day = df[transaction_column].resample('D').count()
    
    # Plot the trend of transactions over time
    plt.figure(figsize=(10, 6))
    transactions_per_day.plot(title='Transactions Per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Transactions')
    plt.show()



def univariate_analysis(data, column_name):
    """
    Performs univariate analysis on a given array or DataFrame column.
    
    If the data is numerical, it plots the histogram and boxplot.
    If the data is categorical, it plots a bar chart for category distribution.
    
    Parameters:
    data (pd.Series or np.array): The data/column to be analyzed.
    column_name (str): The name of the column (used for labeling in the plot).
    
    Returns:
    None: Displays the appropriate plot based on the data type.
    """
    logger.info(f"univariate_analysis function called for column {column_name}")

    try:
        plt.figure(figsize=(12, 6))

        # Check if the data is numerical or categorical
        if data.dtype == 'object' or data.dtype == 'category':
            # Categorical data: Plot category distribution using a bar plot
            sns.countplot(y=data, order=data.value_counts().index, palette='viridis')
            plt.title(f'Category Distribution: {column_name}')
            plt.xlabel('Frequency')
            plt.ylabel(column_name)
            logger.info(f"Category distribution plot generated for column {column_name}")

        else:
            # Numerical data: Plot histogram and boxplot
            plt.subplot(1, 2, 1)
            sns.histplot(data, kde=True, bins=30, color='blue')
            plt.title(f'Distribution of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            sns.boxplot(data=data, color='blue')
            plt.title(f'Boxplot of {column_name}')
            plt.xlabel(column_name)
            logger.info(f"Histogram and boxplot generated for column {column_name}")
        
        # Show plots
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"An error occurred during univariate analysis for column {column_name}: {e}")



def correlation_analysis(df):
    """
    Performs correlation analysis on numerical columns of a DataFrame.
    Computes the correlation matrix and visualizes it as a heatmap.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing numerical columns.
    
    Returns:
    corr_matrix (pd.DataFrame): The correlation matrix.
    """
    logger.info("correlation_analysis function called.")
    
    try:
        # Select only the numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64'])
        logger.info(f"Numerical columns selected: {list(numerical_cols.columns)}")

        # Compute the correlation matrix
        corr_matrix = numerical_cols.corr()
        logger.info("Correlation matrix computed successfully.")

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
        plt.title('Correlation Heatmap of Numerical Features')
        plt.show()
        logger.info("Correlation heatmap displayed successfully.")

        return corr_matrix

    except Exception as e:
        logger.error(f"An error occurred during correlation analysis: {e}")
        return None
    


def plot_top_tens(df, numerical_cols):
    """
    Plots the top ten values for each numerical column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing numerical columns.
    
    Returns:
    None: Displays bar plots of the top ten values for each numerical column.
    """
    logger.info("plot_top_tens function called.")
    
    try:
        # Select only the numerical columns
        logger.info(f"Numerical columns selected: {list(numerical_cols.columns)}")

        # Plot top ten values for each numerical column
        for col in numerical_cols.columns:
            top_ten = df[col].nlargest(10)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_ten.values, y=top_ten.index, palette='viridis')
            plt.title(f'Top 10 Values of {col}')
            plt.xlabel(col)
            plt.ylabel('Index')
            plt.show()
            logger.info(f"Top ten values plot generated for column {col}")

    except Exception as e:
        logger.error(f"An error occurred while plotting top tens for numerical columns: {e}")


def plot_top_customers(top_customers_by_amount, top_customers_by_transactions):
    try:
        logger.info("Starting plot_top_customers function")

        # Plot top ten customers based on the amount they pay
        logger.info("Plotting top ten customers based on amount paid")
        plt.figure(figsize=(12, 6))
        top_customers_by_amount.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Customers Based on Amount Paid')
        plt.xlabel('CustomerId')
        plt.ylabel('Total Amount Paid')
        plt.xticks(rotation=45)
        plt.show()

        # Plot top ten customers based on the number of transactions
        logger.info("Plotting top ten customers based on number of transactions")
        plt.figure(figsize=(12, 6))
        top_customers_by_transactions.plot(kind='bar', color='lightgreen')
        plt.title('Top 10 Customers Based on Number of Transactions')
        plt.xlabel('CustomerId')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        plt.show()

        logger.info("plot_top_customers function completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in plot_top_customers function: {e}")
        raise