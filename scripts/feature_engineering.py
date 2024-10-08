import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

class AggregateFeatures:
    def __init__(self, df):
        """
        Initialize with the dataframe.
        """
        self.df = df
        logger.info("AggregateFeatures initialized with dataframe of shape %s", df.shape)

    def sum_all_transactions(self):
        """
        Calculate the total transaction amount per customer and merge with the original dataframe.
        """
        logger.info("Calculating total transaction amount per customer.")
        try:
            customer_transaction_sum = self.df.groupby('CustomerId')['Amount'].sum().reset_index()
            customer_transaction_sum.rename(columns={'Amount': 'TotalTransactionAmount'}, inplace=True)
            self.df = self.df.merge(customer_transaction_sum, on='CustomerId', how='left')
            logger.info("Total transaction amount per customer calculated successfully.")
        except Exception as e:
            logger.error("Error calculating total transaction amount per customer: %s", e)

    def average_transaction_amount(self):
        """
        Calculate the average transaction amount per customer and merge with the original dataframe.
        """
        logger.info("Calculating average transaction amount per customer.")
        try:
            average_transaction_amount = self.df.groupby('CustomerId')['Amount'].mean().reset_index()
            average_transaction_amount.rename(columns={'Amount': 'AverageTransactionAmount'}, inplace=True)
            self.df = self.df.merge(average_transaction_amount, on='CustomerId', how='left')
            logger.info("Average transaction amount per customer calculated successfully.")
        except Exception as e:
            logger.error("Error calculating average transaction amount per customer: %s", e)

    def transaction_count(self):
        """
        Calculate the number of transactions per customer and merge with the original dataframe.
        """
        logger.info("Calculating number of transactions per customer.")
        try:
            transaction_per_customer = self.df.groupby('CustomerId')['TransactionId'].count().reset_index()
            transaction_per_customer.rename(columns={'TransactionId': 'TotalTransactions'}, inplace=True)
            self.df = self.df.merge(transaction_per_customer, on='CustomerId', how='left')
            logger.info("Number of transactions per customer calculated successfully.")
        except Exception as e:
            logger.error("Error calculating number of transactions per customer: %s", e)

    def standard_deviation_amount(self):
        """
        Calculate the standard deviation of transaction amounts per customer and merge with the original dataframe.
        """
        logger.info("Calculating standard deviation of transaction amounts per customer.")
        try:
            standard_deviation_transaction_amount = self.df.groupby('CustomerId')['Amount'].std().reset_index()
            standard_deviation_transaction_amount.rename(columns={'Amount': 'StdTransactionAmount'}, inplace=True)
            self.df = self.df.merge(standard_deviation_transaction_amount, on='CustomerId', how='left')
            logger.info("Standard deviation of transaction amounts per customer calculated successfully.")
        except Exception as e:
            logger.error("Error calculating standard deviation of transaction amounts per customer: %s", e)

    def get_dataframe(self):
        """
        Return the modified dataframe.
        """
        logger.info("Returning the modified dataframe.")
        return self.df
    

class Extracting_features:
    def __init__(self, df):
        """
        Initialize with the dataframe.
        """
        self.df = df
        logger.info("Extracting_features initialized with dataframe of shape %s", df.shape)

    def transaction_hour(self):
        """
        Extract the hour of the transaction from the TransactionStartTime column.
        """
        #Convert the TransactionStartTime to datetime format
        logger.info("Converting TransactionStartTime to datetime")
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])

        logger.info("Extracting the hour of the transaction.")
        try:
            self.df['TransactionHour'] = self.df['TransactionStartTime'].dt.hour
            logger.info("Hour of the transaction extracted successfully.")
        except Exception as e:
            logger.error("Error extracting the hour of the transaction: %s", e)

    def transaction_day(self):
        """
        Extract the day of the transaction from the TransactionStartTime column.
        """
        logger.info("Extracting the day of the transaction.")
        try:
            self.df['TransactionDay'] = self.df['TransactionStartTime'].dt.day
            logger.info("Day of the transaction extracted successfully.")
        except Exception as e:
            logger.error("Error extracting the day of the transaction: %s", e)
    def transaction_month(self):
        """
        Extract the month of the transaction from the TransactionStartTime column.
        """
        logger.info("Extracting the month of the transaction.")
        try:
            self.df['TransactionMonth'] = self.df['TransactionStartTime'].dt.month
            logger.info("Month of the transaction extracted successfully.")
        except Exception as e:
            logger.error("Error extracting the month of the transaction: %s", e)
    def transaction_year(self):
        """
        Extract the year of the transaction from the TransactionStartTime column.
        """
        logger.info("Extracting the year of the transaction.")
        try:
            self.df['TransactionYear'] = self.df['TransactionStartTime'].dt.year
            logger.info("Year of the transaction extracted successfully.")
        except Exception as e:
            logger.error("Error extracting the year of the transaction: %s", e)
    def get_dataframe(self):
        """
        Return the modified dataframe.
        """
        logger.info("Returning the modified dataframe.")
        return self.df
    

def normalize_numerical_features(df, numerical_cols, method='standardize'):
    """
    Normalizes or standardizes numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with numerical features.
    numerical_cols : list
        List of numerical columns to normalize/standardize.
    method : str, optional
        The method for scaling ('standardize' or 'normalize').

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized or standardized numerical features.
    """
    logger.info("Starting normalization/standardization of numerical features.")
    try:
        if method == 'standardize':
            scaler = StandardScaler()
            logger.info("Using StandardScaler for standardization.")
        elif method == 'normalize':
            scaler = MinMaxScaler()
            logger.info("Using MinMaxScaler for normalization.")
        else:
            logger.error("Invalid method provided for scaling: %s", method)
            raise ValueError("Method must be either 'standardize' or 'normalize'.")

        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logger.info("Numerical features %s scaled successfully using method: %s", numerical_cols, method)
        
        # Set the TransactionId to index
        df.set_index('TransactionId', inplace=True)
        logger.info("TransactionId set as index successfully.")
        
        return df
    except Exception as e:
        logger.error("Error in normalization/standardization of numerical features: %s", e)
        raise