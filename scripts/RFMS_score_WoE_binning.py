import pandas as pd
import logging
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import roc_auc_score, classification_report

# Logging Configuration
# =====================
log_dir = os.path.join(os.getcwd(), 'logs')

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

# Load Dataset
# ============
try:
    logger.info("Loading dataset")
    df = pd.read_csv("../data/processed_data.csv")
    logger.info("Dataset loaded successfully with shape: %s", df.shape)
except Exception as e:
    logger.error("Error loading dataset: %s", e)
    raise

class RFMSCalculator:
    def __init__(self, df):
        self.df = df

    def calculate_rfms(self):
        print("Starting RFMS calculation")
        self._calculate_recency()
        freq_df = self._calculate_frequency()
        mon_value_df = self._calculate_monetary_value()
        
        # Rename the conflicting columns in one of the DataFrames before merging
        mon_value_df.rename(columns={'AverageTransactionAmount': 'MonetaryAverageTransactionAmount'}, inplace=True)

        # Merge Frequency and Monetary DataFrames
        rfms_df = pd.merge(freq_df, mon_value_df, on='CustomerId', how='outer')
        
        # Debug: Check merged DataFrame columns and preview the data
        print("Merged RFMS DataFrame columns:", rfms_df.columns.tolist())
        print(rfms_df.head())
        
        # Ensure all necessary columns for RFMS Score calculation are present
        required_columns = ['TotalTransactionAmount', 'TotalTransactions', 'MonetaryAverageTransactionAmount']
        for col in required_columns:
            if col not in rfms_df.columns:
                print(f"Error: '{col}' column missing in RFMS DataFrame")
                return rfms_df  # Return to stop further processing if column is missing
        
        # Calculate RFMS Score using renamed column
        rfms_df['RFMS_Score'] = (
            rfms_df['TotalTransactionAmount'].fillna(0) + 
            rfms_df['MonetaryAverageTransactionAmount'].fillna(0) + 
            rfms_df['TotalTransactions'].fillna(0)
        )

        # Debug: Check if RFMS_Score is created
        print("RFMS Score calculation completed")
        print("RFMS DataFrame after Score Calculation:")
        print(rfms_df[['CustomerId', 'RFMS_Score']].head())
        
        return rfms_df

    def _calculate_recency(self):
        print("Calculating Recency")
        self.df['TransactionStartTime'] = pd.to_datetime(
            self.df[['TransactionYear', 'TransactionMonth', 'TransactionDay']].astype(str).agg('-'.join, axis=1)
        )
        max_date = self.df['TransactionStartTime'].max()
        self.df['Recency'] = (max_date - self.df['TransactionStartTime']).dt.days
        print("Recency calculation completed")

    def _calculate_frequency(self):
        print("Calculating Frequency")
        freq_df = self.df.groupby('CustomerId').agg(
            TotalTransactions=('Amount', 'count'),
            AverageTransactionAmount=('Amount', 'mean'),
            StdTransactionAmount=('Amount', 'std')
        ).reset_index()
        print("Frequency calculation completed")
        return freq_df

    def _calculate_monetary_value(self):
        print("Calculating Monetary Value")
        mon_value_df = self.df.groupby('CustomerId').agg(
            TotalTransactionAmount=('Amount', 'sum'),
            AverageTransactionAmount=('Amount', 'mean'),
            StdTransactionAmount=('Amount', 'std')
        ).reset_index()
        print("Monetary value calculation completed")
        return mon_value_df

class Labeling:
    def __init__(self, df):
        self.df = df

    def assign_good_bad_labels(self):
        print("Assigning Good/Bad labels")
        # Check if RFMS_Score is in the DataFrame
        if 'RFMS_Score' not in self.df.columns:
            print("Error: 'RFMS_Score' column not found in DataFrame")
            return self.df  # Exit the method if the column is missing

        # Assign labels based on thresholding RFMS (here it's simplified for illustrative purposes)
        threshold = self.df['RFMS_Score'].median()
        self.df['RFMS_Label'] = np.where(self.df['RFMS_Score'] >= threshold, 0, 1)
        self.df['User_Label'] = np.where(self.df['RFMS_Label'] == 0, 'Good', 'Bad')
        print("Good/Bad labels assigned")
        return self.df

# Perform WoE Binning
# ===================
class WoEBinning:
    def __init__(self, df, feature, target):
        """
        Initializes the WoE binning class with DataFrame and selected feature and target.
        Parameters:
        df : pd.DataFrame
            DataFrame containing the feature and target.
        feature : str
            Feature to be binned.
        target : str
            Target variable (Good/Bad or default status).
        """
        self.df = df
        self.feature = feature
        self.target = target
        logger.info("WoE Binning initialized for feature %s", feature)

    def calculate_woe(self):
        """
        Calculate WoE for each bin in the selected feature.
        Returns: DataFrame with WoE values.
        """
        logger.info("Calculating WoE for feature %s", self.feature)
        total_good = len(self.df[self.df[self.target] == 'Good'])
        total_bad = len(self.df[self.df[self.target] == 'Bad'])

        binned_df = pd.cut(self.df[self.feature], bins=10, duplicates='drop')

        # Group by the bins and calculate WoE
        woe_df = self.df.groupby(binned_df)[self.target].value_counts(normalize=False).unstack()
        woe_df['Good_Dist'] = woe_df['Good'] / total_good
        woe_df['Bad_Dist'] = woe_df['Bad'] / total_bad
        woe_df['WoE'] = np.log(woe_df['Good_Dist'] / woe_df['Bad_Dist'])

        logger.info("WoE calculation completed")
        return woe_df[['Good_Dist', 'Bad_Dist', 'WoE']]
