import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
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

def missing_values_table(df):
    logger.info("missing_values_table function called")
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_data_types = df.dtypes
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Otype'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    logger.info(f"DataFrame has {df.shape[1]} columns, {mis_val_table_ren_columns.shape[0]} columns have missing values")
    return mis_val_table_ren_columns

def column_summary(df):
    logger.info("column_summary function called")
    summary_data = []
    
    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_of_nulls = df[col_name].isnull().sum()
        num_of_non_nulls = df[col_name].notnull().sum()
        num_of_distinct_values = df[col_name].nunique()
        
        if num_of_distinct_values <= 10:
            distinct_values_counts = df[col_name].value_counts().to_dict()
        else:
            top_10_values_counts = df[col_name].value_counts().head(10).to_dict()
            distinct_values_counts = {k: v for k, v in sorted(top_10_values_counts.items(), key=lambda item: item[1], reverse=True)}

        summary_data.append({
            'col_name': col_name,
            'col_dtype': col_dtype,
            'num_of_nulls': num_of_nulls,
            'num_of_non_nulls': num_of_non_nulls,
            'num_of_distinct_values': num_of_distinct_values,
            'distinct_values_counts': distinct_values_counts
        })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info("Column summary generated")
    return summary_df

def detect_outliers(df, numerical_columns):
    logger.info("detect_outliers function called")
    outliers_dict = {}
    
    for column in numerical_columns:
        if column in df.columns:
            # Z-score method
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outliers_z = np.where(z_scores > 3)[0]
            outliers_dict[column] = outliers_z.tolist()
            logger.info(f"Outliers detected in column {column} using Z-score method")
            
            # Box plot method
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[column])
            plt.title(f'Box Plot for {column}')
            plt.show()
            logger.info(f"Box plot generated for column {column}")
            
    return outliers_dict

def fix_outlier(df, col):
    logger.info(f"fix_outlier function called for column {col}")
    df[col] = np.where(df[col] > df[col].quantile(0.95), df[col].quantile(0.95), df[col])
    df[col] = np.where(df[col] < df[col].quantile(0.05), df[col].quantile(0.05), df[col])
    logger.info(f"Outliers fixed for column {col}")
    return df

def remove_outliers(df, column_to_process, z_score_threshold):
    logger.info(f"remove_outliers function called for column {column_to_process} with threshold {z_score_threshold}")
    z_scores = zscore(df[column_to_process])
    outlier_column = column_to_process + '_outlier'
    df[outlier_column] = (np.abs(z_scores) > z_score_threshold).astype(int)
    df = df[df[outlier_column] == 0]
    df = df.drop(columns=[outlier_column], errors='ignore')
    logger.info(f"Outliers removed for column {column_to_process}")
    return df



def plot_categorical_column(df, column):
    logger.info(f"plot_categorical_column function called for column {column}")
    if column in df.columns and df[column].dtype == 'object':
        plt.figure(figsize=(10, 6))
        sns.countplot(y=column, data=df, order=df[column].value_counts().index)
        plt.title(f'Frequency of each category in {column}')
        plt.xlabel('Frequency')
        plt.ylabel(column)
        plt.show()
        logger.info(f"Frequency plot generated for column {column}")
    else:
        logger.error(f"Column {column} is not in the DataFrame or not a categorical column")