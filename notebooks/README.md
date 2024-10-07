# Notebooks

This folder contains Jupyter notebooks related to the development and analysis of Bati Bank’s **Credit Scoring Model** for its **Buy-Now-Pay-Later (BNPL)** service, using eCommerce platform data. These notebooks focus on data exploration, feature engineering, model development, and evaluation processes.

## Folder Structure

The notebooks are organized as follows:

1. **01_Data_Loading_and_Cleaning.ipynb**
   - Purpose: Loads the raw data and performs initial cleaning steps such as removing duplicates, handling missing values, and ensuring data types are correct.
   - Key Steps:
     - Data import and basic validation.
     - Missing values treatment and duplicates removal.
     - Data type conversions for numerical and categorical features.
   - Output: Cleaned dataset ready for Exploratory Data Analysis (EDA).

2. **02_Exploratory_Data_Analysis.ipynb**
   - Purpose: Conducts **Exploratory Data Analysis (EDA)** to gain insights into the dataset and understand variable distributions, correlations, and patterns.
   - Key Analyses:
     - Descriptive statistics and summary metrics.
     - Univariate analysis for both categorical and numerical variables.
     - Bivariate analysis and correlation matrix.
     - Visualizations: Histograms, boxplots, count plots, and time series plots.
   - Output: Visual and statistical insights to inform feature engineering.

3. **03_Feature_Engineering.ipynb**
   - Purpose: Engineers new features from existing data, including transformations, binning, and encoding categorical variables.
   - Key Steps:
     - Handling categorical variables (e.g., encoding).
     - Creating new features like RFMS scores (Recency, Frequency, Monetary, Stability).
     - Weight of Evidence (WoE) and Information Value (IV) calculations for feature selection.
   - Output: Feature-enhanced dataset for modeling.

4. **04_Model_Development.ipynb**
   - Purpose: Develops and fine-tunes credit scoring models to predict default risk. It also includes model evaluation metrics to assess performance.
   - Key Models:
     - Logistic Regression.
     - Decision Trees.
     - Random Forests.
     - XGBoost.
   - Output: Trained models with performance metrics (e.g., AUC, accuracy, recall).

5. **05_Model_Evaluation_and_Interpretability.ipynb**
   - Purpose: Evaluates the performance of the models using various metrics and applies interpretability techniques like SHAP values to understand feature importance.
   - Key Steps:
     - Confusion matrix, AUC-ROC curves, precision-recall metrics.
     - SHAP and LIME for model explainability.
   - Output: Final evaluation results and model interpretation.

6. **06_Prediction_and_Deployment.ipynb**
   - Purpose: Prepares the model for deployment, including saving the trained model, testing with new data, and exporting results.
   - Key Steps:
     - Saving the model using `joblib` or `pickle`.
     - Predicting credit risk on unseen data.
     - Exporting predictions and model pipeline for production.
   - Output: Serialized model and prediction outputs.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/AmanReshid/Bati-Bank-Credit-Scoring-Model.git
