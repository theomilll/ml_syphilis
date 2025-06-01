# preprocessing.py
# Defines functions for data cleaning, feature engineering, and preprocessing pipelines.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Seed for reproducibility as per project guidelines
import random, os
random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# --- Column Definitions (Based on EDA and Feature Engineering) ---
TARGET_COL = 'congenital_syphilis'

# Final numeric columns after EDA
NUM_COLS = [
    'AGE', 'NUM_ABORTIONS', 'NUM_LIV_CHILDREN', 'NUM_PREGNANCIES', 'NUM_RES_HOUSEHOLD'
]

# Final categorical columns after EDA and feature engineering
CAT_COLS = [
    'CONS_ALCOHOL', 'RH_FACTOR', 'SMOKER', 'PLAN_PREGNANCY', 'BLOOD_GROUP',
    'HAS_PREG_RISK', 'TET_VACCINE', 'IS_HEAD_FAMILY', 'MARITAL_STATUS',
    'FOOD_INSECURITY', 'FAM_PLANNING', 'TYPE_HOUSE', 'HAS_FAM_INCOME',
    'LEVEL_SCHOOLING', 'CONN_SEWER_NET', 'HAS_FRU_TREE', 'HAS_VEG_GARDEN',
    'FAM_INCOME', 'HOUSING_STATUS', 'WATER_TREATMENT', 'IS_AGE_IMPUTED',
    'IS_ADOLESCENT', 'PREG_RISK_INTERACTION'
]

def clean_and_engineer_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list, list, str]:
    """
    Cleans the raw data and engineers new features based on the notebook steps.
    Args:
        df_raw: Raw pandas DataFrame.
    Returns:
        df_processed: Cleaned and engineered pandas DataFrame.
        final_num_cols: List of numeric column names.
        final_cat_cols: List of categorical column names.
        target_col_name: Name of the target column.
    """
    df = df_raw.copy()

    # --- 1. Initial Setup & Target Variable Handling ---
    # Rename original target column
    if 'VDRL_RESULT' in df.columns:
        df.rename(columns={'VDRL_RESULT': TARGET_COL}, inplace=True)
    elif TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' or 'VDRL_RESULT' not found in DataFrame.")

    # Correct target variable encoding (0.0 -> 1 (Positive), 1.0 -> 0 (Negative))
    # Ensure this mapping is correct based on EDA findings (1.98% positive cases)
    df[TARGET_COL] = df[TARGET_COL].map({0.0: 1, 1.0: 0}).astype(int)

    # --- 2. Data Cleaning ---
    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Handle impossible ages (<10 or >60)
    df['IS_AGE_IMPUTED'] = 0
    invalid_age_condition = (df['AGE'] < 10) | (df['AGE'] > 60)
    df.loc[invalid_age_condition, 'IS_AGE_IMPUTED'] = 1
    
    valid_ages = df.loc[(df['AGE'] >= 10) & (df['AGE'] <= 60), 'AGE']
    if not valid_ages.empty:
        median_valid_age = valid_ages.median()
        df.loc[invalid_age_condition, 'AGE'] = median_valid_age
    else: # Handle case where all ages might be invalid, though unlikely
        df.loc[invalid_age_condition, 'AGE'] = df['AGE'].median() # Fallback

    # --- 3. Feature Engineering ---
    # Create IS_ADOLESCENT flag (age < 20)
    df['IS_ADOLESCENT'] = (df['AGE'] < 20).astype(int)

    # Create PREG_RISK_INTERACTION (HAS_PREG_RISK * IS_ADOLESCENT)
    df['PREG_RISK_INTERACTION'] = df['HAS_PREG_RISK'] * df['IS_ADOLESCENT']
    
    # The global NUM_COLS and CAT_COLS already reflect these engineered features.
    return df, NUM_COLS, CAT_COLS, TARGET_COL

# --- Preprocessing Pipelines (as per section 3.4 of roadmap) ---

# Numeric features pipeline: Median Imputation + Scaling
num_pipe = Pipeline([
    ('imputer_num', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features pipeline: Mode Imputation (or constant for 'unknown') + OneHotEncoding
cat_pipe = Pipeline([
    ('imputer_cat', SimpleImputer(strategy='constant', fill_value=-1)), # Using -1 for unknown/missing
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create the full preprocessing ColumnTransformer
# This transformer will apply the specified transformations to the respective columns.
preprocess_transformer = ColumnTransformer([
    ('num', num_pipe, NUM_COLS),
    ('cat', cat_pipe, CAT_COLS)
], remainder='passthrough') # Keep other columns (like target) if not dropped

print(f"preprocessing.py loaded. Target: '{TARGET_COL}', {len(NUM_COLS)} numeric features, {len(CAT_COLS)} categorical features.")
