# Data management imports
import pandas as pd
import numpy as np

# Evidently imports
from evidently import DataDefinition

def map_to_def(dataFrame: pd.DataFrame, id_col: str, timestamp_col: str = None):
    """
    Processes a DataFrame to identify column types and prepares a DataDefinition for Evidently.
    """
    df = dataFrame.copy()
    df.columns = df.columns.astype(str)

    datetime_cols = []
    
    # Explicitly handle the designated timestamp column
    if timestamp_col and timestamp_col in df.columns:
        print(f"Column '{timestamp_col}' designated as timestamp. Converting.")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        datetime_cols.append(timestamp_col)

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=np.number).columns.to_list()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()

    # Remove ID and timestamp from lists as they are special columns
    if id_col in numerical_cols:
        numerical_cols.remove(id_col)
    if timestamp_col and timestamp_col in numerical_cols:
         numerical_cols.remove(timestamp_col)
         
    if id_col in categorical_cols:
        categorical_cols.remove(id_col)
    if timestamp_col and timestamp_col in categorical_cols:
        categorical_cols.remove(timestamp_col)

    # Auto-detect other potential datetime columns from remaining categorical columns
    remaining_cat = categorical_cols[:]
    for col in remaining_cat:
        # Skip the already processed timestamp column
        if col == timestamp_col:
            continue
        try:
            # Attempt to convert a sample to see if it's a datetime column
            pd.to_datetime(df[col].dropna().iloc[:10], errors='raise')
            print(f"Column '{col}' auto-detected as datetime. Converting entire column.")
            df[col] = pd.to_datetime(df[col], errors='coerce')
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except (ValueError, TypeError):
            # Not a datetime column, so we pass
            pass
    
    print("\n--- Column Classification ---")
    print(f"ID Column: {id_col}")
    print(f"Timestamp Column: {timestamp_col}")
    print(f"Numerical Columns: {numerical_cols}")
    print(f"Categorical Columns: {categorical_cols}")
    print(f"Datetime Columns: {datetime_cols}")
    print("---------------------------\n")

    return df, DataDefinition(
        id_column=id_col,
        timestamp=timestamp_col,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols
    )