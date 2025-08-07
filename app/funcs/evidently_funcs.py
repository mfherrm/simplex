# Data management imports
import pandas as pd
import numpy as np

# Evidently imports
from evidently import DataDefinition

def map_to_def(dataFrame: pd.DataFrame, id_col: list = [], datetime_cols: list = []):
    """
    Processes a DataFrame to identify column types and prepares a DataDefinition for Evidently.
    """
    df = dataFrame.copy()
    df.columns = df.columns.astype(str)

    numerical_cols = df.select_dtypes(include=np.number).columns.to_list()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()

    id_column_name = id_col[0] if id_col and len(id_col) > 0 else None

    # Ensure special columns are not in the feature lists
    if id_column_name:
        if id_column_name in numerical_cols:
            numerical_cols.remove(id_column_name)
        if id_column_name in categorical_cols:
            categorical_cols.remove(id_column_name)

    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns:
                print(f"Converting {col} to datetime")
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if col in numerical_cols:
                    numerical_cols.remove(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)

    return df, DataDefinition(
        id_column=id_column_name,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
    )