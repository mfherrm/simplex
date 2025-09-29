# Control imports
import os
import pandas as pd
import numpy as np

# MLFlow imports
import mlflow

# disy Cadenza imports
import cadenzaanalytics as ca

def get_models_and_runs(metadata: ca.RequestMetadata, data):
    """
    Send a token and an experiment name and get models and runs, as well as their ID in a dataframe, e.g.

    | DF_ID | Run ID | Run Name |
    |   1   | a8d9v7 | Test-Run |
    """
    attribute_groups = metadata.get_columns_by_attribute_group()

    # Extract token
    token_group = attribute_groups["token"]
    token_col = [c.name for c in token_group]
    token = data[token_col]

    # Extract experiment ID
    experiment = [str(metadata.get_parameter("experiment"))]

    # Set token as environment variable to pass the authenticator
    os.environ["MLFLOW_TRACKING_TOKEN"] = token.values[0][0]

    # Get all runs for the provided experiment ID
    list_of_runs = mlflow.search_runs(experiment)

    # Extract run IDs and names
    run_ids, run_names = [list_of_runs.iloc[i]["run_id"] for i in np.arange(0,len(list_of_runs),1)], [list_of_runs.iloc[i]["tags.mlflow.runName"] for i in np.arange(0,len(list_of_runs),1)]
    
    # Build dataframe
    run_frame = pd.DataFrame({"run_id": run_ids, "run_name": run_names})

    print(run_frame)

    # Build metadata
    run_metadata = [
        ca.ColumnMetadata(
            name="run_id",
            print_name="Run ID",
            data_type=ca.DataType.STRING,
            attribute_group_name='runs',
            role=ca.AttributeRole.DIMENSION),
        ca.ColumnMetadata(
            name="run_name",
            print_name="Run Name",
            data_type=ca.DataType.STRING,
            attribute_group_name='runs',
            role=ca.AttributeRole.DIMENSION)
    ]

    print(run_metadata)

    return ca.CsvResponse(run_frame, run_metadata)
