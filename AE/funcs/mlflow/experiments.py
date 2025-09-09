# Control imports
import os
import pandas as pd

# MLFlow imports
import mlflow

# disy Cadenza imports
import cadenzaanalytics as ca

def get_experiments(metadata: ca.RequestMetadata, data):
    """
    Send a token and get experiments in a dataframe, e.g.

    Experiment_ID | Experiment_Name |
          1       | Test_Experiment |
    """

    attribute_groups = metadata.get_columns_by_attribute_group()

    token_group = attribute_groups["token"]
    token_col = [c.name for c in token_group]
    token = data[token_col]

    os.environ["MLFLOW_TRACKING_TOKEN"] = token.values[0][0]

    experiments = mlflow.search_experiments()

    experiment_ids, experiment_names = [experiment.experiment_id for experiment in experiments], [experiment.name for experiment in experiments]

    experiment_frame = pd.DataFrame({"experiment_id": experiment_ids, "experiment_name": experiment_names})

    print(experiment_frame)

    experiment_metadata = [
        ca.ColumnMetadata(
            name="experiment_id",
            print_name="Experiment ID",
            data_type=ca.DataType.INT64,
            attribute_group_name='Experiments',
            role=ca.AttributeRole.DIMENSION),
        ca.ColumnMetadata(
            name="experiment_name",
            print_name="Experiment Name",
            data_type=ca.DataType.STRING,
            attribute_group_name='Experiments',
            role=ca.AttributeRole.DIMENSION)
    ]

    print(experiment_metadata)

    return ca.CsvResponse(experiment_frame, experiment_metadata)
