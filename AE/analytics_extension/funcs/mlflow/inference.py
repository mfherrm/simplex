# Control imports
import os
import pandas as pd
import numpy as np

# MLFlow imports
import mlflow
import pickle

# disy Cadenza imports
import cadenzaanalytics as ca

def get_row(df, row):
    """
    Maps rows for the RowWiseCSVResponse
    """
    response_row = []
    # Ensure identical types 
    for idx, val in df.iloc[row.name].items():
        if idx in df.select_dtypes("Int64"):
            response_row.append(val.astype(int))
        else:
            response_row.append(val)
    print(response_row)
    return response_row

# -----------------------------------------------------------------------------------------------------
# As AE of type enrichment
# -----------------------------------------------------------------------------------------------------

def get_predictions_enr(metadata: ca.RequestMetadata, data):
    """
    Send input data and a run_id and get predictions, e.g.

    | Input_01 | Input_02 | Input_03 | Prediction |
    |    1     |    0.5   |   0.25   |     17     |
    """

    attribute_groups = metadata.get_columns_by_attribute_group()

    # Get input data from Cadenza
    data_group = attribute_groups["input_data"]
    input_cols = [c.name for c in data_group]
    input_data = data[input_cols]

    # Get token and set it as an environment variable
    token= str(metadata.get_parameter("token"))
    os.environ["MLFLOW_TRACKING_TOKEN"] = token

    # Use the run ID provided to get the components of the model path 
    run_id = str(metadata.get_parameter("run_id"))
    print(run_id)
    logged_run = mlflow.get_run(run_id)
    model_id = logged_run.outputs.to_dictionary()["model_outputs"][0].model_id
    artifact_path = logged_run.info.artifact_uri.split("/"+logged_run.info.run_id+"/")
    
    # Construct the artifact ID to get the model
    model_path = os.path.join(artifact_path[0], "models", model_id, artifact_path[1], "model.pkl").replace("\\","/")

    # Download the model and instantiate it
    model = mlflow.artifacts.download_artifacts(model_path, dst_path = "./")
    loaded_model = pickle.load(open(model, 'rb'))

    # Get predictions
    predictions = loaded_model.predict(input_data)

    # Delete model
    os.remove(model) 

    # prediction_frame = input_data

    prediction_frame = pd.DataFrame({'ID':np.arange(0, len(predictions)), 'predictions':predictions})
    # prediction_frame["predictions"] = predictions
    # prediction_frame.insert(0, 'ID', prediction_frame.index)
    
    print(prediction_frame)

    # input_metadata = metadata.get_columns_by_attribute_group()['input_data']

    id_metadata = metadata.get_columns_by_attribute_group()['net.disy.cadenza.keyAttributeGroup']
    print(id_metadata)

    prediction_metadata = [ca.ColumnMetadata(
            name="predictions",
            print_name="Predictions",
            data_type=ca.DataType.FLOAT64,
            attribute_group_name='Predictions',
            role=ca.AttributeRole.MEASURE)]
    
    response_metadata = id_metadata + prediction_metadata #input_metadata + prediction_metadata

    return ca.CsvResponse(prediction_frame, response_metadata)

    # return ca.RowWiseMappingCsvResponse(response_metadata, lambda row: get_row(prediction_frame, row))


# -----------------------------------------------------------------------------------------------------
# As AE of type calculation
# -----------------------------------------------------------------------------------------------------

def get_predictions_cal(metadata: ca.RequestMetadata, data):
    """
    Send input data and a run_id and get predictions, e.g.

    | Input_01 | Input_02 | Input_03 | Prediction |
    |    1     |    0.5   |   0.25   |     17     |
    """

    attribute_groups = metadata.get_columns_by_attribute_group()

    # Get input data from Cadenza
    data_group = attribute_groups["input_data"]
    input_cols = [c.name for c in data_group]
    input_data = data[input_cols]

    # Get token and set it as an environment variable
    token= str(metadata.get_parameter("token"))
    os.environ["MLFLOW_TRACKING_TOKEN"] = token

    # Use the run ID provided to get the components of the model path 
    run_id = str(metadata.get_parameter("run_id"))
    print(run_id)
    logged_run = mlflow.get_run(run_id)
    model_id = logged_run.outputs.to_dictionary()["model_outputs"][0].model_id
    artifact_path = logged_run.info.artifact_uri.split("/"+logged_run.info.run_id+"/")
    
    # Construct the artifact ID to get the model
    model_path = os.path.join(artifact_path[0], "models", model_id, artifact_path[1], "model.pkl").replace("\\","/")

    # Download the model and instantiate it
    model = mlflow.artifacts.download_artifacts(model_path, dst_path = "./")
    loaded_model = pickle.load(open(model, 'rb'))

    # Get predictions
    predictions = loaded_model.predict(input_data)

    # Delete model
    os.remove(model) 

    prediction_frame = input_data
    prediction_frame["predictions"] = predictions

    print(prediction_frame)

    input_metadata = metadata.get_columns_by_attribute_group()['input_data']

    prediction_metadata = [ca.ColumnMetadata(
            name="predictions",
            print_name="Predictions",
            data_type=ca.DataType.FLOAT64,
            attribute_group_name='Predictions',
            role=ca.AttributeRole.MEASURE)]

    response_metadata = input_metadata + prediction_metadata

    return ca.CsvResponse(prediction_frame, response_metadata)
