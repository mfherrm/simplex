import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Control imports
import requests
import json
import time
import orjson
from flask import jsonify
from dotenv import load_dotenv

# Clustering imports
from dask_ml.preprocessing import StandardScaler
from dask_ml.cluster import KMeans
from dask.distributed import wait
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.cluster import MiniBatchKMeans as SklearnKMeans

# Data processing imports
import pandas as pd
import dask as dd
from dask.distributed import Client
from dask.distributed import LocalCluster
import numpy as np

# MLFlow imports
import mlflow

# disy Cadenza imports
import cadenzaanalytics as ca
from AE.url_response import UrlResponse

WEBSERVICE_HOST = os.getenv('VISUALISATION_HOST', 'http://127.0.0.1:5000')
URL_PART = f"{WEBSERVICE_HOST}"

def get_experiments(metadata: ca.RequestMetadata, data):
    """
    Send a token and get experiments in a dataframe, e.g.

    Experiment_ID | Experiment_Name
          1       | Test_Experiment
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

# Token
token = ca.AttributeGroup(
    name="token",
    print_name="MLFlow Access Token",
    data_types=[ca.DataType.STRING],
    min_attributes=1,
    max_attributes = 1
)

experiments_extension = ca.CadenzaAnalyticsExtension(
    relative_path="experiments-extension", 
    analytics_function= get_experiments, 
    print_name="MLFlow Experiments Extension",
    extension_type=ca.ExtensionType.CALCULATION,
    attribute_groups=[token]
)

###

###

def get_models_and_runs(metadata: ca.RequestMetadata, data):
    """
    Send a token and an experiment name and get models and runs, as well as their ID in a dataframe, e.g.

    DF_ID | Experiment 
    1     |
    """
    attribute_groups = metadata.get_columns_by_attribute_group()

    token_group = attribute_groups["token"]
    token_col = [c.name for c in token_group]
    token = data[token_col]

    experiment = [str(metadata.get_parameter("experiment"))]


    os.environ["MLFLOW_TRACKING_TOKEN"] = token.values[0][0]

    list_of_runs = mlflow.search_runs(experiment)


    run_ids, run_names = [list_of_runs.iloc[i]["run_id"] for i in np.arange(0,len(list_of_runs),1)], [list_of_runs.iloc[i]["tags.mlflow.runName"] for i in np.arange(0,len(list_of_runs),1)]
    run_frame = pd.DataFrame({"run_id": run_ids, "run_name": run_names})

    print(run_frame)

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

# Token
token = ca.AttributeGroup(
    name="token",
    print_name="MLFlow Access Token",
    data_types=[ca.DataType.STRING],
    min_attributes=1,
    max_attributes = 1

)

# Experiment
experiment = ca.Parameter(
    name="experiment",
    print_name="Experiment ID",
    parameter_type=ca.DataType.INT64,
    required=True
    
)

model_and_runs_extension = ca.CadenzaAnalyticsExtension(
    relative_path="model-and-runs-extension", 
    analytics_function= get_models_and_runs, 
    print_name="MLFlow models and runs extension",
    extension_type=ca.ExtensionType.CALCULATION,
    attribute_groups=[token],
    parameters=[experiment]
)

###

###

def get_samples(df, random_state = 17):
    # Check for dask or pandas
    is_pandas = isinstance(df, pd.DataFrame)
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not is_pandas:
        numeric_df = numeric_df.astype("float64")
    numeric_df = numeric_df.dropna()
    
    # Get dimensions
    if is_pandas:
        n_rows = len(numeric_df)
    else:
        n_rows = numeric_df.shape[0].compute()
    n_cols = len(numeric_df.columns)
    
    # Handle empty dataframe
    if n_rows == 0 or n_cols == 0:
        print("Got empty dataframe after filtering")
        return df.head(0)
    

    return df.sample(frac=0.15, random_state=random_state)


def get_column_names(attribute_groups: dict, group:str):
    """
    Returns column names and column print names for specified group.
    """
    try:
        columns = attribute_groups[group]
        print(c.print_name for c in columns)
        return [c.name for c in columns], [c.print_name for c in columns]
    except:
        print("Column type not given")
        return [], []

def process_data(ref_data, new_data, new_data_cols, new_data_print_cols, common_columns, npartitions=4):
    # Initialize local dask cluster
    with LocalCluster(processes=False, n_workers=1, threads_per_worker=4, memory_limit='4GB') as cluster:
        with Client(cluster) as client:
            print(f"Dask dashboard available at: {client.dashboard_link}")
            
            # Drop nas
            ref_data.dropna(inplace=True)
            new_data.dropna(inplace=True)

            print("New data before:", new_data)

            # Process ref data
            ref_data = ref_data[common_columns]
            print("ref-data passed")
            
            # Process new data
            rename_dict_new = dict(zip(new_data_cols, new_data_print_cols))
            print("dict passed", rename_dict_new)
            short_dict = {key:value for key, value in rename_dict_new.items() if value in common_columns}

            print("short-dict passed")
            new_data = new_data[short_dict.keys()].rename(columns=short_dict)

            print("New data after:", new_data)
            
            # Re-order new data
            print("Old column order: ", new_data)
            desired_order = ref_data.columns.tolist()
            new_data = new_data[desired_order]

            print("New column order: ", new_data)

            print("New data processed")

            # Sample datasets
            frac = ref_data.shape[0] / new_data.shape[0] 

            if frac < 1: 
                new_data = new_data.sample(frac=frac)
            else:
                ref_data= ref_data.sample(frac=1/frac)

            new_data = get_samples(new_data)
            ref_data = get_samples(ref_data)

    
    return new_data, ref_data


def calculate_data_drift(metadata: ca.RequestMetadata, data):
    """
    Sends a dataframe and a Run ID to the Flask service to calculate data drift
    and returns a URL to the drift report.
    """

    t0 = time.time()
    attribute_groups = metadata.get_columns_by_attribute_group()
    print(attribute_groups)
    
    # Get newdata ID column name(s)
    newdata_id_column_name, newdata_id_column_print_name = get_column_names(attribute_groups, "newdata_id")

    t1 = time.time()
    print("Got new data:", t1-t0)

    # Get newdata column names
    newdata_column_names, newdata_column_print_names = get_column_names(attribute_groups, "newdata")

    # Get newdata datetime column names
    newdata_datetime_column_names, newdata_datetime_column_print_names = get_column_names(attribute_groups, "newdata_date")

    t2 = time.time()
    print("Got new data column names:", t2-t1)

    # Get Training data
    
    # columns = attribute_groups["run_id"]
    # id_col = [c.name for c in columns]
    # run_id = data[id_col]

    logged_run = mlflow.get_run(os.getenv('RUN_ID'))

    # Get artifact path via the dataset source logged with the dataset
    artifact_path = os.path.join(logged_run.info.artifact_uri, logged_run.inputs.dataset_inputs[0].dataset.source.split("artifacts/")[-1]).replace("\\","/")[:-2]

    training_data = mlflow.artifacts.download_artifacts(artifact_path, dst_path = "./")

    ref_data = pd.read_parquet(training_data)

    os.remove(training_data) 
    # This code is nicer:
    # dataset_source = mlflow.data.get_source(logged_dataset)
    # local_dataset = dataset_source.load()
    # But it currently does not work due to authentication problems (downloads login page as HTML file)
    # If the method above returns a not implemented error: dataset was commited without source before (https://github.com/mlflow/mlflow/issues/13015)
    # -> Need to slightly change dataset to change digest

    
    # Get Training data column names
    ref_data_print_cols = ref_data.columns

    # Get Arrays to select and rename the data
    new_data_cols = newdata_id_column_name + newdata_column_names + newdata_datetime_column_names
    new_data_print_cols = newdata_id_column_print_name + newdata_column_print_names + newdata_datetime_column_print_names

    common_columns = list(set(ref_data_print_cols).intersection(new_data_print_cols))

    print(newdata_column_print_names)

    print(ref_data_print_cols)

    print("\n", new_data_print_cols)

    # Fetch last report if there is an uneven number of columns
    if len(common_columns)<1: #(len(new_data_print_cols) != len(ref_data_print_cols)):
        if analytics_service.last_url != None:
            return UrlResponse(analytics_service.last_url)
        else:
            return ca.ErrorResponse(f"Error: No common columns between new data and reference data")
    
    try:
        #(ref_data, new_data, new_data_cols, new_data_print_cols, ref_data_print_cols, npartitions=4)
        new_data, ref_data = process_data(ref_data, data, new_data_cols, new_data_print_cols, common_columns, npartitions=4)
        
        t3 = time.time()
        print("Processed data:", t3-t2)

        print("new_data: ", new_data.shape)
        print("ref_data: ", ref_data.shape)

        # Prepare the data for the POST request
        payload = {
            "reference_data": {
                "data": ref_data.to_dict(orient='split'),
                "id_column": [],
                # "numerical_columns": refdata_column_names,
                "datetime_columns": []
            },
            "current_data": {
                "data": new_data.to_dict(orient='split'),
                "id_column": newdata_id_column_print_name,
                # "numerical_columns": newdata_column_names,
                "datetime_columns": newdata_datetime_column_print_names
            }
        }
    except Exception as e:
        print(f"Error processing data: {e}")
        return ca.ErrorResponse(f"Error processing data: {e}")
    
    headers = {'Content-Type': 'application/json'}

    t4 = time.time()
    print("Constructed payload:", t4-t3)

    # Send the request to the data drift calculation endpoint

    try:
        response = requests.post(f"{URL_PART}/app/data_drift", data=orjson.dumps(payload, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY), headers=headers)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        print(f"Error calling data drift service: {e}")
        return ca.ErrorResponse(f"Error generating report: {e}")
    
    t5 = time.time()
    print("Generated report:", t5-t4)

    response_data = response.json()
    # Return the URL to view the generated report
    report_url = response_data.get("report_url")
    print(report_url.split(")"))

    # Return the URL to view the generated report
    report_url = f"{URL_PART}"+report_url.split(")")[1]
    analytics_service.last_url = report_url
    return UrlResponse(report_url)

# New dataset ID column
new_attribute_group_id = ca.AttributeGroup(
    name="newdata_id",
    print_name="New Data ID Column",
    data_types=[ca.DataType.STRING, ca.DataType.INT64],
    min_attributes=0,
    max_attributes = 1
)

# New dataset 
new_attribute_group = ca.AttributeGroup(
    name="newdata",
    print_name="New Data",
    data_types=[ca.DataType.STRING, ca.DataType.INT64, ca.DataType.FLOAT64],
    min_attributes=1,
    max_attributes = None
)

# New dataset datetime column(s)
new_attribute_group_date = ca.AttributeGroup(
    name="newdata_date",
    print_name="New Data Datetime Column(s)",
    data_types=[ca.DataType.ZONEDDATETIME],
)

# Run id
# run_id = ca.AttributeGroup(
#     name="run_id",
#     print_name="MLFlow Run ID",
#     data_types=[ca.DataType.STRING],
# )

model_data_drift_extension = ca.CadenzaAnalyticsExtension(
    relative_path="model-data-drift-extension-rs", 
    analytics_function= calculate_data_drift, 
    print_name="Model Data Drift Extension Random Sampling",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, new_attribute_group_id, new_attribute_group_date]#, run_id]
)

analytics_service = ca.CadenzaAnalyticsExtensionService()
analytics_service.add_analytics_extension(experiments_extension)
analytics_service.add_analytics_extension(model_and_runs_extension)
analytics_service.add_analytics_extension(model_data_drift_extension)
MEGABYTE = (2 ** 10) ** 2
analytics_service._app.config['MAX_CONTENT_LENGTH'] = os.getenv('MAX_CONTENT_LENGTH')
analytics_service._app.config['MAX_FORM_PARTS'] = os.getenv('MAX_FORM_PARTS')
analytics_service._app.config['MAX_FORM_MEMORY_SIZE'] = os.getenv('MAX_FORM_MEMORY_SIZE')
analytics_service.last_url = None

if __name__ == '__main__':
    load_dotenv()
    analytics_service.run_development_server(5005)