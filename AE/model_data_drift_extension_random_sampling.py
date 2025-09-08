import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Control imports
import requests
import json
import time
import orjson
from flask import jsonify

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
os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.simplex4learning.de"
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["RUN_ID"] = "3cdd0c9d090c444fa8be1a4b7ac39882"
# os.environ["MLFLOW_TRACKING_TOKEN"] = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJqNUpOUlhIX0JWZUZQS1lRanFUZ3lyWi1rYVJoUmJybGtPOWlmZjdDSDJvIn0.eyJleHAiOjE3NTcxMTAxODAsImlhdCI6MTc1NzA3NDE4MCwianRpIjoiMzU2ZmFhZjItMWJhYi00OWY3LTk4YzAtMWI1MzY1ZTg3Mzg5IiwiaXNzIjoiaHR0cHM6Ly9hdXRoLnNpbXBsZXg0bGVhcm5pbmcuZGUvcmVhbG1zL3NpbXBsZXg0bGVhcm5pbmciLCJhdWQiOlsibWxmbG93IiwiYWNjb3VudCJdLCJzdWIiOiJjN2U2YTU2NC0xZDAyLTQxZGEtOWRlOC0xNTA2ZTc0YzM5YjUiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJtbGZsb3ctYXBpIiwic2Vzc2lvbl9zdGF0ZSI6IjJiZTMyYWQ5LTk1ZmUtNGYyOC1iMDEyLWVkODQ4YzdmMTllYyIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiLyoiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNpbXBsZXg0bGVhcm5pbmciXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIiwic2lkIjoiMmJlMzJhZDktOTVmZS00ZjI4LWIwMTItZWQ4NDhjN2YxOWVjIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJNYXJpdXMgSGVycm1hbm4iLCJncm91cHMiOlsiQWRtaW5pc3RyYXRvciIsIkFuYWx5c3QiLCJDcmVhdG9yIiwiVmlld2VyIl0sInByZWZlcnJlZF91c2VybmFtZSI6Im1hcml1cyIsImdpdmVuX25hbWUiOiJNYXJpdXMiLCJmYW1pbHlfbmFtZSI6IkhlcnJtYW5uIiwiZW1haWwiOiJtYXJpdXMuaGVycm1hbm5AZGlzeS5uZXQifQ.JuALFZVSl4QXtdVZeMY7yGfnn8TnUOJNYq1gkLVxV2amGAbHOQqsachPoCsrAYfzL4zMEz0xkpqAlrGcY3s6f3iSXkv5kHEL6HiBoul6S1T9U8jjo89kSb6c3CJUw69a49zor7ZBHcnpahZ06b_VLKdrxRvzrGHO7rlSu2R95xr-mczbtyUONiLIM_t1AglYxa-iuGjKD8lH6uv9ut3ZS0Pmoj2HXRzv-_9bvvuRNlSUMXn5W_5jFrWs32nGJtlttNIEsUIw2UcBmKlga1Eb3moxP6s8PWHr2eM4LlAXOd3DkxWLcottsbQf5lceC2b6xH_wvg92yTqG-_uMrQCFkA"

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

def process_data(data, new_data_cols, new_data_print_cols, ref_data_cols, ref_data_print_cols, npartitions=4):
    # Initializing local dask cluster
    with LocalCluster(processes=False, n_workers=1, threads_per_worker=4, memory_limit='4GB') as cluster:
        with Client(cluster) as client:
            print(f"Dask dashboard available at: {client.dashboard_link}")
            
            # Handle data preprocessing with pandas directly to avoid large graph warnings
            data.dropna(inplace=True)
            
            # Process new data with pandas directly
            rename_dict_new = dict(zip(new_data_cols, new_data_print_cols))
            new_data = data[new_data_cols].rename(columns=rename_dict_new)
            print("New data processed")

            # Process reference data with pandas directly
            rename_dict_ref = dict(zip(ref_data_cols, ref_data_print_cols))
            ref_data = data[ref_data_cols].rename(columns=rename_dict_ref)

            new_data = get_samples(new_data)
            ref_data = get_samples(ref_data)

            print("Reference data processed")

    
    return new_data, ref_data


def calculate_data_drift(metadata: ca.RequestMetadata, data):
    """
    Sends two dataframes to the Flask service to calculate data drift
    and returns a URL to the drift report.
    """
    
    t0 = time.time()
    attribute_groups = metadata.get_columns_by_attribute_group()
    print(attribute_groups)
    
    # Get newdata ID column name(s)
    newdata_id_column_name, newdata_id_column_print_name = get_column_names(attribute_groups, "newdata_id")

    t1 = time.time()
    print("t1:", t1-t0)

    # Get newdata column names
    newdata_column_names, newdata_column_print_names = get_column_names(attribute_groups, "newdata")

    # Get newdata datetime column names
    newdata_datetime_column_names, newdata_datetime_column_print_names = get_column_names(attribute_groups, "newdata_date")

    t2 = time.time()
    print("t2:", t2-t1)

    # Get Training data

      # Get Training data
    
    columns = attribute_groups["run_id"]
    id_col = [c.name for c in columns]
    run_id = data[id_col]

    token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJqNUpOUlhIX0JWZUZQS1lRanFUZ3lyWi1rYVJoUmJybGtPOWlmZjdDSDJvIn0.eyJleHAiOjE3NTcxMTAxODAsImlhdCI6MTc1NzA3NDE4MCwianRpIjoiMzU2ZmFhZjItMWJhYi00OWY3LTk4YzAtMWI1MzY1ZTg3Mzg5IiwiaXNzIjoiaHR0cHM6Ly9hdXRoLnNpbXBsZXg0bGVhcm5pbmcuZGUvcmVhbG1zL3NpbXBsZXg0bGVhcm5pbmciLCJhdWQiOlsibWxmbG93IiwiYWNjb3VudCJdLCJzdWIiOiJjN2U2YTU2NC0xZDAyLTQxZGEtOWRlOC0xNTA2ZTc0YzM5YjUiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJtbGZsb3ctYXBpIiwic2Vzc2lvbl9zdGF0ZSI6IjJiZTMyYWQ5LTk1ZmUtNGYyOC1iMDEyLWVkODQ4YzdmMTllYyIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiLyoiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLXNpbXBsZXg0bGVhcm5pbmciXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIiwic2lkIjoiMmJlMzJhZDktOTVmZS00ZjI4LWIwMTItZWQ4NDhjN2YxOWVjIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJNYXJpdXMgSGVycm1hbm4iLCJncm91cHMiOlsiQWRtaW5pc3RyYXRvciIsIkFuYWx5c3QiLCJDcmVhdG9yIiwiVmlld2VyIl0sInByZWZlcnJlZF91c2VybmFtZSI6Im1hcml1cyIsImdpdmVuX25hbWUiOiJNYXJpdXMiLCJmYW1pbHlfbmFtZSI6IkhlcnJtYW5uIiwiZW1haWwiOiJtYXJpdXMuaGVycm1hbm5AZGlzeS5uZXQifQ.JuALFZVSl4QXtdVZeMY7yGfnn8TnUOJNYq1gkLVxV2amGAbHOQqsachPoCsrAYfzL4zMEz0xkpqAlrGcY3s6f3iSXkv5kHEL6HiBoul6S1T9U8jjo89kSb6c3CJUw69a49zor7ZBHcnpahZ06b_VLKdrxRvzrGHO7rlSu2R95xr-mczbtyUONiLIM_t1AglYxa-iuGjKD8lH6uv9ut3ZS0Pmoj2HXRzv-_9bvvuRNlSUMXn5W_5jFrWs32nGJtlttNIEsUIw2UcBmKlga1Eb3moxP6s8PWHr2eM4LlAXOd3DkxWLcottsbQf5lceC2b6xH_wvg92yTqG-_uMrQCFkA"

    os.environ["MLFLOW_TRACKING_TOKEN"] = token

    logged_run = mlflow.get_run(os.getenv('RUN_ID'))
    
    # logged_run = mlflow.get_run(os.getenv('RUN_ID'))

    subfolder = "data_folder"
    file_path = '../Data/water_temps.parquet'

    training_data = mlflow.artifacts.download_artifacts(os.path.join(logged_run.info.artifact_uri, subfolder, file_path.split("/")[-1]).replace("\\","/"), dst_path = "./")

    ref_data = pd.read_parquet(training_data)

    print(ref_data)

    # Get Training data ID column name
    refdata_id_column_name, refdata_id_column_print_name = get_column_names(attribute_groups, "refdata_id")

    t3 = time.time()
    print("t3:",t3-t2)

    # Get Training data column names
    refdata_column_names, refdata_column_print_names = get_column_names(attribute_groups, "refdata")

    # Get Training data datetime column names
    refdata_datetime_column_names, refdata_datetime_column_print_names = get_column_names(attribute_groups, "refdata_date")
        
    t4 = time.time()
    print("t4:", t4-t3)

    # Get Arrays to select and rename the data
    new_data_cols = newdata_id_column_name + newdata_column_names + newdata_datetime_column_names
    new_data_print_cols = newdata_id_column_print_name + newdata_column_print_names + newdata_datetime_column_print_names

    ref_data_cols = refdata_id_column_name + refdata_column_names + refdata_datetime_column_names
    ref_data_print_cols = refdata_id_column_print_name + refdata_column_print_names + refdata_datetime_column_print_names

    print(newdata_column_print_names)

    print(ref_data_print_cols)

    print("\n", new_data_print_cols)

    # Fetch last report if there is an uneven number of columns
    if (len(new_data_print_cols) != len(ref_data_print_cols)):
        if analytics_service.last_url != None:
            return UrlResponse(analytics_service.last_url)
        else:
            return ca.ErrorResponse(f"Error: Uneven number of columns between new data and reference data")
    
    try:
        new_data, ref_data = process_data(data, new_data_cols, new_data_print_cols, ref_data_cols, ref_data_print_cols, npartitions=4)
        
        t5 = time.time()
        print("t5:", t5-t4)

        print("new_data: ", ref_data.shape)
        print("ref_data: ", ref_data.shape)

        # Prepare the data for the POST request
        payload = {
            "reference_data": {
                "data": ref_data.to_dict(orient='split'),
                "id_column": refdata_id_column_print_name,
                # "numerical_columns": refdata_column_names,
                "datetime_columns": refdata_datetime_column_print_names
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

    t6 = time.time()
    print("t6:", t6-t5)

    # Send the request to the data drift calculation endpoint

    try:
        response = requests.post(f"{URL_PART}/app/data_drift", data=orjson.dumps(payload, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY), headers=headers)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        print(f"Error calling data drift service: {e}")
        return ca.ErrorResponse(f"Error generating report: {e}")
    
    t7 = time.time()
    print("t7:", t7-t6)

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
run_id = ca.AttributeGroup(
    name="run_id",
    print_name="MLFlow Run ID",
    data_types=[ca.DataType.STRING],
)

data_drift_extension = ca.CadenzaAnalyticsExtension(
    relative_path="model-data-drift-extension-rs", 
    analytics_function= calculate_data_drift, 
    print_name="Data Drift Extension",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, new_attribute_group_id, new_attribute_group_date, run_id]
)


analytics_service = ca.CadenzaAnalyticsExtensionService()
analytics_service.add_analytics_extension(data_drift_extension)
MEGABYTE = (2 ** 10) ** 2
analytics_service._app.config['MAX_CONTENT_LENGTH'] = None
analytics_service._app.config['MAX_FORM_PARTS'] = 500000
analytics_service._app.config['MAX_FORM_MEMORY_SIZE'] = 500 * MEGABYTE
analytics_service.last_url = None

if __name__ == '__main__':
    analytics_service.run_development_server(5005)
