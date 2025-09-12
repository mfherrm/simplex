# Control imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import requests

# Data processing
import polars as pl
import polars.selectors as cs
import orjson

# MLFlow imports
import mlflow

# disy Cadenza imports
import cadenzaanalytics as ca
from AE.extension_funcs.url_response import UrlResponse

WEBSERVICE_HOST = os.getenv('VISUALISATION_HOST', 'http://127.0.0.1:5000')
URL_PART = f"{WEBSERVICE_HOST}"

def get_column_names(attribute_groups: dict, group:str):
    """
    Returns column names and column print names for specified group.
    """
    try:
        columns = attribute_groups[group]
        return [c.name for c in columns], [c.print_name for c in columns]
    except:
        print("Column type not given")
        return [], []

def process_data(data: pl.DataFrame, common_columns, new_data_cols = None, new_data_print_cols = None, desired_order = None):
    """
    Preprocesses data by renaming, reordering and selecting only numeric columns. Samples data afterwards.
    """
    data = data.drop_nans()

    if new_data_cols:
        # Map cadenza column names to their print names
        rename_dict_new = dict(zip(new_data_cols, new_data_print_cols))

        # Select only common columns (columns that occur in both training and new data)
        short_dict = {key:value for key, value in rename_dict_new.items() if value in common_columns}

        # Select common columns, rename them and order them identically to the training data
        data = data[list(short_dict.keys())].rename(short_dict)[desired_order]
        
    # Select numeric columns
    numeric_df = data.select(cs.numeric())

    # Handle empty dataframe
    if any([dim==0 for dim in numeric_df.shape]):
        print("No numerical columns in the dataframe")
        return data.head(0)
    
    # Sample to speed up report generation
    return numeric_df.sample(fraction=0.15, seed=17)


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
    # mlflow.get_run(run_id)

    logged_run = mlflow.get_run(os.getenv('RUN_ID'))

    # Get artifact path via the dataset source logged with the dataset
    artifact_path = os.path.join(logged_run.info.artifact_uri, logged_run.inputs.dataset_inputs[0].dataset.source.split("artifacts/")[-1]).replace("\\","/")[:-2]

    training_data = mlflow.artifacts.download_artifacts(artifact_path, dst_path = "./")

    ref_data = pl.read_parquet(training_data)

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
        # if analytics_service.last_url != None:
        #     return UrlResponse(analytics_service.last_url)
        # else:
        return ca.ErrorResponse(f"Error: No common columns between new data and reference data")
    
    try:
        # Process training data 
        ref_data = process_data(ref_data, common_columns)

        # Process new data
        new_data = process_data(pl.from_pandas(data), common_columns, new_data_cols, new_data_print_cols, desired_order = ref_data.columns)
        
        t3 = time.time()
        print("Processed data:", t3-t2)

        # Prepare the data for the POST request
        payload = {
            "reference_data": {
                "data": ref_data.to_dicts(),
                "id_column": [],
                # "numerical_columns": refdata_column_names,
                "datetime_columns": []
            },
            "current_data": {
                "data": new_data.to_dicts(),
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
    # analytics_service.last_url = report_url
    return UrlResponse(report_url)
