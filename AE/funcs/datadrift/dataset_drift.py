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

# SKlearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# disy Cadenza imports
import cadenzaanalytics as ca
from AE.extension_funcs.url_response import UrlResponse

WEBSERVICE_HOST = os.getenv('VISUALISATION_HOST', 'http://127.0.0.1:5000')
URL_PART = f"{WEBSERVICE_HOST}"

# -----------------------------------------------------------------------------------------------------
# Methods to process data
# -----------------------------------------------------------------------------------------------------


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

def process_data(data: pl.DataFrame, common_columns, data_cols = None, data_print_cols = None, desired_order = [], clustering = False, fraction = 0.15):
    """
    Preprocesses data by renaming, reordering and selecting only numeric columns. Samples data afterwards.
    """
    data = data.drop_nans()

    if data_cols:
        # Map cadenza column names to their print names
        rename_dict_new = dict(zip(data_cols, data_print_cols))

        # Select only common columns (columns that occur in both training and new data)
        short_dict = {key:value for key, value in rename_dict_new.items() if value in common_columns}

        # Select common columns, order and rename them identically to the training data
        data = data[list(short_dict.keys())].rename(short_dict)

    # Select numeric columns
    numeric_df = data.select(cs.numeric())

    if any(desired_order):
            numeric_df = numeric_df[desired_order]

    # Handle empty dataframe
    if any([dim==0 for dim in numeric_df.shape]):
        print("No numerical columns in the dataframe")
        return data.head(0)
    
    if clustering:
        return get_clustering_samples(numeric_df, n_clusters = 5, fraction = fraction, seed=17)
    else:
    # Sample to speed up report generation
        return numeric_df.sample(fraction = fraction, seed=17)

def get_clustering_samples(df:pl.DataFrame, n_clusters: int = 5, fraction = 0.15, seed = 17):
    """
    Clusters and samples input data 
    """
    try:
        # Handle empty dataframe
        if any([dim==0 for dim in df.shape]):
            print("No numerical columns in the dataframe")
            return df.head(0)
        
        # Adjust n_clusters if needed
        actual_n_clusters = min(n_clusters, df.shape[0] - 1)
        if actual_n_clusters != n_clusters:
            print(f"Adjusted number of clusters to {actual_n_clusters}")

        scaler = StandardScaler().set_output(transform="polars")
        X_scaled = scaler.fit_transform(df)

        # Perform clustering
        kmeans = KMeans(n_clusters=actual_n_clusters,
                                  init="k-means++",
                                  random_state=seed,
                                  n_init=10).set_output(transform="polars")
        cluster= kmeans.fit_predict(X_scaled)

        # Reattach labels to original DataFrame
        df_copy = df.clone()
        df_copy = df_copy.with_columns(
            cluster = cluster
        )
        # Sample from each cluster
        print("Sampling from clusters...")
        result = pl.DataFrame()
        for cluster_id in range(actual_n_clusters):
            
            cluster_data = df_copy.filter(df_copy['cluster'] == cluster_id)
            
            if not cluster_data.is_empty():
                sample = cluster_data.sample(fraction=fraction, seed=seed)
                result = pl.concat([result, sample])
        
        return result.drop('cluster')
        
    except Exception as e:
        print(f"Error in clustering: {e}")
        print("Falling back to random sampling")
        return df.sample(fraction=fraction, seed=seed)

# -----------------------------------------------------------------------------------------------------
# Calculate dataset drift report
# -----------------------------------------------------------------------------------------------------

def calculate_data_drift(metadata: ca.RequestMetadata, data, clustering=False):
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

    # Get Training data ID column name
    refdata_id_column_name, refdata_id_column_print_name = get_column_names(attribute_groups, "refdata_id")

    # Get Training data column names
    refdata_column_names, refdata_column_print_names = get_column_names(attribute_groups, "refdata")

    # Get Training data datetime column names
    refdata_datetime_column_names, refdata_datetime_column_print_names = get_column_names(attribute_groups, "refdata_date")

    # Get Arrays to select and rename the data
    new_data_cols = newdata_id_column_name + newdata_column_names + newdata_datetime_column_names
    new_data_print_cols = newdata_id_column_print_name + newdata_column_print_names + newdata_datetime_column_print_names

    ref_data_cols = refdata_id_column_name + refdata_column_names + refdata_datetime_column_names
    ref_data_print_cols = refdata_id_column_print_name + refdata_column_print_names + refdata_datetime_column_print_names

    common_columns = list(set(ref_data_print_cols).intersection(new_data_print_cols))

    # Fetch last report if there is an uneven number of columns
    if len(common_columns)<1: #(len(new_data_print_cols) != len(ref_data_print_cols)):
        # if analytics_service.last_url != None:
        #     return UrlResponse(analytics_service.last_url)
        # else:
        return ca.ErrorResponse(f"Error: No common columns between new data and reference data")

    try:
        # Process training data 
        ref_data = process_data(pl.from_pandas(data[ref_data_cols]), common_columns, ref_data_cols, ref_data_print_cols,  clustering=clustering)

        print("Finished processing the reference data")
        # Process new data
        new_data = process_data(pl.from_pandas(data[new_data_cols]), common_columns, new_data_cols, new_data_print_cols, desired_order = ref_data.columns, clustering=clustering)
        
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
    return report_url

# -----------------------------------------------------------------------------------------------------
# Dataset drift extension random sampling
# -----------------------------------------------------------------------------------------------------


def get_random_sampling_report(metadata: ca.RequestMetadata, data):
    report_url = calculate_data_drift(metadata, data, clustering=False)
    return UrlResponse(report_url)

# -----------------------------------------------------------------------------------------------------
# Dataset drift extension random clustering
# -----------------------------------------------------------------------------------------------------

def get_random_clustering_report(metadata: ca.RequestMetadata, data):
    report_url = calculate_data_drift(metadata, data, clustering=True)
    return UrlResponse(report_url)
