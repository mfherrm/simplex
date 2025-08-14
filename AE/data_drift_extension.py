import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Control imports
import requests
import json
import time

# Clustering imports
from dask_ml.preprocessing import StandardScaler
from dask_ml.cluster import KMeans
from dask.distributed import wait

# Data processing imports
import pandas as pd
import dask as dd
from dask.distributed import Client
import numpy as np
import graphviz
# Set Graphviz executable path
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"



# disy Cadenza imports
import cadenzaanalytics as ca
from AE.url_response import UrlResponse

WEBSERVICE_HOST = os.getenv('VISUALISATION_HOST', 'http://127.0.0.1:5000')
URL_PART = f"{WEBSERVICE_HOST}"

def sample_partition(pdf, samples_per_cluster: float = 0.25, random_state = 17):
    return pdf.groupby("cluster", group_keys=False).apply(
        lambda x: x.sample(frac=samples_per_cluster, random_state=random_state)
)

def get_clustering_samples(df: dd.dataframe.DataFrame, n_clusters: int = 5, random_state = 17):

    # Select numeric columns and ensure consistent dtype
    numeric_df = df.select_dtypes(include=[np.number]).astype("float64").dropna()

    # Handle case where no usable columns remain
    n_rows = numeric_df.shape[0].compute()
    n_cols = len(numeric_df.columns)
    if n_rows == 0 or n_cols == 0:
        return df.head(0)

    # Convert to Dask array and standardize
    X = numeric_df.to_dask_array(lengths=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, init="k-means||", random_state=random_state)
    kmeans.fit(X_scaled)
    labels_da = kmeans.predict(X_scaled).persist()
    wait(labels_da)

    # Reattach labels to original cleaned Dask DataFrame
    cluster_series = dd.dataframe.from_dask_array(labels_da, index=numeric_df.index, columns="cluster")

    # Sample from each cluster
    clustered_df = numeric_df.assign(cluster=cluster_series)

    sampled_df = clustered_df.map_partitions(sample_partition, meta=clustered_df._meta)

    return sampled_df.drop("cluster", axis=1).reset_index(drop=True)


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

def process_data(data, new_data_cols, new_data_print_cols, ref_data_cols, ref_data_print_cols, npartitions=4):
    # Initialize Dask client
    with Client() as client:
        # Convert to Dask DataFrame
        data.dropna(inplace=True)
        dask_df = dd.dataframe.from_pandas(data, npartitions=npartitions)
        
        # Process new data
        rename_dict_new = dict(zip(new_data_cols, new_data_print_cols))
        new_data = dask_df[new_data_cols].rename(columns=rename_dict_new)
        # Explicitly compute to materialize the result and break dependency chains
        new_data = dd.dataframe.from_pandas(data=new_data.compute(),npartitions=npartitions)
        
        # Process reference data with a separate operation
        rename_dict_ref = dict(zip(ref_data_cols, ref_data_print_cols))
        ref_data = dask_df[ref_data_cols].rename(columns=rename_dict_ref)
        # Explicitly compute to materialize the result and break dependency chains
        ref_data = dd.dataframe.from_pandas(data=ref_data.compute(),npartitions=npartitions)

        ref_data.visualize(tasks=True, filename='ref.svg')

        # Cluster separately
        new_data = get_clustering_samples(new_data)
        ref_data = get_clustering_samples(ref_data)
        
        # Ensure computation is complete before exiting the client context
        new_data = new_data.persist()
        ref_data = ref_data.persist()
        wait(new_data)
        wait(ref_data)
    
    return new_data, ref_data


def calculate_data_drift(metadata: ca.RequestMetadata, data: dd.dataframe):
    """
    Sends two dataframes to the Flask service to calculate data drift
    and returns a URL to the drift report.
    """
    t0 = time.time()
    attribute_groups = metadata.get_columns_by_attribute_group()
    
    # Get newdata ID column name(s)
    newdata_id_column_name, newdata_id_column_print_name = get_column_names(attribute_groups, "newdata_id")

    t1 = time.time()
    print("t1: ", t1-t0)

    # Get newdata column names
    newdata_column_names, newdata_column_print_names = get_column_names(attribute_groups, "newdata")

    # Get newdata datetime column names
    newdata_datetime_column_names, newdata_datetime_column_print_names = get_column_names(attribute_groups, "newdata_date")

    t2 = time.time()
    print("t2: ", t2-t1)

    # Get Training data ID column name
    refdata_id_column_name, refdata_id_column_print_name = get_column_names(attribute_groups, "refdata_id")

    t3 = time.time()
    print("t3: ",t3-t2)

    # Get Training data column names
    refdata_column_names, refdata_column_print_names = get_column_names(attribute_groups, "refdata")

    # Get Training data datetime column names
    refdata_datetime_column_names, refdata_datetime_column_print_names = get_column_names(attribute_groups, "refdata_date")
        
    t4 = time.time()
    print("t4: ", t4-t3)

    # Separate the combined dataframe into the two original dataframes
    # based on the attribute groups defined below.
    new_data_cols = newdata_id_column_name + newdata_column_names + newdata_datetime_column_names
    new_data_print_cols = newdata_id_column_print_name + newdata_column_print_names + newdata_datetime_column_print_names

    ref_data_cols = refdata_id_column_name + refdata_column_names + refdata_datetime_column_names
    ref_data_print_cols = refdata_id_column_print_name + refdata_column_print_names + refdata_datetime_column_print_names

    try:
        new_data, ref_data = process_data(data, new_data_cols, new_data_print_cols, ref_data_cols, ref_data_print_cols, npartitions=4)
        
        t5 = time.time()
        print("t5: ", t5-t4)
    
        # Compute DataFrames once and store results to avoid recomputation
        ref_data_computed = ref_data.compute()
        new_data_computed = new_data.compute()
        
        # Prepare the data for the POST request
        payload = {
            "reference_data": {
                "data": ref_data_computed.to_json(orient='split'),
                "id_column": refdata_id_column_print_name,
                # "numerical_columns": refdata_column_names,
                "datetime_columns": refdata_datetime_column_print_names
            },
            "current_data": {
                "data": new_data_computed.to_json(orient='split'),
                "id_column": newdata_id_column_print_name,
                # "numerical_columns": newdata_column_names,
                "datetime_columns": newdata_datetime_column_print_names
            }
        }
    except Exception as e:
        print(f"Error processing data: {e}")
        return UrlResponse(f"data:text/plain,Error processing data: {e}")
    headers = {'Content-Type': 'application/json'}

    t6 = time.time()
    print("t6: ", t6-t5)

    # Send the request to the data drift calculation endpoint
    try:
        response = requests.post(f"{URL_PART}/app/data_drift", data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        # Handle connection errors or bad responses
        # In a real application, you'd want to log this error
        # and potentially return an error response to the user.
        print(f"Error calling data drift service: {e}")
        # For now, we'll return a simple error message to the user
        return UrlResponse(f"data:text/plain,Error generating report: {e}")
    
    t7 = time.time()
    print("t7: ", t7-t6)

    response_data = response.json()
    # Return the URL to view the generated report
    report_url = response_data.get("report_url")
    print(report_url.split(")"))

    # Return the URL to view the generated report
    report_url = f"{URL_PART}"+report_url.split(")")[1]
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

# Training dataset ID column
ref_attribute_group_id = ca.AttributeGroup(
    name="refdata_id",
    print_name="Training Data ID Column",
    data_types=[ca.DataType.STRING, ca.DataType.INT64],
    min_attributes=0,
    max_attributes = 1
)

# Training dataset 
ref_attribute_group = ca.AttributeGroup(
    name="refdata",
    print_name="Training Data",
    data_types=[ca.DataType.STRING, ca.DataType.INT64, ca.DataType.FLOAT64],
    min_attributes=1,
    max_attributes = None
)

# Training dataset datetime column
ref_attribute_group_date = ca.AttributeGroup(
    name="refdata_date",
    print_name="Training Data Datetime Column(s)",
    data_types=[ca.DataType.ZONEDDATETIME],
)

params = ca.Parameter(
    name="sampling",
    print_name ="Use clustering sampling?",
    parameter_type="BOOLEAN",
    required = True,
    options = [True, False],	
    default_value= False
) 

data_drift_extension = ca.CadenzaAnalyticsExtension(
    relative_path="data-drift-extension", 
    analytics_function= calculate_data_drift, 
    print_name="Data Drift Extension",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, new_attribute_group_id, new_attribute_group_date, ref_attribute_group, ref_attribute_group_id, ref_attribute_group_date],
    parameters=[params]
)


analytics_service = ca.CadenzaAnalyticsExtensionService()
analytics_service.add_analytics_extension(data_drift_extension)

if __name__ == '__main__':
    analytics_service.run_development_server(5005)

