import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Control imports
import pandas as pd
import requests
import json
import time

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# disy Cadenza
import cadenzaanalytics as ca
from AE.url_response import UrlResponse



WEBSERVICE_HOST = os.getenv('VISUALISATION_HOST', 'http://127.0.0.1:5000')
URL_PART = f"{WEBSERVICE_HOST}"

def get_clustering_samples(df:pd.DataFrame, n_clusters:int=5, samples_per_cluster:int=10):
    # Handle non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Drop rows with any NaN values and store the result
    cleaned_numeric_df = numeric_df.dropna()
    
    # Check if there's any data left to cluster
    if cleaned_numeric_df.empty:
        # Return an empty DataFrame, or handle this case as appropriate
        # depending on what you want to do with a dataset that's all NaNs.
        return pd.DataFrame() 

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_numeric_df)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Add the cluster labels to the cleaned DataFrame, not the original
    cleaned_numeric_df['cluster'] = cluster_labels
    
    # Sample from each cluster from the cleaned DataFrame
    sampled_df = cleaned_numeric_df.groupby('cluster').apply(
        lambda x: x.sample(min(len(x), samples_per_cluster))
    )
    
    # Clean up the resulting DataFrame
    return sampled_df.drop('cluster', axis=1).reset_index(level=0, drop=True)
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

def calculate_data_drift(metadata: ca.RequestMetadata, data: pd.DataFrame):
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

    rename_dict = dict(zip(new_data_cols, new_data_print_cols))
    new_data=data[new_data_cols].rename(columns=rename_dict)
    new_data = get_clustering_samples(new_data, n_clusters=10, samples_per_cluster=300)

    ref_data_cols = refdata_id_column_name + refdata_column_names + refdata_datetime_column_names
    ref_data_print_cols = refdata_id_column_print_name + refdata_column_print_names + refdata_datetime_column_print_names

    rename_dict = dict(zip(ref_data_cols, ref_data_print_cols))
    ref_data= data[ref_data_cols].rename(columns=rename_dict)
    ref_data = get_clustering_samples(ref_data, n_clusters=10, samples_per_cluster=300)
    
    t5 = time.time()
    print("t5: ", t5-t4)

    # Prepare the data for the POST request
    payload = {
        "reference_data": {
            "data": ref_data.to_json(orient='split'),
            "id_column": refdata_id_column_print_name,
            # "numerical_columns": refdata_column_names,
            "datetime_columns": refdata_datetime_column_print_names
        },
        "current_data": {
            "data": new_data.to_json(orient='split'),
            "id_column": newdata_id_column_print_name,
            # "numerical_columns": newdata_column_names,
            "datetime_columns": newdata_datetime_column_print_names
        }
    }
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
