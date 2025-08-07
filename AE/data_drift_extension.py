import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Control imports
import os 
import pandas as pd
import requests
import json

# disy Cadenza
import cadenzaanalytics as ca
from AE.url_response import UrlResponse

WEBSERVICE_HOST = os.getenv('VISUALISATION_HOST', 'http://127.0.0.1:5000')
URL_PART = f"{WEBSERVICE_HOST}"

def calculate_data_drift(metadata: ca.RequestMetadata, data: pd.DataFrame):
    """
    Sends two dataframes to the Flask service to calculate data drift
    and returns a URL to the drift report.
    """

    # Get newdata ID column name(s)
    newdata_id_column_name = []
    try:
        newdata_id = metadata.get_columns_by_attribute_group()['newdata_id']
        if newdata_id:
            newdata_id_column_name = [c.name for c in newdata_id]
    except:
        pass
    # Get newdata column names
    newdata_column_names = [c.name for c in metadata.get_columns_by_attribute_group()['newdata']]

    # Get newdata datetime column names
    newdata_datetime_column_names = []
    try:
        newdata_date = metadata.get_columns_by_attribute_group()['newdata_date']
        if newdata_date:
            newdata_datetime_column_names = [c.name for c in newdata_date]
    except:
        pass

    # Get Training data ID column name
    refdata_id_column_name = []
    try:
        refdata_id = metadata.get_columns_by_attribute_group()['refdata_id']
        if refdata_id:
            refdata_id_column_name = [c.name for c in refdata_id]
    except:
        pass

    # Get Training data column names
    refdata_column_names = [c.name for c in metadata.get_columns_by_attribute_group()['refdata']]

    # Get Training data datetime column names
    refdata_datetime_column_names = []
    try:
        refdata_date = metadata.get_columns_by_attribute_group()['refdata_date']
        if refdata_date:
            refdata_datetime_column_names = [c.name for c in refdata_date]
    except:
        pass
    
    # Separate the combined dataframe into the two original dataframes
    # based on the attribute groups defined below.
    new_data_cols = newdata_id_column_name + newdata_column_names + newdata_datetime_column_names
    new_data = data[new_data_cols]

    ref_data_cols = refdata_id_column_name + refdata_column_names + refdata_datetime_column_names
    ref_data = data[ref_data_cols]
    
    # Prepare the data for the POST request
    payload = {
        "reference_data": {
            "data": ref_data.to_json(orient='split'),
            "id_column": refdata_id_column_name,
            # "numerical_columns": refdata_column_names,
            "datetime_columns": refdata_datetime_column_names
        },
        "current_data": {
            "data": new_data.to_json(orient='split'),
            "id_column": newdata_id_column_name,
            # "numerical_columns": newdata_column_names,
            "datetime_columns": newdata_datetime_column_names
        }
    }
    headers = {'Content-Type': 'application/json'}

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


    # Return the URL to view the generated report
    report_url = f"{URL_PART}/app/view_report"
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



data_drift_extension = ca.CadenzaAnalyticsExtension(
    relative_path="data-drift-extension", 
    analytics_function= calculate_data_drift, 
    print_name="Data Drift Extension",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, new_attribute_group_id, new_attribute_group_date, ref_attribute_group, ref_attribute_group_id, ref_attribute_group_date]
)

analytics_service = ca.CadenzaAnalyticsExtensionService()
analytics_service.add_analytics_extension(data_drift_extension)

if __name__ == '__main__':
    analytics_service.run_development_server(5005)
