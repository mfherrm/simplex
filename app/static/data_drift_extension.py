import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Control imports
import os 
import pandas as pd

# disy Cadenza
import cadenzaanalytics as ca
from app.funcs.url_response import UrlResponse

WEBSERVICE_HOST= os.getenv('VISUALISATION_HOST')



urlpart = f"{WEBSERVICE_HOST}/"

#TODO
def calculate_data_drift(metadata: ca.RequestMetadata, data: pd.DataFrame):
    url= "https://images.squarespace-cdn.com/content/v1/5369465be4b0507a1fd05af0/1528837069483-LD1R6EJDDHBY8LBPVHIU/randall-ruiz-272502.jpg"#f"{urlpart}app/data_drift"
    return UrlResponse(url)


new_attribute_group = ca.AttributeGroup(
    name="newdata",
    print_name="New Data",
    data_types=[ca.DataType.STRING, ca.DataType.INT64, ca.DataType.FLOAT64, ca.DataType.ZONEDDATETIME],
    min_attributes=0,
    max_attributes = None
)

ref_attribute_group = ca.AttributeGroup(
    name="refdata",
    print_name="Model Training Data",
    data_types=[ca.DataType.STRING, ca.DataType.INT64, ca.DataType.FLOAT64, ca.DataType.ZONEDDATETIME],
    min_attributes=0,
    max_attributes = None
)

data_drift_extension = ca.CadenzaAnalyticsExtension(
    relative_path="data-drift-extension", 
    analytics_function= calculate_data_drift, 
    print_name="Data Drift Extension",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, ref_attribute_group]
)

analytics_service = ca.CadenzaAnalyticsExtensionService()
analytics_service.add_analytics_extension(data_drift_extension)

if __name__ == '__main__':
    analytics_service.run_development_server(5005)
