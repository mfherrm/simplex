import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Control imports
from dotenv import load_dotenv

# disy Cadenza imports
import cadenzaanalytics as ca

# Analytics extensions
from AE.funcs.mlflow.model_dataset_drift import calculate_data_drift as model_data_drift
from AE.funcs.mlflow.model_runs import get_models_and_runs as model_runs
from AE.funcs.mlflow.experiments import get_experiments as model_experiments

WEBSERVICE_HOST = os.getenv('VISUALISATION_HOST', 'http://127.0.0.1:5000')
URL_PART = f"{WEBSERVICE_HOST}"

# -----------------------------------------------------------------------------------------------------
# Get MLFlow experiments
# -----------------------------------------------------------------------------------------------------

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
    analytics_function= model_experiments, 
    print_name="MLFlow Experiments Extension",
    extension_type=ca.ExtensionType.CALCULATION,
    attribute_groups=[token]
)

# -----------------------------------------------------------------------------------------------------
# Get MLFlow model runs
# -----------------------------------------------------------------------------------------------------

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
    analytics_function= model_runs, 
    print_name="MLFlow models and runs extension",
    extension_type=ca.ExtensionType.CALCULATION,
    attribute_groups=[token],
    parameters=[experiment]
)

# -----------------------------------------------------------------------------------------------------
# Model specific data drift
# -----------------------------------------------------------------------------------------------------


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
    analytics_function= model_data_drift, 
    print_name="Model Data Drift Extension Random Sampling",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, new_attribute_group_id, new_attribute_group_date]#, run_id]
)

# -----------------------------------------------------------------------------------------------------
# Analytics extension
# -----------------------------------------------------------------------------------------------------

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