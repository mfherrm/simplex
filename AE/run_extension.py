# Control imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables when not running in Docker
from dotenv import load_dotenv

# disy Cadenza imports
import cadenzaanalytics as ca
from analytics_extension.utils.logging_config import setup_logging
 
setup_logging()

 # Analytics extensions
from analytics_extension.funcs.mlflow.model_dataset_drift import get_random_sampling_report as model_data_drift_rs
from analytics_extension.funcs.mlflow.model_dataset_drift import get_random_clustering_report as model_data_drift_rc
from analytics_extension.funcs.mlflow.model_runs import get_models_and_runs as model_runs
from analytics_extension.funcs.mlflow.experiments import get_experiments as model_experiments
from analytics_extension.funcs.mlflow.inference import get_predictions_enr as model_inference_enr
from analytics_extension.funcs.mlflow.inference import get_predictions_cal as model_inference_cal
from analytics_extension.funcs.mlflow.training import retrain_model as model_training
from analytics_extension.funcs.datadrift.dataset_drift import get_random_sampling_report as data_drift_rs
from analytics_extension.funcs.datadrift.dataset_drift import get_random_clustering_report as data_drift_rc

# Set host for the data drift reports
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

# Define extension
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

# Define extension
model_and_runs_extension = ca.CadenzaAnalyticsExtension(
    relative_path="model-and-runs-extension", 
    analytics_function= model_runs, 
    print_name="MLFlow models and runs extension",
    extension_type=ca.ExtensionType.CALCULATION,
    attribute_groups=[token],
    parameters=[experiment]
)

# -----------------------------------------------------------------------------------------------------
# Model specific data drift with random sampling
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

# Define extension
model_data_drift_extension_rs = ca.CadenzaAnalyticsExtension(
    relative_path="model-data-drift-extension-rs", 
    analytics_function= model_data_drift_rs, 
    print_name="Model Data Drift Extension Random Sampling",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, new_attribute_group_id, new_attribute_group_date]
)


# -----------------------------------------------------------------------------------------------------
# Model specific data drift with random clustering
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

# Define extension
model_data_drift_extension_rc = ca.CadenzaAnalyticsExtension(
    relative_path="model-data-drift-extension-rc", 
    analytics_function= model_data_drift_rc, 
    print_name="Model Data Drift Extension Random Clustering",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, new_attribute_group_id, new_attribute_group_date]
)

# -----------------------------------------------------------------------------------------------------
# Get MLFlow model predictions via enrichment
# -----------------------------------------------------------------------------------------------------

# Input data
inputs = ca.AttributeGroup(
    name="input_data",
    print_name="Input data",
    data_types=[ca.DataType.STRING, ca.DataType.INT64, ca.DataType.FLOAT64],
    min_attributes=1,

)

# Run ID
run_id = ca.Parameter(
    name="run_id",
    print_name="Run ID",
    parameter_type=ca.DataType.STRING,
    required=True
    
)

# Token
mltoken = ca.Parameter(
    name="token",
    print_name="MLFlow Access Token",
    parameter_type=ca.DataType.STRING,
    required=True

)

# Define extension
model_inference_extension_enr = ca.CadenzaAnalyticsExtension(
    relative_path="model-inference-extension-enr", 
    analytics_function= model_inference_enr, 
    print_name="MLFlow model inference extension enrichment",
    extension_type=ca.ExtensionType.ENRICHMENT,
    attribute_groups=[inputs],
    parameters=[run_id]#, mltoken]
)


# -----------------------------------------------------------------------------------------------------
# Get MLFlow model predictions via calculation
# -----------------------------------------------------------------------------------------------------

# Input data
inputs = ca.AttributeGroup(
    name="input_data",
    print_name="Input data",
    data_types=[ca.DataType.STRING, ca.DataType.INT64, ca.DataType.FLOAT64],
    min_attributes=1,

)

# Run ID
run_id = ca.Parameter(
    name="run_id",
    print_name="Run ID",
    parameter_type=ca.DataType.STRING,
    required=True
    
)

# Token
mltoken = ca.Parameter(
    name="token",
    print_name="MLFlow Access Token",
    parameter_type=ca.DataType.STRING,
    required=True

)

# Define extension
model_inference_extension_cal = ca.CadenzaAnalyticsExtension(
    relative_path="model-inference-extension-cal", 
    analytics_function= model_inference_cal, 
    print_name="MLFlow model inference extension calculation",
    extension_type=ca.ExtensionType.CALCULATION,
    attribute_groups=[inputs],
    parameters=[run_id, mltoken]
)


# -----------------------------------------------------------------------------------------------------
# Retrain MLFlow model
# -----------------------------------------------------------------------------------------------------

# Input data
inputs = ca.AttributeGroup(
    name="input_data",
    print_name="Input data",
    data_types=[ca.DataType.STRING, ca.DataType.INT64, ca.DataType.FLOAT64],
    min_attributes=1,

)

# Run ID
run_id = ca.Parameter(
    name="run_id",
    print_name="Run ID",
    parameter_type=ca.DataType.STRING,
    required=True
    
)

# Token
mltoken = ca.Parameter(
    name="token",
    print_name="MLFlow Access Token",
    parameter_type=ca.DataType.STRING,
    required=True

)

# Define extension
model_training_extension = ca.CadenzaAnalyticsExtension(
    relative_path="model-training-extension", 
    analytics_function= model_training, 
    print_name="MLFlow model training extension",
    extension_type=ca.ExtensionType.CALCULATION,
    attribute_groups=[inputs],
    parameters=[run_id, mltoken]
)


# -----------------------------------------------------------------------------------------------------
# Dataset drift extension random sampling
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

# Define extension
data_drift_extension_rs = ca.CadenzaAnalyticsExtension(
    relative_path="data-drift-extension-rs", 
    analytics_function= data_drift_rs, 
    print_name="Data Drift Extension Random Sampling",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, new_attribute_group_id, new_attribute_group_date, ref_attribute_group, ref_attribute_group_id, ref_attribute_group_date]
)

# -----------------------------------------------------------------------------------------------------
# Dataset drift extension random clustering
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

# Define extension
data_drift_extension_rc = ca.CadenzaAnalyticsExtension(
    relative_path="data-drift-extension-rc", 
    analytics_function= data_drift_rc, 
    print_name="Data Drift Extension Random Clustering",
    extension_type=ca.ExtensionType.VISUALIZATION,
    attribute_groups=[new_attribute_group, new_attribute_group_id, new_attribute_group_date, ref_attribute_group, ref_attribute_group_id, ref_attribute_group_date]
)


# -----------------------------------------------------------------------------------------------------
# Analytics extension
# -----------------------------------------------------------------------------------------------------

# Instantiate analytics extension and add endpoints
analytics_service = ca.CadenzaAnalyticsExtensionService()
analytics_service.add_analytics_extension(experiments_extension)
analytics_service.add_analytics_extension(model_and_runs_extension)
analytics_service.add_analytics_extension(model_data_drift_extension_rs)
analytics_service.add_analytics_extension(model_data_drift_extension_rc)
analytics_service.add_analytics_extension(model_inference_extension_enr)
analytics_service.add_analytics_extension(model_inference_extension_cal)
analytics_service.add_analytics_extension(model_training_extension)
analytics_service.add_analytics_extension(data_drift_extension_rs)
analytics_service.add_analytics_extension(data_drift_extension_rc)

# Create WSGI app (Gunicorn target)
app = analytics_service()

# Apply Flask config for large uploads
MEGABYTE = (2 ** 10) ** 2
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 1073741824))   # 1 GB
app.config['MAX_FORM_PARTS'] = int(os.getenv('MAX_FORM_PARTS', 500000))
app.config['MAX_FORM_MEMORY_SIZE'] = int(os.getenv('MAX_FORM_MEMORY_SIZE', 524288000)) # 500 MB
analytics_service.last_url = None

# Development server entrypoint
if __name__ == '__main__':
    load_dotenv()
    analytics_service.run_development_server(5005)