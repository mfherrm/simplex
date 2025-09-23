# Simplex

This repository contains an analytics extension that adds a data processing pipeline to disy Cadenza.

## Implementation
In order to run this analytics extension, disy Cadenza > 10.2.171 must be installed and the FuE feature toggle must be set (``-Dnet.disy.cadenza.analytics.extensions.fue=true``). 

Additionally, all endpoints need to be added in disy Cadenza.
This is done in the 'external content' tab of the management center.

### Endpoints
The endpoints are as follows:
- [experiments-extension](AE/analytics_extension/funcs/mlflow/experiments.py)
- [model-and-runs-extension](AE/analytics_extension/funcs/mlflow/model_runs.py)
- [model-data-drift-extension-rs](AE/analytics_extension/funcs/mlflow/model_dataset_drift.py)
- [model-data-drift-extension-rc](AE/analytics_extension/funcs/mlflow/model_dataset_drift.py)
- [model-inference-extension-enr](AE/analytics_extension/funcs/mlflow/inference.py)
- [model-inference-extension-cal](AE/analytics_extension/funcs/mlflow/inference.py)
- [model-training-extension](AE/analytics_extension/funcs/mlflow/training.py)
- [data-drift-extension-rs](AE/analytics_extension/funcs/datadrift/dataset_drift.py)
- [data-drift-extension-rc](AE/analytics_extension/funcs/datadrift/dataset_drift.py)

Thus, endpoints are added like ``http://127.0.0.1:5005/experiments-extension``

## Structure
The repository is made up of the following parts:
- [AE](AE): this directory contains the Docker container for the analytics extension ([Dockerfile](AE/Dockerfile))
- [app](app): this directory contains the Docker container for the data drift report generation ([Dockerfile](app/Dockerfile))

Both containers need to be run in order to use the model data drift extensions and the data drift extension. If these extensions are not needed, running only the AE container is sufficient.

If you do not want to run the analytics extension in a Docker container, you can run it directly from the [run_extension.py](AE/run_extension.py) file. This requires all packages in [AE/requirements.txt](AE/requirements.txt) to be installed in the python environment the file is run from.

## Workflow
The pipeline assumes a pretrained scikit-learn model which has been pushed to MLFlow. A user then receives new data with the same structure. The workflow then looks as follows:
- Generate a MLFlow access token and paste it into the [token.txt](token.txt) file
- Import the token in a Cadenza Data Store
- Use the [experiments extension](AE/analytics_extension/funcs/mlflow/experiments.py)
  - Input: access token
  - Output: Experiment ID, experiment name
- Use the [models and runs extension](AE/analytics_extension/funcs/mlflow/model_runs.py)
  - Input: access token, experiment ID
  - Output: Run ID, run name
- Use the [model data drift extension](AE/analytics_extension/funcs/mlflow/model_dataset_drift.py)
  - Inputs: new data columns, new data ID columns, new data datetime columns
  - Run ID and token are hard coded due to limitations of disy Cadenza
  - Outputs: generated data drift report is embedded into the disy Cadenza view
- Use the [model training extension](AE/analytics_extension/funcs/mlflow/training.py) for significant data drift
  - Inputs: input data, run ID, access token
  - The model specified in the run and its training data is downloaded from MLFlow and used in tandem with the new data to retrain the model
  - The trained model is then logged to MLFlow
  - Outputs: run ID, model ID
- Use the [model inference extension](AE/analytics_extension/funcs/mlflow/inference.py) for insignificant data drift
  - Inputs: new data
  - Outputs: model predictions 