# Control imports
import getpass
import os

# Data management imports
import pandas as pd

# MLFlow imports
import mlflow

def mlflow_auth(URI = "https://mlflow.simplex4learning.de/"):
    MLFLOW_TRACKING_URI = URI

    # 1. Set MLflow Tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

    # 2. get bearer token
    auth_token = getpass.getpass("\nPlease enter the bearer token you got from the network tab in your web browser.")

    # 3. Set the token and run MLflow command
    if auth_token and auth_token.strip():
        os.environ["MLFLOW_TRACKING_TOKEN"] = auth_token
        # If the server has a self-signed SSL certificate
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    else:
        print("\nNo token provided. Aborting MLflow operation.")

    try:
        print("\nTesting connection to MLflow server...")
        # Make the lightest possible API call to test the token
        mlflow.search_experiments(max_results=1, view_type=mlflow.entities.ViewType.ALL)
        print("Authentication successful! Connection to MLflow is established.")
        return True
    except mlflow.MlflowException as e:
        print("\nAuthentication failed. Please check that your token is correct and has not expired.")
        # Specifically check for authentication error codes if possible
        if "UNAUTHENTICATED" in str(e) or "PERMISSION_DENIED" in str(e):
             print("Error details: The server rejected the token.")
        else:
             print(f"Error details: {e}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred while testing the connection: {e}")
        return False
    
def get_runs(verbose=True):
    try:   
        # Get all runs in current connection and print them
        all_runs = mlflow.search_runs(search_all_experiments=True)
        print("\n--- MLflow Runs ---")
        if verbose: print(all_runs)
        print("-------------------")
        if all_runs.empty:
            print("Connection successful, but no runs were found.")
        else:
            print("Successfully retrieved all runs.")
        return  all_runs

    except Exception as e:
        print(f"\nAn error occurred during the MLflow operation: {e}")

def get_experiments(verbose=True):
    try: 
        # Shows only active experiments, use view_type= mlflow.entities.ViewType.ALL to show all
        experiments = mlflow.search_experiments()
        if not experiments:
            print("No experiments found on the server.")
            return pd.DataFrame()

        # Create a list of dictionaries from the experiment objects
        exp_data = [
            {
                "experiment_id": int(exp.experiment_id),
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
                "creation_time": pd.to_datetime(exp.creation_time, unit='ms'),
                "last_update_time": pd.to_datetime(exp.last_update_time, unit='ms'),
                "tags": exp.tags,
            }
            for exp in experiments
        ]
        
        exp_df = pd.DataFrame(exp_data)
        print("\n--- MLflow Experiments ---")
        if verbose: print(exp_df) 
        print("-------------------")
        return exp_df

    except Exception as e:
        print(f"\nAn error occurred during the MLflow operation: {e}")