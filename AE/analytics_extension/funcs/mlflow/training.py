
# Control imports
import os
import pandas as pd
import numpy as np
import json

# MLFlow imports
import mlflow
from mlflow.models import infer_signature
import pickle

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# disy Cadenza imports
import cadenzaanalytics as ca


def retrain_model(metadata: ca.RequestMetadata, data):
    """
    Send input data and a run_id to retrain a model and get the new run and model id, e.g.

    | New_run_id |New_model_id |
    |  d6fui82h  |  a34sdf234  |
    """

    attribute_groups = metadata.get_columns_by_attribute_group()

    # Get input data from Cadenza
    data_group = attribute_groups["input_data"]
    input_cols = [c.name for c in data_group]
    input_data = data[input_cols]

    # Get token and set it as an environment variable
    token= str(metadata.get_parameter("token"))
    os.environ["MLFLOW_TRACKING_TOKEN"] = token

    # Get the run from the run_id
    run_id = str(metadata.get_parameter("run_id"))
    logged_run = mlflow.get_run(run_id)

    # Use the run provided to get the model name
    models = mlflow.search_logged_models(
        experiment_ids=[logged_run.info.experiment_id]
    )
    model_name = models[models.creation_timestamp== models.creation_timestamp.max()].name
    
    # Use the run provided to get the components of the model path 
    model_id = logged_run.outputs.to_dictionary()["model_outputs"][0].model_id
    model_artifact_path = logged_run.info.artifact_uri.split("/"+logged_run.info.run_id+"/")
    
    # Construct the artifact ID to get the model
    model_path = os.path.join(model_artifact_path[0], "models", model_id, model_artifact_path[1], "model.pkl").replace("\\","/")

    # Download the model and instantiate it
    model = mlflow.artifacts.download_artifacts(model_path, dst_path = "./")
    loaded_model = pickle.load(open(model, 'rb'))

    # Get artifact path via the dataset source logged with the dataset
    artifact_path = os.path.join(logged_run.info.artifact_uri, logged_run.inputs.dataset_inputs[0].dataset.source.split("artifacts/")[-1]).replace("\\","/")[:-2]
    training_data_path = mlflow.artifacts.download_artifacts(artifact_path, dst_path = "./")
    training_data_old = pd.read_parquet(training_data_path)

    # Rename new data
    rename_dict_new = dict(zip(input_data.columns.values, training_data_old.columns.values))
    input_data = input_data.rename(columns=rename_dict_new)

    # Construct the input example path
    model_input_path = os.path.join(model_artifact_path[0], "models", model_id, model_artifact_path[1], "input_example.json").replace("\\","/")
    model_input = mlflow.artifacts.download_artifacts(model_input_path, dst_path = "./")

    # Get the predictand
    with open(model_input) as f:
        training_columns = json.load(f)["columns"]

    new_columns = input_data.columns
    target_array = []
    for col in new_columns:
        target_array.append(col in training_columns)

    target = new_columns[target_array.index(False)]

    datasets = [training_data_old, input_data]

    # Ensure there are no values with a comma in target
    # Implementing a check like this for other columns might be a good idea
    for dataset in datasets:
        if any([type(item)!=float for item in dataset[target]]):
            temp_list = [float(str(item).replace(',', '.')) for item in dataset[target]]
            dataset[target] = temp_list

    # Construct combined training data
    training_data_new = pd.concat(datasets)

    training_data_new.dropna(axis=0, inplace=True)

    # Save combined training data to disk in order to pass it to MLFlow later
    file_path = "./training_data_new.parquet"
    training_data_new.to_parquet(file_path)


    # Set predictor and predictand
    X = training_data_new.drop(target, axis=1)

    y = np.array(training_data_new[[target]].values, dtype=float)

    print("Splitting the dataset into training and test sets...")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=17
    )

    # Ensure model is trained with feature names
    feature_names = training_data_new.columns.drop(target)
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_train = pd.Series(y_train.reshape(-1), name="target")
    y_test = pd.Series(y_test.reshape(-1), name="target")

    print("Defining the model hyperparameters...")

    # Define the model hyperparameters
    params = {
        "criterion": "squared_error",
        "random_state": 17,
    }

    print("Training the model...")

    # Train the model
    loaded_model.fit(X_train, y_train)

    print("Predicting on the test set...")

    # Predict on the test set
    y_pred = loaded_model.predict(X_test)

    print("Calculating metrics...")

    # Calculate metric
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("Training complete!")

    print("Logging model parameters, metrics, and artifacts to MLFlow...")

    # Set experiment the run is logged to
    mlflow.set_experiment(experiment_id=logged_run.info.experiment_id)

    # Create a new run
    with mlflow.start_run() as run:
        print("Started run")

        # Log metrics
        mlflow.log_metrics(
            metrics={"mean squared error": mse,
                    "root mean squared error": rmse
                    }
        )

        # Get conda env file
        conda_env_filePath =os.path.join(os.getcwd(), "AE", "conda_env.yml").replace("\\", "/")

        if not os.path.exists(conda_env_filePath):
            return ca.ErrorResponse(f"Error: Please provide a conda env file in the directory this file is run from.")

        # Log the model, which inherits the parameters and metric
        model_info = mlflow.sklearn.log_model(
            sk_model=loaded_model,
            conda_env=conda_env_filePath,
            registered_model_name=model_name[0],
            signature=infer_signature(X_train, y_pred),
            input_example=X_train[:10],
            metadata={
                "author": "disy Cadenza User",
                # "version": "0.0.1",
            },
            params=params,
            tags={"Training Info": "This model was retrained via the disy Cadenza analytics extension.",
                "Additional Info": f"Predictors are {feature_names}, predictand is {target}.",
                "Extra Info": f"This model has an MSE of {mse} and an RMSE of {rmse}."
                },
            name=model_name[0],
        )

        # Create subfolder to log the training dataset to
        subfolder = "data_folder"
        print("subfolder")

        # Log the training dataset as an artifcat
        mlflow.log_artifact(local_path=training_data_path,
                            artifact_path=subfolder,
                            )
        
        # get the run name and id
        run_name = run.info.run_name
        run_id = run.info.run_id

        print("\nRun name: ", run_name)
        print("Run id: ", run_id)

        # Get the run itself
        logged_run = mlflow.get_run(run_id)

        artifact_url_parts = run.info.artifact_uri.split('/')
        experiment_num = artifact_url_parts[1]
        artifact_path = artifact_url_parts[2]

        # Build the url where the training dataset is logged to
        dataset_source_url = os.path.join("https://mlflow.simplex4learning.de/#/experiments/", experiment_num, "runs", artifact_path, "artifacts", subfolder, file_path.split("/")[-1]).replace("\\","/")

        # Associate the training dataset with a URL
        training_dataset = mlflow.data.from_pandas(training_data_new, source = dataset_source_url, name = "maximum water temperatures", targets = target)
        mlflow.log_input(training_dataset, context="training")

        print("Complete: Everything is logged to MLFlow Server. Let's delete the variables now...")

        mlflow.end_run()

    # Delete model and training data
    os.remove(model)
    os.remove(training_data_path) 
    os.remove(file_path)
    os.remove("./input_example.json") 
    
    # Build response frame
    model_info_frame = pd.DataFrame({"run_id":run_id, "model_id":logged_run.to_dictionary()["outputs"]["model_outputs"][0].model_id}, index = [1])
    print(model_info_frame)
    
    # Model id and associated run
    training_metadata = [ca.ColumnMetadata(
            name="run_id",
            print_name="Run ID of new model",
            data_type=ca.DataType.STRING,
            attribute_group_name='Run ID',
            role=ca.AttributeRole.DIMENSION),
    ca.ColumnMetadata(
            name="model_id",
            print_name="ID of new model",
            data_type=ca.DataType.STRING,
            attribute_group_name='Model ID',
            role=ca.AttributeRole.DIMENSION)]
    print(training_metadata)

    return ca.CsvResponse(model_info_frame, training_metadata)


def train_model(metadata: ca.RequestMetadata, data):
    """
    Send input data and a run_id to retrain a decision tree and get the new run and model id, e.g.

    | New_run_id |New_model_id |
    |  d6fui82h  |  a34sdf234  |
    """

    attribute_groups = metadata.get_columns_by_attribute_group()

    # Get input data from Cadenza
    predictor_group = attribute_groups["predictors"]
    predictor_cols = [c.name for c in predictor_group]
    
    predictand_group = attribute_groups["predictand"]
    predictand_col = [c.name for c in predictand_group]



    # Get token and set it as an environment variable
    token= str(metadata.get_parameter("token"))
    os.environ["MLFLOW_TRACKING_TOKEN"] = token

    # Extract experiment ID
    experiment = str(metadata.get_parameter("experiment"))

    print(experiment)

    # Set experiment the run is logged to
    mlflow.set_experiment(experiment)


    # Use the experiment id provided to get the model name
    try:
        models = mlflow.search_logged_models()
        print("Found experiment")
        model_name = models[models.creation_timestamp== models.creation_timestamp.max()].name[0]
        print("Setting model name to: ", model_name)
    except:
        print("Experiment not found, creating a new experiment")
        # Handle case when no model name is provided
        model_name = experiment + "_model"

    data.dropna(axis=0, inplace=True)

    # Save combined training data to disk in order to pass it to MLFlow later
    file_path = "./training_data_new.parquet"
    data.to_parquet(file_path)


    # Set predictor and predictand
    X = data[predictor_cols]

    y = np.array(data[predictand_col].values, dtype=float)

    print("Splitting the dataset into training and test sets...")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=17
    )

    # Ensure model is trained with feature names
    X_train = pd.DataFrame(X_train, columns=predictor_cols)
    X_test = pd.DataFrame(X_test, columns=predictor_cols)
    y_train = pd.Series(y_train.reshape(-1), name="target")
    y_test = pd.Series(y_test.reshape(-1), name="target")

    # Define the model hyperparameters
    params = {
        "criterion": "squared_error",
        "random_state": 17,
    }

    print("Training the model...")
    # Train the model
    lr = DecisionTreeRegressor(**params)
    lr.fit(X_train, y_train)

    print("Predicting on the test set...")
    # Predict on the test set
    y_pred = lr.predict(X_test)

    print("Calculating metrics...")
    # Calculate metric
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("Training complete!")

    print("Logging model parameters, metrics, and artifacts to MLFlow...")

    # Create a new run
    with mlflow.start_run() as run:
        print("Started run")

        # Log metrics
        mlflow.log_metrics(
            metrics={"mean squared error": mse,
                    "root mean squared error": rmse
                    }
        )

        # Get conda env file
        conda_env_filePath =os.path.join(os.getcwd(), "AE", "conda_env.yml").replace("\\", "/")

        if not os.path.exists(conda_env_filePath):
            return ca.ErrorResponse(f"Error: Please provide a conda env file in the directory this file is run from.")
        print("MODELNAME", model_name)
        # Log the model, which inherits the parameters and metric
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            conda_env=conda_env_filePath,
            registered_model_name=model_name,
            signature=infer_signature(X_train, y_pred),
            input_example=X_train[:10],
            metadata={
                "author": "disy Cadenza User",
                # "version": "0.0.1",
            },
            params=params,
            tags={"Training Info": "This model was retrained via the disy Cadenza analytics extension.",
                "Additional Info": f"Predictors are {predictor_cols}, predictand is {predictand_col}.",
                "Extra Info": f"This model has an MSE of {mse} and an RMSE of {rmse}."
                },
            name=model_name,
        )

        # Create subfolder to log the training dataset to
        subfolder = "data_folder"
        print("subfolder")

        # Log the training dataset as an artifcat
        mlflow.log_artifact(local_path=file_path,
                            artifact_path=subfolder,
                            )
        
        # get the run name and id
        run_name = run.info.run_name
        run_id = run.info.run_id

        print("\nRun name: ", run_name)
        print("Run id: ", run_id)

        # Get the run itself
        logged_run = mlflow.get_run(run_id)

        artifact_url_parts = run.info.artifact_uri.split('/')
        experiment_num = artifact_url_parts[1]
        artifact_path = artifact_url_parts[2]

        # Build the url where the training dataset is logged to
        dataset_source_url = os.path.join("https://mlflow.simplex4learning.de/#/experiments/", experiment_num, "runs", artifact_path, "artifacts", subfolder, file_path.split("/")[-1]).replace("\\","/")

        # Associate the training dataset with a URL
        dataset_name = str(experiment)+"_dataset"
        training_dataset = mlflow.data.from_pandas(data, source = dataset_source_url, name = dataset_name, targets = predictand_col[0])
        mlflow.log_input(training_dataset, context="training")

        print("Complete: Everything is logged to MLFlow Server. Let's delete the variables now...")

        mlflow.end_run()

    os.remove(file_path)
    
    # Build response frame
    model_info_frame = pd.DataFrame({"run_id":run_id, "model_id":logged_run.to_dictionary()["outputs"]["model_outputs"][0].model_id}, index = [1])
    print(model_info_frame)
    
    # Model id and associated run
    training_metadata = [ca.ColumnMetadata(
            name="run_id",
            print_name="Run ID of new model",
            data_type=ca.DataType.STRING,
            attribute_group_name='Run ID',
            role=ca.AttributeRole.DIMENSION),
    ca.ColumnMetadata(
            name="model_id",
            print_name="ID of new model",
            data_type=ca.DataType.STRING,
            attribute_group_name='Model ID',
            role=ca.AttributeRole.DIMENSION)]
    print(training_metadata)

    return ca.CsvResponse(model_info_frame, training_metadata)