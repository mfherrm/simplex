from flask import Blueprint, render_template, request, jsonify, send_from_directory
from io import StringIO

# Control imports
import os

# custom funcs
from app.funcs import evidently_funcs as ef
from app.funcs import mlflow_funcs as mf

# Data management imports
import pandas as pd

# ML imports
from sklearn.model_selection import train_test_split

# Evidently imports
from evidently.presets import DataDriftPreset
from evidently import Dataset
from evidently import Report

bp = Blueprint('main', __name__)

WEBSERVICE_HOST = os.getenv('VISUALISATION_HOST'),
urlpart = f"{WEBSERVICE_HOST}"

@bp.route('/app/data_drift', methods=['POST'])
def data_drift():
    """
    Receives two dataframes, computes a data drift report, saves it,
    and returns a success message.
    """
    data = request.get_json()

    print(data)

    try:
        # Unpack payload
        ref_payload = data.get('reference_data')
        new_payload = data.get('current_data')

        if not ref_payload or not new_payload:
            return jsonify({"error": "Missing reference_data or current_data"}), 400

         # --- Process New Data ---
        new_df = pd.read_json(StringIO(new_payload['data']), orient='split')
        new_df.columns = new_df.columns.astype(str)
        new_df.to_csv("newdf.csv", index=False)
        try:
            new_id_col = new_payload['id_column'][0]
        except:
             new_id_col = None

        new_dt_cols = new_payload['datetime_columns']

        # --- Process Reference Data ---
        ref_df = pd.read_json(StringIO(ref_payload['data']), orient='split')
        ref_df.columns = ref_df.columns.astype(str)

        ref_df.to_csv("refdf.csv", index=False)
        try:
            ref_id_col = ref_payload['id_column'][0]
        except:
            ref_id_col = None

        # Use new datetime columns since they are going to be renamed regardless
        ref_dt_cols = new_payload['datetime_columns']
           
        # Move ID column to first position
        if ref_id_col != None and new_id_col != None:
            new_df.insert(0, new_id_col, new_df.pop(new_id_col))
            ref_df.insert(0, ref_id_col, ref_df.pop(ref_id_col))
        

        # Rename the columns
        rename_dict = dict(zip(ref_df.columns[:], new_df.columns[:]))
        ref_df.rename(columns=rename_dict, inplace=True)

        # Process data and create dataset templates
        new_processed, data_def_new = ef.map_to_def(new_df, new_id_col, new_dt_cols)
        new_dataset = Dataset.from_pandas(new_processed, data_def_new)

        ref_processed, data_def_ref = ef.map_to_def(ref_df, ref_id_col, ref_dt_cols)
        ref_dataset = Dataset.from_pandas(ref_processed, data_def_ref)
        
        # wasserstein does not work for categorical data
        report = Report([DataDriftPreset(method="psi")], include_tests="True")
        reps = report.run(ref_dataset, new_dataset)
        
        # Save the report to a file
        reps.save_html('app/templates/report.html')

        return jsonify({"message": "Report generated successfully"})
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@bp.route('/app/view_report', methods=['GET'])
def view_report():
    """
    Renders the previously generated data drift report.
    """
    return send_from_directory(os.path.abspath('app/templates'), 'report.html')