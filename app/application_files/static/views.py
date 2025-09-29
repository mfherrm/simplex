from flask import Blueprint, request, jsonify, send_from_directory

# Control imports
import os
import hashlib
import time

# custom funcs
from application_files.funcs import evidently_funcs as ef

# Data management imports
import polars as pl
import orjson

# Evidently imports
from evidently.presets import DataDriftPreset
from evidently import Dataset
from evidently import Report

bp = Blueprint('main', __name__)

# Set report directory and connection to itself
REPORT_DIR = 'application_files/reports'
WEBSERVICE_HOST = os.getenv('VISUALISATION_HOST'),
urlpart = f"{WEBSERVICE_HOST}"

@bp.route('/app/data_drift', methods=['POST'])
def data_drift():
    """
    Receives two dataframes, computes a data drift report, saves it,
    and returns a success message.
    """
    # Get data as bytes
    t0 = time.time()
    bytes = request.get_data(cache=True)

    t1 = time.time()
    print("Received data:", t1-t0)

    # Hash the input data and build possible report url
    data_hash = hashlib.sha256(bytes).hexdigest()
    report_filename = f'{data_hash}.html'
    report_path = os.path.join(REPORT_DIR, report_filename)

    t2 = time.time()
    print("Hashed data:", t2-t1)

    # Check if a report with this hash already exists
    if os.path.exists(report_path):
        print(f"Report for hash {data_hash} found in cache. Serving existing file.")

        # Return report if it exists
        report_url = f'http://{WEBSERVICE_HOST}/app/view_report/{report_filename}'
        t3 = time.time()
        print("Found report:", t3-t2)
        return jsonify({'report_url': report_url}), 200
    else:
        print(f"Report for hash {data_hash} not found. Generating new report.")

    t3 = time.time()
    print("Did not find report:", t3-t2)

    # Get data
    data = orjson.loads(bytes)
    if not data:
        return jsonify({"error": "Invalid or empty JSON payload"}), 400
    
    t4 = time.time()
    print("Loaded payload:", t4-t3)

    try:
        # Unpack payload
        ref_payload = data.get('reference_data')
        new_payload = data.get('current_data')

        # Check if the data is empty
        if not ref_payload or not new_payload:
            return jsonify({"error": "Missing reference_data or current_data"}), 400
        
        # Build dataframes from new and reference data
        ref_df = pl.DataFrame(ref_payload['data'])
        new_df = pl.DataFrame(new_payload['data'])

        t5 = time.time()
        print("Unpacked payload:", t5-t4)

        # Process New Data
        new_df.columns = new_df.columns
        
        # Move ID column to first
        try:
            new_id_col = new_payload['id_column'][0]
        except:
             new_id_col = None

        # Assign datetime columns
        new_dt_cols = new_payload['datetime_columns']

        t6 = time.time()
        print("Set new ID and datetime columns:", t6-t5) 

        # Process Reference Data
        ref_df.columns = ref_df.columns

        # Move ID column to first
        try:
            ref_id_col = ref_payload['id_column'][0]
        except:
            ref_id_col = None

        t7 = time.time()
        print("Set reference ID and datetime columns:", t7-t6) 

        # Use new datetime columns since they are going to be renamed regardless
        ref_dt_cols = new_payload['datetime_columns']
           
        # Move ID column to first position
        if ref_id_col != None and new_id_col != None:
            new_df.insert(0, new_id_col, new_df.pop(new_id_col))
            ref_df.insert(0, ref_id_col, ref_df.pop(ref_id_col))
        
        # Rename the columns
        rename_dict = dict(zip(ref_df.columns[:], new_df.columns[:]))
        ref_df = ref_df.rename(rename_dict)

        t8 = time.time()
        print("Renamed and reordered columns:", t8-t7) 

        # Process data and create dataset templates
        new_processed, data_def_new = ef.map_to_def_pl(new_df, new_id_col, new_dt_cols)
        
        t9 = time.time()
        print("Mapped new columns:", t9-t8) 

        # Create an evidently dataset for the new data
        new_dataset = Dataset.from_pandas(new_processed.to_pandas(), data_def_new)

        t10 = time.time()
        print("Created new dataset:", t10-t9) 

        # Process data and create dataset templates
        ref_processed, data_def_ref = ef.map_to_def_pl(ref_df, ref_id_col, ref_dt_cols)

        t11 = time.time()
        print("Mapped reference columns:", t11-t10) 

        # Create an evidently dataset for the reference data
        ref_dataset = Dataset.from_pandas(ref_processed.to_pandas(), data_def_ref)

        t12 = time.time()
        print("Created reference dataset:", t12-t11) 
        
        # wasserstein does not work for categorical data
        report = Report([DataDriftPreset(method="wasserstein")], include_tests="True")
        reps = report.run(ref_dataset, new_dataset)

        t13 = time.time()
        print("Calculated report:", t13-t12) 
        
        # Save the report to a file
        reps.save_html(report_path)

        # Build report URL
        report_url = f'http://{WEBSERVICE_HOST}/app/view_report/{report_filename}'
        t14 = time.time()
        print("Generated report URL:", t14-t13) 

        return jsonify({'report_url': report_url}), 200
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@bp.route('/app/view_report/<path:filename>', methods=['GET'])
def view_report(filename):
    """
    Renders the previously generated data drift report.
    """
    return send_from_directory(os.path.abspath(REPORT_DIR), filename)