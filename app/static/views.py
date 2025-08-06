from flask import Blueprint, render_template

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

@bp.route('/app/data_drift', methods=("POST", "GET"))
def index():
    # df = pd.read_parquet('data_nonne_enriched_full_dropped_na.parquet').tail(20)

    # a, b = train_test_split(df, test_size = 0.5)
    # a_processed, data_def_a = ef.map_to_def(a.iloc[:10,:], "Unnamed: 0", "datum")
    # b_processed, data_def_b = ef.map_to_def(b.iloc[:10,:], "Unnamed: 0", "datum")

    # dataset_1 = Dataset.from_pandas(a_processed, data_def_a)
    # dataset_2 = Dataset.from_pandas(b_processed, data_def_b)

    # report = Report([
    #     DataDriftPreset(method="wasserstein")
    # ],
    # include_tests="True")

    # rep = report.run(reference_data=dataset_1, current_data=dataset_2)
    # rep.save_html('app/templates/report.html')

    return render_template('report.html')