from flask import Blueprint, render_template

# custom funcs
from app.funcs import evidently_funcs as ef
from app.funcs import mlflow_funcs as mf

# Data management imports
import pandas as pd

import os

bp = Blueprint('main', __name__)

@bp.route('/', methods=("POST", "GET"))
def index():
    print("Hier")
    df = pd.read_parquet('data_nonne_enriched_full_dropped_na.parquet').tail(2)
    print("Und da")
    print(os.getcwd())
    return render_template('index.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
    