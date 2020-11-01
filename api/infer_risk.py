from Connector import SQL_connector
from utils import load_connection_dict
from Infer import infer_risk
from datetime import date

import configparser
import sys
import logging

if len(sys.argv) == 1:
    ValueError("Must pass configuration name")
elif len(sys.argv) == 2:
    param_set = sys.argv[1]
else:
    logging.error("infer_risk only accepts one argument")


config = configparser.ConfigParser()
config.read("../config/config.ini")
server = config[param_set]["server"]

connector_dict = load_connection_dict(config, server)
connector = SQL_connector(connector_dict)

if "today" in config[param_set]:
    today = config[param_set]["today"]
else:
    today = str(date.today())

if "model" in config[param_set]:
    model = config[param_set]["model"]
else:
    model = "auto"

study = config[param_set]["study"]
time_step = config[param_set]["time_step"]
lag_periods = config[param_set].getint("lag_periods")
path_to_saved_models = config[param_set]["path_to_saved_models"]
path_to_saved_KRIs = config[param_set]["path_to_saved_KRIs"]
path_to_saved_risk_scores = config[param_set]["path_to_saved_risk_scores"]
max_precision = config[param_set]["max_precision"]

scores, KRIs = infer_risk(
    connector,
    study,
    load_model=model,
    time_step=time_step,
    params={
        "today": today,
        "explain": False,
        "lag_periods": lag_periods,
        "path_to_saved_models": path_to_saved_models,
        "path_to_saved_KRIs": path_to_saved_KRIs,
        "path_to_saved_risk_scores": path_to_saved_risk_scores,
        "max_precision": max_precision,
    },
)
