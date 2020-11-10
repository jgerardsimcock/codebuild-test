from utils import load_connection_study_dict
from Train import train_ORM
from datetime import date

import configparser
import sys
import logging

if len(sys.argv) == 1:
    param_set = "DEFAULT"
elif len(sys.argv) == 2:
    param_set = sys.argv[1]
else:
    logging.error("train_ORM only accepts one argument")
    raise ValueError("train_ORM only accepts one argument")


config = configparser.ConfigParser()
config.read("../config/config.ini")

connector_study_dict = load_connection_study_dict(config, param_set=param_set)

if "today" in config[param_set]:
    today = config[param_set]["today"]
else:
    today = str(date.today())

if "model" in config[param_set]:
    model = config[param_set]["model"]
else:
    model = "auto"


time_step = config[param_set]["time_step"]
lag_periods = config[param_set].getint("lag_periods")
num_training_steps = config[param_set].getint("num_training_steps")
num_testing_steps = config[param_set].getint("num_testing_steps")
path_to_saved_models = config[param_set]["path_to_saved_models"]

perf, mats = train_ORM(
    connector_study_dict,
    time_step=time_step,
    num_training_steps=num_training_steps,
    num_testing_steps=num_testing_steps,
    params={
        "today": today,
        "warm_start": model,
        "lag_periods": lag_periods,
        "path_to_saved_models": path_to_saved_models,
        "verbose": True,
    },
)

print(perf)