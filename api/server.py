from utils import load_connection_study_dict
from Train import train_ORM

# from infer_risk import infer_risk

from Infer import infer_risk
from datetime import date
from typing import Optional, Set, Any, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import logging


import configparser
import sys


import socket
import numpy as np
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logging.info("Loading Model RBM")

app = FastAPI(
    title="eCS AI Architecture Docs",
    description="API docs for the updated eCS Architecture",
    openapi_url="/rbm/openapi.json",
    redoc_url="/rbm/redoc",
    docs_url="/rbm/docs",
    version="0.0.1",
)


@app.get("/rbm/")
def home():
    """
    Perform regular status checks to make sure the container is up and responding

    Returns
    =======
    str
    """
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print("Hostname :  ", host_name)
        print("IP : ", host_ip)
    except BaseException:
        print("Unable to get Hostname and IP")

    return "Hello World I am the RBM Model with a great change. Hostname is {} and host_ip is {}".format(
        host_name, host_ip
    )


@app.post("/rbm/train/")
def train(request: Dict[Any, Any]):
    """
    Params
    ======
    connector: connection string to db connection
    study: (str) the name of the study to score
    load_model: (str) the name of the model to load, if auto the most recently trained model is loaded
    time_step: (str) the length of time for computing KRIs
    num_training_steps: (str)
    num_testing_steps: (str)
    params: (dict)



    Returns
    =======
    List: Hostname and IP address


    """

    today = request.get("today", str(date.today()))
    model = request.get("model", "auto")
    time_step = request.get("time_step", "30d")
    params = request.get("params", {})

    if "num_training_steps" in request:
        num_training_steps = request.getint("num_training_steps")
    else:
        num_training_steps = 10

    if "num_testing_steps" in request:
        num_testing_steps = request.getint("num_testing_steps")
    else:
        num_testing_steps = 4

    perf, mats = train_ORM(
        request["connector_study_dict"],
        time_step=time_step,
        num_training_steps=num_training_steps,
        num_testing_steps=num_testing_steps,
        params=params,
    )

    logging.info(f"Performance is {perf}")

    # get the container ip
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)

    return host_name, host_ip


@app.post("/rbm/predict")
def predict(request: Dict[Any, Any]):
    """



    Params
    ======
    connector: connection string to db connection
    study: (str) the name of the study to score
    load_model: (str) the name of the model to load, if auto the most recently trained model is loaded
    time_step: (str) the length of time for computing KRIs


    Returns
    =======
    List: Hostname and IP address


    """

    today = request["params"].get("today", str(date.today()))
    model = request.get("model", "auto")
    time_step = request.get("time_step", "30d")

    scores, KRIs = infer_risk(
        request["connector"],
        request["study"],
        load_model=model,
        time_step=time_step,
        params=request["params"],
    )
    #####

    logging.info(f"KRIs computed as: {KRIs}")
    logging.info(f"SCORES computed as: {scores}")

    # get the container ip
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)

    return host_name, host_ip