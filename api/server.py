from utils import load_connection_study_dict
from Train import train_ORM
# from infer_risk import infer_risk

from Infer import infer_risk
from datetime import date
from typing import Optional, Set, Any, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import logging 


import configparser
import sys


import socket
import numpy as np
from datetime import datetime



logging.basicConfig(level=logging.INFO)
logging.info('Loading Model RBM')

app = FastAPI(title="eCS AI Architecture Docs",
                description="API docs for the updated eCS Architecture",
                openapi_url='/rbm/openapi.json',
                redoc_url='/rbm/redoc',
                docs_url="/rbm/docs",
                version="0.0.1",)



@app.get('/rbm/')
def home():

    '''
    Perform regular status checks to make sure the container is up and responding

    Returns
    =======
    str
    '''
    try: 
        host_name = socket.gethostname() 
        host_ip = socket.gethostbyname(host_name) 
        print("Hostname :  ",host_name) 
        print("IP : ",host_ip) 
    except: 
        print("Unable to get Hostname and IP") 


    return "Hello World I am the RBM Model. Hostname is {} and host_ip is {}".format(host_name, host_ip)


@app.post('/rbm/train')
def train(request: Dict[Any, Any]):
    '''
    

    '''

    if "today" in request:
        today = request["today"]
    else:
        today = str(date.today())

    if "model" in request:
        model = request["model"]
    else:
        model = "auto"

    if "time_step" in request:
        time_step = request["time_step"]
    else: 
        time_step = "30d"

    if "num_training_steps" in request:
        num_training_steps = request.getint("num_training_steps")
    else: 
        num_training_steps = 10

    if "num_testing_steps" in request:
        num_testing_steps = request.getint("num_testing_steps")
    else:
        num_testing_steps = 4

    if "params" in request:
        params = request['params']
    else: None


        

    connector_study_dict = request["connector_study_dict"]



    perf, mats = train_ORM(
        connector_study_dict,
        time_step=time_step,
        num_training_steps=num_training_steps,
        num_testing_steps=num_testing_steps,
        params=params
     
        )

    logging.info(f'Perfomance is {perf}')

    #get the container ip
    host_name = socket.gethostname() 
    host_ip = socket.gethostbyname(host_name) 

    return host_name, host_ip
    



@app.post('/rbm/predict')
def predict(request: Dict[Any, Any]):
    '''


    Params
    ======
    connector: connection string to db connection
    study: (str) the name of the study to score
    load_model: (str) the name of the model to load, if auto the most recently trained model is loaded
    time_step: (str) the length of time for computing KRIs



    '''




    connector = request['connector']
    study = request["study"]

    #kw args
    if "model" in request:
        model = request['model']
    else:
        model = "auto"
    if "time_step" in request:
        time_step = request["time_step"]
    else:
        time_step = "30d"
    
    if "today" in request["params"]:
        today = request["params"]["today"]
    else:
        request["params"]["today"] =  str(date.today())

    params = request["params"]


    scores, KRIs = infer_risk(
        connector,
        study,
        load_model=model,
        time_step=time_step,
        params=params
        )


    logging.info(f'KRIs computed as: {KRIs}')
    logging.info(f'SCORES computed as: {scores}')


    #get the container ip
    host_name = socket.gethostname() 
    host_ip = socket.gethostbyname(host_name) 

    return host_name, host_ip







