from typing import Optional, Set
from fastapi import FastAPI
from pydantic import BaseModel
import logging 
# import RBM
# import DataReview
# import model 
# import model3

# from train import *
import socket
import numpy as np
from datetime import datetime



logging.basicConfig(level=logging.INFO)
logging.info('Loading Model A')

app = FastAPI(title="eCS AI Architecture Docs",
                description="API docs for the updated eCS Architecture",
                version="0.0.1",)



# class Item(BaseModel):
#     name: str
#     description: Optional[str] = None
#     price: float
#     tax: Optional[float] = None
#     tags: Set[str] = []


@app.get("/")
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


    return "Hello World I am Model A. Hostname is {} and host_ip is {}".format(host_name, host_ip)

@app.get("/train/")
def train(model_id: str, 
            model_source: str, 
            data_source: str, 
            model_destination: str, 
            model_details: Optional[str] = None):

    '''
    Performs training on model given a dataset location and saves model outputs to a model destination


    Parameters
    ==========
    model_id: str 
        unique name for model.
    model_source: str 
        location/path to pull model file.
    data_source: str 
        location/path to pull training data.
    model_destination: str 
        location/path to store trained model file.
    model_details: str 
        additional details that may be relevant.


    Returns
    =======
    item: dict of model_id, model_source, model_destination, model_details

    '''


    item = {

        'model_id': model_id,
        'model_source': model_source,
        'model_destination': model_destination,
        'model_details': model_details
        }


    # current date and time
    now = datetime.now()
    TRAINING_TIMESTAMP = datetime.timestamp(now)
    MODEL_ACCURACY = np.random.random_sample()


    logging.info('TRAINING_TIMESTAMP: {}'.format(TRAINING_TIMESTAMP))
    logging.info('MODEL_ACCURACY: {}'.format(MODEL_ACCURACY))
    logging.info('MODEL_DESTINATION: {}'.format(model_destination))
    logging.info(model_id, model_source, model_destination, model_details)


 
    return item


@app.get("/predict/")
def predict(model_id: str, 
            model_source: str, 
             model_details: Optional[str] = None):
    '''

    Parameters
    ==========
    model_id: str 
        unique name for model.
    model_source: str 
        location/path to pull model file.
    model_details: str 
        additional details that may be relevant.


    Returns
    =======
    item: dict 
        model_id, model_source, model_details
    '''


    item = {
        'model_id': model_id,
        'model_source': model_source,
        'model_details': model_details
        }

    logging.info(f'Pulling {model_id} from {model_source} for prediction')
    logging.info('Running prediction...')
    logging.info('Prediction is...')

    return item


