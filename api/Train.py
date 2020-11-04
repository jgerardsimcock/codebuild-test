from Score import score_KRIs
from utils import transform_KRI_table, save_df_to_s3
from Models import ORM
from Time_Range import Time_Range


import pandas as pd
import re
import os
import logging


def train_ORM(
    connector_study_dict,
    time_step="30d",
    num_training_steps=10,
    num_testing_steps=4,
    params={
        "lag_periods": 2,
        "path_to_saved_models": "models",
        "warm_start": False,
        "path_to_saved_KRIs": None,
        "additional_KRIs": None,
        "s3": False, 
        "bucket": None
    },
):
    """
    Train an ORM from automatically computed KRI data.

    Args:
        connector_study_dict: (dict) a dictionary for data ingestion,  keys are SQL_connectors and values are lists of studies to train on
        time_step: (str) the length of time for computing KRIs
        num_training_steps: (int) the number of periods for computing KRI data
        num_testing_steps: (int) the number of hold-out periods for model validation

    Returns:
        3 pandas Dataframes, the first contains aggregate performance scores, the last two contain confusion matrices for the ORM and the ZKM, respectively
    """

    if "today" in params:
        today = params["today"]
    else:
        today = pd.to_datetime("today").floor(freq="D")

    if "lag_periods" in params:
        lag_periods = params["lag_periods"]
    else:
        lag_periods = 0

    if "warm_start" in params:
        warm_start = params["warm_start"]
    else:
        warm_start = False

    if "path_to_saved_KRIs" in params:
        path_to_saved_KRIs = params["path_to_saved_KRIs"]
    else:
        path_to_saved_KRIs = None

    if "additional_KRIs" in params:
        additional_KRIs = params["additional_KRIs"]
    else:
        additional_KRIs = None

    if "verbose" in params:
        verbose = params["verbose"]
    else:
        verbose = False

    if isinstance(time_step, str):
        time_step = pd.Timedelta(time_step)

    if isinstance(today, str):
        today = pd.Timestamp(today)
    
    if "s3" in params:
        s3 = params["s3"]
    else:
        s3 = False
    
    if "bucket" in params:
        bucket = params["bucket"]
    else:
        bucket = None

    today_string = str(today.strftime("%Y-%m-%d"))
    time_step_string = str(time_step.days) + "d"
    run_name = "_".join(
        [
            today_string,
            time_step_string,
            str(num_training_steps),
            str(num_testing_steps),
        ]
    )

    training_start_w_lag = today - (num_training_steps + 1 + lag_periods) * time_step
    training_start = today - (num_training_steps + 1) * time_step
    testing_start = today - (num_training_steps + 1 - num_testing_steps) * time_step

    dataframe_list = []

    logging.info("Scoring training KRIs")

    for connector, study_list in connector_study_dict.items():
        for study in study_list:
            KRIs = score_KRIs(connector, study, training_start_w_lag, today, time_step, params)
            KRI_data = transform_KRI_table(KRIs, params={"lag_periods": lag_periods})
            if isinstance(KRI_data, pd.DataFrame):
                KRI_data = KRI_data[KRI_data["Start_Time"] >= training_start]
                dataframe_list.append(KRI_data)

    all_data = pd.concat(dataframe_list)

    if additional_KRIs:
        try:
            all_data.merge(additional_KRIs, how="left", on=["Site", "Start_Time"])
        except ValueError:
            logging.warning(
                "Additional KRI merge failed, continuing with built-in KRIs"
            )

    if path_to_saved_KRIs:
        if s3 == False:
            all_data.to_csv(
                os.path.join(path_to_saved_KRIs, run_name + ".csv"), index=False
            )
        else:
            
            s3_file_path = os.path.join(path_to_saved_KRIs,  run_name + ".csv")
            save_df_to_s3(all_data, bucket=bucket, file_path=s3_file_path, index=False)
            

    training_bool = all_data["Start_Time"] < testing_start
    testing_bool = (~training_bool) & (all_data["Start_Time"] < today)

    training_data = all_data[training_bool]
    testing_data = all_data[testing_bool]

    KRI_train = training_data.drop(columns=["site_active_next_period"])
    KRI_test = testing_data.drop(columns=["site_active_next_period"])

    if "KRIs" in params:
        KRIs = params["KRIs"]
    else:
        KRIs = list(KRI_train.columns.drop(["Start_Time", "Site"]))

    target_train = training_data["site_active_next_period"]
    target_test = testing_data["site_active_next_period"]

    model = ORM(KRIs)

    if warm_start:
        if warm_start == "auto":
            if "path_to_saved_models" not in params:
                logging.error("Must provide path to models for auto warm start.")
                raise ValueError
            else:
                path_to_saved_models = params["path_to_saved_models"]
                potential_models = os.listdir(path_to_saved_models)
                format_checker = re.compile(r"\d{4}-\d{2}-\d{2}_\d+d_\d+_\d+.pkl")
                formatted_models = list(
                    filter(lambda x: format_checker.match(x), potential_models)
                )
                if len(formatted_models) >= 1:
                    most_recent_model = sorted(formatted_models)[-1]
                    path_to_model = os.path.join(
                        path_to_saved_models, most_recent_model
                    )
                    if s3 == True:
                        model.load_model_s3(bucket=bucket, model_name=path_to_model)
                    else:
                        model.load_model(path_to_model)
                    logging.info(f"Warm starting from {most_recent_model}")
                else:
                    logging.info("No models found, performing cold start")
        else:
            try:
                if s3 == True:
                    model.load_model_s3(bucket=bucket, model_name=warm_start)
                
                else:
                    model.load_model(warm_start)
            except FileNotFoundError:
                logging.info("Model not found, performing cold start.")
    else:
        logging.info("Cold starting model")

    model.fit(KRI_train, target_train, KRI_test, target_test)
    model.tune_threshholds(KRI_train, target_train)
    perf, mats = model.validate(KRI_test, target_test)

    if verbose:
        logging.info(perf)
        logging.info(mats["ORM"])
        logging.info(mats["ZKM"])

    path_to_saved_models = params["path_to_saved_models"]
    if s3 == True:
        model.save_model_s3(bucket=bucket, model_name= os.path.join(path_to_saved_models, run_name))
    else: 
        model.save_model(path=path_to_saved_models, name=run_name)
        logging.info(f"Saved model as {run_name} in {path_to_saved_models}")
    return perf, mats


def train_salvo(
    connector_study_dict,
    start_time,
    end_time,
    time_step="30d",
    num_training_steps=10,
    num_testing_steps=4,
    params={
        "lag_periods": 2,
        "path_to_saved_models": "models",
        "warm_start": False,
        "path_to_saved_KRIs": None,
        "additional_KRIs": None,
    },
):
    """
    Train a sequence of ORMs from automatically computed KRI data.

    Args:
        connector_study_dict: (dict) a dictionary for data ingestion,  keys are SQL_connectors and values are lists of studies to train on
        start_time: (str) the first training date
        end_time: (str) the final training date
        time_step: (str) the length of time for computing KRIs
        num_training_steps: (int) the number of periods for computing KRI data
        num_testing_steps: (int) the number of hold-out periods for model validation

    Returns:
        a list of aggregate model perfomaces over time, as well as a plot of F1 scores over the training window
    """
    time_range = Time_Range(start_time, end_time, time_step=time_step, params={})

    perfs = []
    for time in time_range:
        params["today"] = time
        params["verbose"] = False
        perf, mats = train_ORM(
            connector_study_dict,
            time_step=time_step,
            num_training_steps=num_training_steps,
            num_testing_steps=num_testing_steps,
            params=params,
        )

        perf["Training Date"] = time
        perfs.append(perf)

    perfs = pd.concat(perfs)
    perfs = perfs.reset_index()
    return perfs
