from Score import score_KRIs
from Dataset import Dataset
from utils import transform_KRI_table
from Models import ORM
from utils import save_df_to_s3
import s3fs
import pandas as pd
import os
import re
import logging


def infer_risk(
    connector,
    study,
    load_model="auto",
    time_step="30d",
    params={
        "lag_periods": 2,
        "path_to_saved_models": "models",
        "path_to_saved_KRIs": None,
        "path_to_saved_risk_scores": "risk",
        "s3_source_bucket": None,
        "s3_destination_bucket": None,
    },
):
    """
    Infer risk for a study using a trained ORM.

    Args:
        connector: (SQL_connector) PYODBC connector to the SQL data
        study: (str) the name of the study to score
        load_model: (str) the name of the model to load, if auto the most recently trained model is loaded
        time_step: (str) the length of time for computing KRIs

    Returns:
        2 pandas Dataframes, the first contains the KRIs, the second contains the risk scores
    """

    model = ORM([])

    if "path_to_saved_models" not in params:
        logging.error("Must provide path to models for auto warm start.")
        raise ValueError

    path_to_saved_models = params["path_to_saved_models"]
    source_bucket = params["s3_source_bucket"]
    destination_bucket = params["s3_destination_bucket"]

    if load_model == "auto":
        if source_bucket is None:
            potential_models = os.listdir(path_to_saved_models)

            format_checker = re.compile(r"\d{4}-\d{2}-\d{2}_\d+d_\d+_\d+.pkl")
            most_recent_model = sorted(filter(format_checker.match, potential_models))[
                -1
            ]
            path_to_model = os.path.join(path_to_saved_models, most_recent_model)

            model.load_model(path_to_model)
            logging.info(f"Loaded {most_recent_model}")
        else:
            fs = s3fs.S3FileSystem(anon=True)

            potential_models = fs.ls(source_bucket + "/" + path_to_saved_models)
            potential_models = [m.split("/")[-1] for m in potential_models]
            format_checker = re.compile(r"\d{4}-\d{2}-\d{2}_\d+d_\d+_\d+.pkl")
            most_recent_model = sorted(filter(format_checker.match, potential_models))[
                -1
            ]

            path_to_model = os.path.join(path_to_saved_models, most_recent_model)

            model.load_model_s3(bucket=source_bucket, model_name=path_to_model)
            logging.info(f"Loaded {most_recent_model} from {source_bucket}")

    else:
        try:

            path_to_model = os.path.join(path_to_saved_models, load_model)
            if source_bucket is None:
                model.load_model(path_to_model)
            else:
                model.load_model_s3(bucket=source_bucket, model_name=path_to_model)
        except FileNotFoundError:
            logging.error("Model not found")
            raise FileNotFoundError

    if "today" in params:
        today = params["today"]
        if isinstance(today, str):
            today = pd.to_datetime(today)

    else:
        today = pd.to_datetime("today").floor(freq="D")

    lag_periods = params.get("lag_periods", 0)
    path_to_saved_KRIs = params.get("path_to_saved_KRIs", None)
    path_to_saved_risk_scores = params.get("path_to_saved_risk_scores", None)
    max_precision = params.get("max_precision", 4)

    float_format = f"%.{max_precision}f"

    if isinstance(time_step, str):
        time_step = pd.Timedelta(time_step)

    KRI_start_time_w_lag = today - (lag_periods + 1) * time_step

    logging.info("Scoring KRIs")
    KRI_raw_data = score_KRIs(
        connector, study, KRI_start_time_w_lag, today, time_step, params
    )

    KRI_data = transform_KRI_table(KRI_raw_data, params=params)
    if not isinstance(KRI_data, pd.DataFrame):
        logging.warning("No sites to score.")
    else:

        KRI_data = KRI_data.loc[KRI_data["Start_Time"] == today]

        KRI_scores = KRI_data.drop(columns=["site_active_next_period"])

        if "KRIs" not in params:
            KRIs = list(KRI_scores.columns.drop(["Start_Time", "Site"]))
            params["KRIs"] = KRIs

        logging.info("Scoring risk")
        scores = model.score_risk(KRI_scores, params=params)
        scores = scores.reset_index(drop=True)

        scored_sites = scores["Site"].unique()
        no_population = []

        scores["Risk_Score"] = scores["Risk_Score"].clip(lower=0)

        today_string = str(today.strftime("%Y-%m-%d"))
        time_step_string = str(time_step.days) + "d"
        run_name = "_".join([today_string, time_step_string])

        if path_to_saved_KRIs:
            path_to_KRI = os.path.join(path_to_saved_KRIs, run_name + ".csv")
            # This line saves the KRI scores in the columnar format
            if destination_bucket is None:
                KRI_scores.to_csv(path_to_KRI, index=False, float_format=float_format)
            else:
                save_df_to_s3(
                    KRI_scores,
                    bucket=destination_bucket,
                    file_path=path_to_KRI,
                    index=False,
                    float_format=float_format,
                )
            # This line saves the KRI scores in the row format
            # KRI_raw_data.dataset.to_csv(path_to_KRI, index=False)

        if path_to_saved_risk_scores:
            path_to_risk = os.path.join(path_to_saved_risk_scores, run_name + ".csv")
            if destination_bucket is None:
                scores.to_csv(path_to_risk, index=False, float_format=float_format)
            else:
                # load data to s3
                save_df_to_s3(
                    scores,
                    bucket=destination_bucket,
                    file_path=path_to_risk,
                    index=False,
                    float_format=float_format,
                )

        return scores, KRI_raw_data