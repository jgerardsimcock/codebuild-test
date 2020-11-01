import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef as MCC
import s3fs
from io import StringIO
from Dataset import Dataset
from Time_Range import Time_Range
# from Connector import SQL_connector

import warnings
import ast
import logging
import os

warnings.filterwarnings("ignore")


def series_to_frame(series, Time_Range, name, params={}):
    """
    Reformat a pandas Series into a Dataframe with standardized schema

    Args:
        series: (pandas Series) the raw data
        Time_Range: (Time_Range) the time range of the data
        name: (str) the name of the data
        params: (dict) optional parameters

    Returns:
        a pandas DataFrame with standardized schema
    """
    if "value_name" in params:
        value_name = params["value_name"]
        temp = series.rename(value_name).to_frame()
    else:
        value_name = "Value"
        temp = series.rename(value_name).to_frame()

    temp = temp.reset_index()
    temp["Start_Time"] = Time_Range.start_time
    temp["End_Time"] = Time_Range.end_time
    temp["Type"] = name

    if "category" in params:
        temp["Category"] = params["category"]
        return temp[["Site", "Category", "Type", "Start_Time", "End_Time", value_name]]

    return temp[["Site", "Type", "Start_Time", "End_Time", value_name]]


def preprocess_data(raw_dataset, dm_dataset):
    """
    Perform initial cleanup on raw data:

    Args:
        raw_dataset: (Dataset) the input data to be cleaned
        dm_dataset: (Dataset) the demographics table for extracting site and country info

    Returns:
        a cleaned Dataset object
    """

    if isinstance(raw_dataset, Dataset) and isinstance(dm_dataset, Dataset):
        # Clone the input dataset
        out = Dataset(params=raw_dataset.__dict__)
        out.raw = False

        # Convert the start and end dates if necessary
        if hasattr(raw_dataset, "start_date"):
            out.dataset[out.start_date] = pd.to_datetime(out.dataset[out.start_date])
        if hasattr(raw_dataset, "end_date"):
            out.dataset[out.end_date] = pd.to_datetime(out.dataset[out.end_date])

        # Rename the subject column if necessary
        if hasattr(out, "subject_column"):
            out.dataset = out.dataset.rename(columns={out.subject_column: "USUBJID"})

        out.subject_column = "USUBJID"
        out.dataset["USUBJID"] = out.dataset["USUBJID"].apply(
            lambda s: s.replace("_", "-")
        )

        # Rename the study column if necessary
        if hasattr(out, "study_column"):
            out.dataset = out.dataset.rename(columns={out.study_column: "STUDYID"})

        out.study_column = "STUDYID"

        # Rename the site column if it exists in the dataset, create it otherwise
        if hasattr(out, "site_column"):
            out.dataset = out.dataset.rename(columns={out.site_column: "Site"})
        else:
            site_info = dm_dataset.dataset[["USUBJID", "SITEID", "COUNTRY"]]
            site_info = site_info.rename(columns={"SITEID": "Site"})
            out.dataset = out.dataset.merge(site_info, on="USUBJID")

        out.site_column = "Site"

        return out


def count_divide(df_row):
    """
    Divide the Value column in a dataframe by the active_subject_count:

    Args:
        df_row: a row of a pandas DataFrame:

    Returns:
        a float
    """
    count = df_row["Value"]
    num_subjects = df_row["active_subject_count"]

    if num_subjects == 0:
        return np.nan
    else:
        return count / num_subjects


def count_per_active_subject(count_dataset, EX_dataset):
    """
    Divide a count table by the number of active subjects to form a rate table.

    Args:
        count_dataset: (Dataset) the raw counts to be divided
        EX_dataset: (Dataset) the exposure dataset from which the subject population is derived

    Returns:
        a Dataset of the counts per active subject
    """

    if hasattr(count_dataset, "count") and count_dataset.count:
        if count_dataset.dataset.empty:
            return Dataset(
                dataset=count_dataset.dataset, params={"KRI": True, "raw": False}
            )
        else:
            start_time = count_dataset.dataset["Start_Time"].iloc[0]
            end_time = count_dataset.dataset["End_Time"].iloc[0]

            if isinstance(start_time, int):
                time_format = "int"
            else:
                time_format = "datetime"

            count_name = count_dataset.dataset["Type"].iloc[0]
            base_name = "_".join(count_name.split("_")[:-1])
            name = base_name + "_per_active_subject"

            active_subject_df = active_subject_count(
                EX_dataset, start_time, end_time
            ).dataset
            count_df = count_dataset.dataset[["Site", "Value"]]
            # count_df = count_df.reindex(EX_dataset.get_site_array(), fill_value=0)
            merged_table = count_df.merge(active_subject_df, on="Site", how="outer")
            merged_table = merged_table.fillna(0)
            merged_table["Value"] = merged_table.apply(count_divide, axis=1)
            merged_table["Type"] = name
            merged_table = merged_table.drop(columns=["active_subject_count"])
            merged_table = merged_table[~merged_table.Value.isna()].reset_index(
                drop=True
            )
            merged_table = merged_table[
                ["Site", "Type", "Start_Time", "End_Time", "Value"]
            ]
            out = Dataset(
                dataset=merged_table,
                params={"KRI": True, "raw": False, "time_format": time_format},
            )
            return out

    else:
        logging.error("Data must be count type")
        raise ValueError


def active_subject_count(EX_dataset, start_time, end_time, params={}):
    """
    Determine the number of active subjects at each site.

    Args:
        EX_dataset: (Dataset) the exposure dataset
        start_time: (int, pd.Timestamp) the start time of the window
        end_time: (int, pd.Timestamp) the end time of the window
        params: (dict) optional parameters:

    Returns:
        a Dataset of the active subject populations
    """
    if EX_dataset.raw:
        logging.error("Data must be preprocessed before counts")
        raise ValueError

    time_window = Time_Range(start_time, end_time)
    time_column_name = EX_dataset.get_time_column_name(time_window)

    left_bool = time_window.start_time <= EX_dataset.dataset[time_column_name]
    right_bool = EX_dataset.dataset[time_column_name] < time_window.end_time

    in_range = EX_dataset.dataset[left_bool & right_bool]
    site_counts = in_range.groupby("Site")["USUBJID"].nunique()
    site_list = EX_dataset.get_site_array()
    site_counts = site_counts.reindex(site_list, fill_value=0)

    if "name" in params:
        name = params["name"]
    else:
        name = "active_subject_count"

    site_count_frame = series_to_frame(
        site_counts, time_window, "active_subject_count", params={"value_name": name},
    )

    out = Dataset(
        dataset=site_count_frame,
        params={"count": True, "raw": False, "time_format": time_window.type},
    )
    return out


def identify_scoreable_value_names(numerical_dataset, threshhold):
    """
    Identify values with enough data to be used in outlier detection.

    Args:
        numerical_dataset: (Dataset) a numerical dataset
        threshold: (float) the maximum allowed frequency difference
    Returns:
        a list of tests which are reported frequently enough to be scored for outliers
    """
    if not isinstance(numerical_dataset, Dataset):
        logging.error("Data must be in dataset form")
        raise ValueError("Data must be in dataset form")

    if not hasattr(numerical_dataset, "value_name_column"):
        logging.error("Dataset must have value_name_column defined")
        raise ValueError

    value_name_column = numerical_dataset.value_name_column
    test_counts = numerical_dataset.dataset[value_name_column].value_counts()
    top_test_count = test_counts.iloc[0]
    reporting_freq_bool = ((top_test_count - test_counts) / top_test_count) < threshhold
    tests = list(test_counts[reporting_freq_bool].index)
    return tests


def align_numerical_table(
    numerical_dataset, scoreable_value_names, params={"drop_na": True}
):
    """
    Reformat a dataset into a configuration which the Outlier_Model class can accept.

    Args:
        numerical_dataset: (Dataset) a numerical dataset
        scoreable_value_names: (list) a list of values to be scored
        params: (dict) a dictionary of optional arguments

    Returns:
        a Dataset object
    """
    if not isinstance(numerical_dataset, Dataset):
        logging.error("Data must be in dataset form")
        raise ValueError

    if not hasattr(numerical_dataset, "value_name_column"):
        logging.error("Dataset must have value_name_column defined")
        raise ValueError

    if not hasattr(numerical_dataset, "value_column"):
        logging.error("Dataset must have value_column defined")
        raise ValueError

    value_name_column = numerical_dataset.value_name_column
    value_column = numerical_dataset.value_column
    day_column = numerical_dataset.start_day

    data = numerical_dataset.dataset
    data = data.rename(columns={day_column: "Time"})

    out = pd.DataFrame()

    for name in scoreable_value_names:

        in_name = data[data[value_name_column] == name]
        in_name = in_name[["USUBJID", "Site", "Time", value_column]].rename(
            columns={value_column: name}
        )

        if out.empty:
            out = in_name
        else:
            in_name = in_name.drop(columns=["Site"])
            out = out.merge(in_name, on=["USUBJID", "Time"])
            out = out.drop_duplicates()

        if params["drop_na"]:
            out = out[out.notna().all(axis=1)]

        out = out.drop_duplicates().reset_index(drop=True)
    return Dataset(dataset=out, params={"outlier_ready": True})


def aggregate_data_risk(dataset, level="site", params={}):
    """
    Aggregate outlier data by subject or site.

    Args:
        dataset: (Dataset) the data to be aggregated, must be scored by an Outlier_Model before aggregation
        level: (str) the level of aggregation, can be site (default) or subject
        params: (dict) additional parameters

    Returns:
        a dataset object with the data risk scores and number of samples per site or subject

    """
    if not isinstance(dataset, Dataset):
        logging.error("Data must be in Dataset format")
        raise ValueError
    if not hasattr(dataset, "outlier_scored"):
        logging.error("Outliers must be predicted before scoring")
        raise ValueError
    if dataset.outlier_scored:
        if level == "site":
            site_df = dataset.dataset.groupby("Site")["Outlier"].mean()
            site_df = site_df.rename("Data_Risk").to_frame()
            site_df["Samples"] = dataset.dataset.groupby("Site")["Outlier"].count()
            site_df = site_df.reset_index()
            return Dataset(dataset=site_df)
        elif level == "subject":
            sub_df = dataset.dataset.groupby("USUBJID")["Outlier"].mean()
            sub_df = sub_df.rename("Data_Risk").to_frame()
            sub_df["Samples"] = dataset.dataset.groupby("USUBJID")["Outlier"].count()
            sub_df = site_df.reset_index()
            return Dataset(dataset=sub_df)
        else:
            logging.error("Level must be site or subject")
            raise ValueError
            return None


def transform_KRI_table(KRI_table, params={}):
    """
    Re-align the KRI data into a column-rich form for modeling, also deduces the site activity.

    Args:
        KRI_table: (Dataset) the data to be re-aligned
        params:
            lag_periods: the number of past KRIs to include in the data, e.g. lag_periods=2 will include the current
                KRI scores, the KRI scores from the previous time-step, and the KRI scores from 2 time-steps ago

    Returns:
        a dataset object with the data ready for the ORM model

    """
    if not KRI_table.KRI:
        logging.error("Only accepts KRI Datasets")
        raise ValueError

    if "lag_periods" in params:
        lag_periods = params["lag_periods"]
    else:
        lag_periods = 0

    site_list = KRI_table.dataset["Site"].unique()
    KRI_list = KRI_table.dataset["Type"].unique()

    site_df_list = []

    for site in site_list:
        in_site = KRI_table.dataset[KRI_table.dataset.Site == site]

        subject_counts = in_site[(in_site.Type == "active_subject_count")].sort_values(
            "Start_Time"
        )
        if subject_counts["Value"].sum() == 0:
            pass
        else:
            active_clusters = compute_active_periods(subject_counts)

            cluster_list = []
            for active_cluster in active_clusters:
                left_bool = active_cluster["start"] <= in_site.Start_Time
                right_bool = in_site.Start_Time <= active_cluster["end"]
                in_active_cluster = in_site[left_bool & right_bool]

                cluster = pd.DataFrame()
                for KRI in KRI_list:
                    df = in_active_cluster[in_active_cluster.Type == KRI][
                        ["Start_Time", "Site", "Value"]
                    ].rename(columns={"Value": KRI})

                    if cluster.empty:
                        cluster = df
                    else:
                        df = df.drop(columns=["Site"])
                        cluster = cluster.merge(df, how="left", on="Start_Time")

                cluster = cluster.fillna(0)

                cluster = cluster.set_index("Start_Time")

                for period in range(1, lag_periods + 1):
                    new_names = {name: name + f"_{period}_lag" for name in KRI_list}
                    shift = cluster[KRI_list].shift(periods=period, fill_value=0)
                    cluster = cluster.merge(
                        shift.rename(columns=new_names),
                        how="left",
                        left_index=True,
                        right_index=True,
                    )

                cluster = cluster.reset_index()
                cluster = cluster.rename(columns={"index": "Start_Time"})

                cluster = score_active_sites(cluster)
                cluster_list.append(cluster)

            site_df_list.extend(cluster_list)
    if site_df_list:
        out = pd.concat(site_df_list)
        out = out.reset_index(drop=True)
        return out


def score_active_sites(cluster_data):
    """
    Computes the site activity for an active cluster:

    Args:
        cluster_data: (pd.DataFrame) a cluster of data where a site has well-defined rate KRIs

    Returns
        a pandas DataFrame with the "site_active_next_period" column scored
    """

    out = cluster_data.copy()
    out["site_active_next_period"] = 1
    out["site_active_next_period"].iloc[-1] = 0

    return out


def compute_active_periods(subject_counts):
    """
    Identify the time periods for which sites support an active population.

    Args:
        subject_counts: (pd.DataFrame) the active subject table for a study

    Returns:
        a dictionary of active time periods
    """
    active_clusters = []
    cluster_dict = {}

    for idx, row in subject_counts.iterrows():
        if row.Value > 0:
            if not cluster_dict:
                cluster_dict["start"] = row.Start_Time
        else:
            if cluster_dict:
                cluster_dict["end"] = row.Start_Time
                active_clusters.append(cluster_dict)
                cluster_dict = {}

    if "start" in cluster_dict and "end" not in cluster_dict:
        cluster_dict["end"] = row.Start_Time
        active_clusters.append(cluster_dict)

    return active_clusters


def MCC_at_threshhold(y_true, y_pred, threshhold):
    """
    Compute the Mathewson Correlation Coeffecient (MCC) for target data at a given threshhold.

    Args:
        y_true: (np.array) the ground truth target values
        y_pred: (np.array) the model output target values
        theshhold: (float) the theshhold at which to cut y_pred

    Returns:
        the MCC score (float)
    """
    y_pred_bool = y_pred > threshhold
    return MCC(y_true, y_pred_bool)


def load_connection_dict(config, server, param_set="DEFAULT"):
    """
    Generate a dictionary to pass to an SQL_connector.

    Args:
        config: (ConfigParser) a loaded configuration file
        server: (str) the server address for the SQL_connector
        param_set: the parameter set to use from the config file, default is "DEFAULT"

    Returns:
        a dictionary containing the information to generate an SQL_cpnnector
    """
    out = {}
    for attr in ["driver", "database", "user", "password", "trusted"]:
        if attr not in config[param_set]:
            raise ValueError(f"{attr} not found")
        else:
            out[attr] = config["DEFAULT"][attr]

    out["server"] = server
    return out


def load_connection_study_dict(config, param_set="DEFAULT"):
    """
    Generate a connector-study dictionary for model training.

    Args:
        config: (ConfigParser) a loaded configuration file
        param_set: the parameter set to use from the config file, default is "DEFAULT"

    Returns:
        a dictionary whose keys are SQL_connectors and values are lists of studies for model training

    """
    servers = ast.literal_eval(config[param_set]["servers"])
    server_study_dict = ast.literal_eval(config[param_set]["server_study_dict"])
    out = {}
    for server in servers:
        connection_dict = load_connection_dict(config, server=server)
        connector = SQL_connector(connection_dict)
        out[connector] = server_study_dict[server]

    return out


def load_csv_from_s3(bucket="", file_path=""):
    """
    Loads csv from bucket into DataFrame
    
    Args:
        bucket: (str) bucket name
        file_path: (str) file path name
        
    Returns:
        pd.DataFrame
    """
    s3 = s3fs.S3FileSystem(anon=True)
    with s3.open(f"{bucket}/{file_path}",'rb') as f:
        df = pd.read_csv(f)
    logging.info(f'Loading {file_path} from {bucket}')
    return df


def load_data_from_s3(bucket="", study=""):
    '''
    Helper function to load in study data
    
    Args:
        bucket: (str) bucket name
        file_path: (str) file path name
    Returns: 
        pd.Dataframe
    
    '''

    AE_DF = load_csv_from_s3(bucket, study + '/AE.csv')
    EX_DF = load_csv_from_s3(bucket, study + '/EX.csv')
    QY_DF = load_csv_from_s3(bucket, study + '/QY.csv')
    LB_DF = load_csv_from_s3(bucket, study + '/LB.csv')
    VS_DF = load_csv_from_s3(bucket, study + '/VS.csv')
    DM_DF = load_csv_from_s3(bucket, study + '/DM.csv')


    return AE_DF, EX_DF, QY_DF, LB_DF, VS_DF, DM_DF
    
def save_df_to_s3(df, bucket="", file_path="", index=False, float_format=None):
    """
    Saves df to bucket in csv format
    
    Args:
        df: (pd.DataFrame) 
        bucket: (str) bucket name
        file_path: (str) file path name
        
    Returns:
        None
    """
    
    s3 = s3fs.S3FileSystem(anon=True)
    with s3.open(f"{bucket}/{file_path}",'w') as f:
        df.to_csv(f, index=index, float_format=float_format)
    logging.info(f'Savinging {file_path} to {bucket}')
