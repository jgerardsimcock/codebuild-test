import numpy as np

from Dataset import Dataset
from Time_Range import Time_Range
from utils import series_to_frame, count_per_active_subject

import warnings
import logging

warnings.filterwarnings("ignore")


def adverse_event_count(AE_dataset, start_time, end_time, params={}):
    """
    Count the number of adverse events which started in the given time window.

    Args:
        AE_dataset: (Dataset) the adverse events dataset
        start_time: (int, pd.Timestamp) the start time of the window
        end_time: (int, pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a Dataset of counts

    """
    if AE_dataset.raw:
        logging.error("Data must be preprocessed before counts")
        raise ValueError

    time_range = Time_Range(start_time, end_time)
    in_range = AE_dataset.get_data_from_time_range(time_range)

    if "adverse_event_type" in params:
        adverse_event_type = params["adverse_event_type"]
        if adverse_event_type == "s":
            ser_bool = in_range.AESER == "Y"
            in_type = in_range[ser_bool]
            adverse_event_name_prefix = "serious_"
        elif adverse_event_type == "r":
            rel_bool = in_range.AEREL == "RELATED"
            in_type = in_range[rel_bool]
            adverse_event_name_prefix = "related_"
        elif adverse_event_type == "sr":
            ser_bool = in_range.AESER == "Y"
            rel_bool = in_range.AEREL == "RELATED"
            both_bool = ser_bool & rel_bool
            in_type = in_range[both_bool]
            adverse_event_name_prefix = "serious_related_"
        else:
            in_type = in_range
    else:
        adverse_event_name_prefix = ""
        in_type = in_range

    site_counts = in_type.groupby("Site")["STUDYID"].count()

    if "site_list" in params:
        site_counts = site_counts.reindex(params["site_list"], fill_value=0)

    site_count_frame = series_to_frame(
        site_counts, time_range, adverse_event_name_prefix + "adverse_event_count"
    )

    out = Dataset(
        dataset=site_count_frame,
        params={"count": True, "raw": False, "time_format": time_range.type},
    )
    return out


def query_count(QY_dataset, start_time, end_time, params={}):
    """
    Count the number of queries which started in the given time window.

    Args:
        QY_dataset: (Dataset) the query dataset
        start_time: (pd.Timestamp) the start time of the window
        end_time: (pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a Dataset of counts

    """
    if QY_dataset.raw:
        logging.error("Data must be preprocessed before counts")
        raise ValueError

    time_range = Time_Range(start_time, end_time)

    if time_range.start_column != "start_date":
        logging.error("Query data can only be binned by date")
        raise ValueError

    in_range = QY_dataset.get_data_from_time_range(time_range)

    if "query_type" in params:
        query_type = params["query_type"]
        if query_type == "m":
            in_type_bool = in_range.QueryRecipient != "System to Site"
            in_type = in_range[in_type_bool]
            query_type_name_prefix = "manual_"
        elif query_type == "a":
            in_type_bool = in_range.QueryRecipient == "System to Site"
            in_type = in_range[in_type_bool]
            query_type_name_prefix = "automatic_"
        else:
            in_type = in_range
    else:
        query_type_name_prefix = ""
        in_type = in_range

    site_counts = in_type.groupby("Site")["STUDYID"].count()

    if "site_list" in params:
        site_counts = site_counts.reindex(params["site_list"], fill_value=0)

    site_count_frame = series_to_frame(
        site_counts, time_range, query_type_name_prefix + "query_count"
    )
    out = Dataset(
        dataset=site_count_frame,
        params={"count": True, "raw": False, "time_format": time_range.type},
    )
    return out


def hanging_query_count(QY_dataset, start_time, end_time, params={}):
    """
    Count the number of hanging queries which started in the given time window.

    Args:
        QY_dataset: (Dataset) the query dataset
        start_time: (pd.Timestamp) the start time of the window
        end_time: (pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a Dataset of counts

    """
    if QY_dataset.raw:
        logging.error("Data must be preprocessed before counts")
        raise ValueError

    time_range = Time_Range(start_time, end_time)

    if time_range.start_column != "start_date":
        logging.error("Query data can only be binned by date")
        raise ValueError

    left_bool = QY_dataset.dataset["OpenDate"] <= time_range.start_time
    right_bool = QY_dataset.dataset["CloseDate"] > time_range.end_time

    in_range = QY_dataset.dataset[left_bool & right_bool]

    site_counts = in_range.groupby("Site")["STUDYID"].count()

    if "site_list" in params:
        site_counts = site_counts.reindex(params["site_list"], fill_value=0)

    site_count_frame = series_to_frame(site_counts, time_range, "hanging_query_count")
    out = Dataset(
        dataset=site_count_frame,
        params={"count": True, "raw": False, "time_format": time_range.type},
    )
    return out


def lab_count(LB_dataset, start_time, end_time, params={}):
    """
    Count the number of labs which started in the given time window.

    Args:
        LB_dataset: (Dataset) the labs dataset
        start_time: (int, pd.Timestamp) the start time of the window
        end_time: (int, pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a Dataset of counts

    """
    if LB_dataset.raw:
        logging.error("Data must be preprocessed before counts")
        raise ValueError

    time_range = Time_Range(start_time, end_time)
    in_range = LB_dataset.get_data_from_time_range(time_range)

    if "lab_type" in params:
        lab_type = params["lab_type"]
        if lab_type == "u":
            unsched_bool = in_range.VISITNUM.apply(lambda x: x % 1) != 0
            in_type = in_range[unsched_bool]
            lab_type_name_prefix = "unscheduled_"
        elif lab_type == "m":
            miss_bool = in_range.LBSTAT == "NOT DONE"
            in_type = in_range[miss_bool]
            lab_type_name_prefix = "missed_"
        else:
            in_type = in_range
    else:
        lab_type_name_prefix = ""
        in_type = in_range

    site_counts = in_type.groupby("Site")["STUDYID"].count()
    if "site_list" in params:
        site_counts = site_counts.reindex(params["site_list"], fill_value=0)

    site_count_frame = series_to_frame(
        site_counts, time_range, lab_type_name_prefix + "lab_count"
    )
    out = Dataset(
        dataset=site_count_frame,
        params={"count": True, "raw": False, "time_format": time_range.type},
    )
    return out


def adverse_event_rates(AE_dataset, EX_dataset, start_time, end_time, params={}):
    """
    Compute the number of adverse events (baseline, serious, relevant, and
        serious relevant) per active subject in the given time window.

    Args:
        AE_dataset: (Dataset) the adverse events dataset
        EX_dataset: (Dataset) the exposure dataset for deriving subject counts
        start_time: (int, pd.Timestamp) the start time of the window
        end_time: (int, pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a list of Datasets of counts per active subject

    """

    AE_counts = adverse_event_count(AE_dataset, start_time, end_time)
    params.update({"adverse_event_type": "s"})
    SAE_counts = adverse_event_count(
        AE_dataset,
        start_time,
        end_time,
        params=params,
    )
    params.update({"adverse_event_type": "r"})
    RAE_counts = adverse_event_count(AE_dataset, start_time, end_time, params=params)

    AE_rates = count_per_active_subject(AE_counts, EX_dataset)
    SAE_rates = count_per_active_subject(SAE_counts, EX_dataset)
    RAE_rates = count_per_active_subject(RAE_counts, EX_dataset)

    out = [AE_rates, SAE_rates, RAE_rates]
    return out


def query_rates(QY_dataset, EX_dataset, start_time, end_time, params={}):
    """
    Compute the number of queries (baseline, automatically generated
    and manually generated) per active subject in the given time window.

    Args:
        QY_dataset: (Dataset) the query dataset
        EX_dataset: (Dataset) the exposure dataset for deriving subject counts
        start_time: (pd.Timestamp) the start time of the window
        end_time: (pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a list of Datasets of counts per active subject

    """

    QY_counts = query_count(QY_dataset, start_time, end_time, params=params)
    params.update({"query_type": "m"})
    MQY_counts = query_count(QY_dataset, start_time, end_time, params=params)
    params.update({"query_type": "a"})
    AQY_counts = query_count(QY_dataset, start_time, end_time, params=params)
    HQY_counts = hanging_query_count(QY_dataset, start_time, end_time)

    QY_rates = count_per_active_subject(QY_counts, EX_dataset)
    MQY_rates = count_per_active_subject(MQY_counts, EX_dataset)
    AQY_rates = count_per_active_subject(AQY_counts, EX_dataset)
    HQY_rates = count_per_active_subject(HQY_counts, EX_dataset)

    out = [QY_rates, MQY_rates, AQY_rates, HQY_rates]
    return out


def lab_rates(LB_dataset, EX_dataset, start_time, end_time, params={}):
    """
    Compute the number of labs (baseline, missed, and unscheduled)
        per active subject in the given time window.

    Args:
        LB_dataset: (Dataset) the lab dataset
        EX_dataset: (Dataset) the exposure dataset for deriving subject counts
        start_time: (int, pd.Timestamp) the start time of the window
        end_time: (int, pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a list of Datasets of counts per active subject

    """
    LB_counts = lab_count(LB_dataset, start_time, end_time, params=params)
    params.update({"lab_type": "u"})
    ULB_counts = lab_count(LB_dataset, start_time, end_time, params=params)
    params.update({"lab_type": "m"})
    MLB_counts = lab_count(LB_dataset, start_time, end_time, params=params)

    LB_rates = count_per_active_subject(LB_counts, EX_dataset)
    ULB_rates = count_per_active_subject(ULB_counts, EX_dataset)
    MLB_rates = count_per_active_subject(MLB_counts, EX_dataset)

    out = [LB_rates, ULB_rates, MLB_rates]
    return out


def missed_dose_count(EX_dataset, start_time, end_time, params={}):
    """
    Count the number of missed doses which started in the given time window.

    Args:
        EX_dataset: (Dataset) the exposure dataset
        start_time: (int, pd.Timestamp) the start time of the window
        end_time: (int, pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a Dataset of counts

    """
    if EX_dataset.raw:
        logging.error("Data must be preprocessed before counts")
        raise ValueError

    time_range = Time_Range(start_time, end_time)
    in_range = EX_dataset.get_data_from_time_range(time_range)

    in_type_bool = in_range["EXDOSE"] == 0
    in_type = in_range[in_type_bool]

    site_counts = in_type.groupby("Site")["STUDYID"].count()
    site_counts = site_counts.reindex(EX_dataset.get_site_array(), fill_value=0)

    site_count_frame = series_to_frame(site_counts, time_range, "missed_dose_count")
    out = Dataset(
        dataset=site_count_frame,
        params={"count": True, "raw": False, "time_format": time_range.type},
    )
    return out


def missed_dose_rate(EX_dataset, start_time, end_time, params={}):
    """
    Compute the number of missed doses per active subject in the given time window.

    Args:
        EX_dataset: (Dataset) the exposure dataset for deriving subject counts
        start_time: (int, pd.Timestamp) the start time of the window
        end_time: (int, pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a list of Datasets of counts per active subject

    """
    miss_IP_count = missed_dose_count(EX_dataset, start_time, end_time, params=params)
    miss_IP_rate = count_per_active_subject(miss_IP_count, EX_dataset)
    return miss_IP_rate


def dosage_variance(EX_dataset, start_time, end_time, params={}):
    """
    Compute the number of dosage variance per site in the given time window.

    Args:
        EX_dataset: (Dataset) the exposure dataset for deriving subject counts
        start_time: (int, pd.Timestamp) the start time of the window
        end_time: (int, pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a Dataset of dosage variances

    """
    if EX_dataset.raw:
        logging.error("Data must be preprocessed before counts")
        raise ValueError

    time_range = Time_Range(start_time, end_time)
    in_range = EX_dataset.get_data_from_time_range(time_range)

    patient_var_group = in_range.groupby(["Site", "USUBJID"])
    patient_vars = patient_var_group.EXDOSE.agg("std").fillna(0)
    site_vars = patient_vars.groupby("Site").agg("mean")
    site_vars = site_vars.reindex(EX_dataset.get_site_array(), fill_value=0)

    site_var_frame = series_to_frame(site_vars, time_range, "dosage_variance")
    out = Dataset(
        dataset=site_var_frame,
        params={"count": True, "raw": False, "time_format": time_range.type},
    )
    return out


def query_response_time(QY_dataset, start_time, end_time, params={}):
    """
    Compute the query response time in the given time window.

    Args:
        QY_dataset: (Dataset) the query dataset
        start_time: (pd.Timestamp) the start time of the window
        end_time: (pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a list of Datasets of response times

    """
    if QY_dataset.raw:
        logging.error("Data must be preprocessed before counts")
        raise ValueError

    time_range = Time_Range(start_time, end_time)

    if time_range.start_column != "start_date":
        logging.error("Query data can only be binned by date")
        raise ValueError

    in_range = QY_dataset.get_data_from_time_range(time_range)

    if "query_type" in params:
        query_type = params["query_type"]
        if query_type == "m":
            in_type_bool = in_range.QueryRecipient != "System to Site"
            in_type = in_range[in_type_bool]
            query_type_name_prefix = "manual_"
        elif query_type == "a":
            in_type_bool = in_range.QueryRecipient == "System to Site"
            in_type = in_range[in_type_bool]
            query_type_name_prefix = "automatic_"
        else:
            in_type = in_range
    else:
        query_type_name_prefix = ""
        in_type = in_range

    closed_queries_bool = in_type.CloseDate.notna()
    in_type = in_type[closed_queries_bool]

    in_type["Response_Time"] = in_type["CloseDate"] - in_type["OpenDate"]
    in_type["Response_Time_Hours"] = in_type["Response_Time"].apply(
        lambda x: float(x.seconds / 3600)
    )

    if not in_type.empty:
        site_means = in_type.groupby("Site")["Response_Time_Hours"].mean()

        if "site_list" in params:
            site_means = site_means.reindex(params["site_list"], fill_value=np.nan)

        site_mean_frame = series_to_frame(
            site_means, time_range, query_type_name_prefix + "query_response_time"
        )
        out = Dataset(
            dataset=site_mean_frame.dropna(),
            params={"count": True, "raw": False, "time_format": time_range.type},
        )
        return out


def query_response_times(QY_dataset, start_time, end_time, params={}):
    """
    Compute the query response times (baseline, automatically generated
    and manually generated) in the given time window.

    Args:
        QY_dataset: (Dataset) the query dataset
        start_time: (pd.Timestamp) the start time of the window
        end_time: (pd.Timestamp) the ends time of the window
        params: (dict) optional parameters

    Returns:
        a list of Datasets of query response times

    """
    QY_response_time = query_response_time(
        QY_dataset, start_time, end_time, params=params
    )
    params.update({"query_type": "m"})
    MQY_response_time = query_response_time(
        QY_dataset, start_time, end_time, params=params
    )
    params.update({"query_type": "a"})
    AQY_response_time = query_response_time(
        QY_dataset, start_time, end_time, params=params
    )

    out = [QY_response_time, MQY_response_time, AQY_response_time]
    return out