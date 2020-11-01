import pandas as pd
from Dataset import Dataset
from Time_Range import Time_Range
from utils import load_data_from_s3

from utils import (
    preprocess_data,
    active_subject_count,
    identify_scoreable_value_names,
    align_numerical_table,
    aggregate_data_risk,
)
from Counts import (
    adverse_event_rates,
    query_rates,
    lab_rates,
    missed_dose_rate,
    query_response_times,
    dosage_variance,
)
from Models import Outlier_Model

import warnings

warnings.filterwarnings("ignore")


def score_KRIs(connector, study, start_time, end_time, time_step, params={}):
    """
    Score all KRIs in a given time window.

    Args:
        connector:
        study:

        start_time: (int, pd.Timestamp) the start time of the window
        end_time: (int, pd.Timestamp) the end time of the window
        time_step: (int, pd.Timedelta) the step for traversing the time window

    Returns:
        a Dataset containing all the KRI scores
    """
    AE_DF, EX_DF, QY_DF, LB_DF, VS_DF, DM_DF = load_data_from_s3(connector, study)



    AE = Dataset(
        dataset=AE_DF,
        study=study,
        table="AE",
        params={"start_day": "AESTDY", "start_date": "AESTDTC"},
    )

    EX = Dataset(
        dataset=EX_DF,
        study=study,
        table="EX",
        params={"start_day": "EXSTDY", "start_date": "EXSTDTC"},
    )

    QY = Dataset(
        dataset=QY_DF,
        study=study + "_ODM_Mapped#",
        table="Query",
        params={
            "start_date": "OpenDate",
            "end_date": "CloseDate",
            "subject_column": "UniqueSubjectID",
            "study_column": "StudyOID",
        },
    )

    LB = Dataset(
        dataset=LB_DF,
        study=study,
        table="LB",
        params={"start_day": "LBDY", "start_date": "LBDTC"},
    )

    VS = Dataset(
        dataset=VS_DF,
        study=study,
        table="VS",
        params={
            "start_day": "VSDY",
            "start_date": "VSDTC",
            "value_name_column": "VSTEST",
        },
    )

    DM = Dataset(
        dataset=DM_DF, study=study, table="DM", params={"site_column": "SITEID"}
    )

    AE = preprocess_data(AE, DM)
    EX = preprocess_data(EX, DM)
    QY = preprocess_data(QY, DM)
    LB = preprocess_data(LB, DM)
    VS = preprocess_data(VS, DM)

    time_range = Time_Range(start_time, end_time, time_step=time_step, params={})

    site_list = EX.get_site_array()

    dataset_list = []
    for time in time_range:
        stepped_time = time + time_range.time_step

        AE_rates = adverse_event_rates(AE, EX, time, stepped_time, params={})
        dataset_list.extend(AE_rates)

#         if time_range.type == "datetime":
#             QY_rates = query_rates(
#                 QY, EX, time, stepped_time, params={"site_list": site_list}
#             )
#             dataset_list.extend(QY_rates)

#             QY_response_times = query_response_times(
#                 QY, time, stepped_time, params={"site_list": site_list}
#             )
#             dataset_list.extend(QY_response_times)

        LB_rates = lab_rates(LB, EX, time, stepped_time, params={})
        dataset_list.extend(LB_rates)

        missed_IP_rate = missed_dose_rate(EX, time, stepped_time)
        dataset_list.append(missed_IP_rate)

        IP_variance = dosage_variance(EX, time, stepped_time)
        dataset_list.append(IP_variance)

        subject_count = active_subject_count(
            EX, time, stepped_time, params={"name": "Value"}
        )
        dataset_list.append(subject_count)

        data_risk = score_data_risk(
            connector,
            study,
            level="site",
            params={
                "start_time": time,
                "end_time": stepped_time,
                "KRI_format": True,
                "lab_data": LB,
                "vital_data": VS,
            },
        )
        dataset_list.append(data_risk)

    dataset_list_clean = list(filter(None, dataset_list))
    dataframe_list = [dataset.dataset for dataset in dataset_list_clean]
    dataframe = pd.concat(dataframe_list).reset_index(drop=True)
    out = Dataset(
        dataset=dataframe,
        params={"KRI": True, "raw": False, "time_format": time_range.type},
    )

    return out


def score_data_risk(connector, study, level="site", params={}):

    if "lab_data" in params:
        LB = params["lab_data"]
        LB.value_name_column = "LBTEST"
    else:
        LB = Dataset(
            connector=connector,
            study=study,
            table="LB",
            params={
                "start_day": "LBDY",
                "start_date": "LBDTC",
                "value_name_column": "LBTEST",
            },
        )

        DM = Dataset(
            connector=connector,
            study=study,
            table="DM",
            params={"site_column": "SITEID"},
        )
        LB = preprocess_data(LB, DM)
    if "vital_data" in params:
        VS = params["vital_data"]
        VS.value_name_column = "VSTEST"
    else:
        VS = Dataset(
            connector=connector,
            study=study,
            table="VS",
            params={
                "start_day": "VSDY",
                "start_date": "VSDTC",
                "value_name_column": "VSTEST",
            },
        )
        DM = Dataset(
            connector=connector,
            study=study,
            table="DM",
            params={"site_column": "SITEID"},
        )
        VS = preprocess_data(VS, DM)

    LB_df = LB.dataset
    VS_df = VS.dataset

    LB_df_numeric = LB_df[
        (LB_df.LBSTRESN.notna())
        & (LB_df.LBSTRESU.notna())
        & (LB_df.LBCAT == "CHEMISTRY")
    ]
    VS_df_numeric = VS_df[(VS_df.VSSTRESN.notna()) & (VS_df.VSSTRESU.notna())]

    if "end_time" in params:
        end_time = params["end_time"]
        LB_df_numeric = LB_df_numeric[LB_df_numeric["LBDTC"] <= end_time]
        VS_df_numeric = VS_df_numeric[VS_df_numeric["VSDTC"] <= end_time]

    LB_numerical = Dataset(
        dataset=LB_df_numeric,
        params={
            "start_day": "LBDY",
            "start_date": "LBDTC",
            "value_name_column": "LBTEST",
            "value_column": "LBSTRESN",
            "unit_column": "LBSTRESU",
        },
    )
    VS_numerical = Dataset(
        dataset=VS_df_numeric,
        params={
            "start_day": "VSDY",
            "start_date": "VSDTC",
            "value_name_column": "VSTEST",
            "value_column": "VSSTRESN",
            "unit_column": "VSSTRESU",
        },
    )

    if "threshhold" in params:
        threshhold = params["threshhold"]
    else:
        threshhold = 0.1

    LB_scoreable_tests = identify_scoreable_value_names(LB_numerical, threshhold)
    VS_scoreable_tests = identify_scoreable_value_names(VS_numerical, threshhold)

    VS_dataset = align_numerical_table(VS_numerical, VS_scoreable_tests)
    LB_dataset = align_numerical_table(LB_numerical, LB_scoreable_tests)

    VS_outlier_model = Outlier_Model()
    LB_outlier_model = Outlier_Model()
    # ################################
    # print(LB_dataset.dataset.head())
    # ################################
    scored_VS = VS_outlier_model.fit_predict(VS_dataset)
    scored_LB = LB_outlier_model.fit_predict(LB_dataset)

    agg_VS = aggregate_data_risk(scored_VS, level=level)
    agg_LB = aggregate_data_risk(scored_LB, level=level)

    agg_VS = agg_VS.dataset
    agg_LB = agg_LB.dataset

    agg_VS["Type"] = "vital_sign_data_risk"
    agg_LB["Type"] = "lab_data_risk"

    if "start_time" in params:
        start_time = params["start_time"]
        agg_VS["Start_Time"] = start_time
        agg_LB["Start_Time"] = start_time

    if "end_time" in params:
        end_time = params["end_time"]
        agg_VS["End_Time"] = end_time
        agg_LB["End_Time"] = end_time

    if "KRI_format" in params:
        if params["KRI_format"]:
            agg_VS = agg_VS.rename(columns={"Data_Risk": "Value"})
            agg_LB = agg_LB.rename(columns={"Data_Risk": "Value"})

            agg_VS = agg_VS.drop(columns=["Samples"])
            agg_LB = agg_LB.drop(columns=["Samples"])

    out = Dataset(dataset=pd.concat([agg_VS, agg_LB]))
    return out
