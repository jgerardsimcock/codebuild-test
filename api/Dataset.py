import pandas as pd
import logging

import warnings

warnings.filterwarnings("ignore")


class Dataset:
    """
    A dataset object.

    Class Variables:
        path: (str) the input path to the data directory
        dataset: (pandas DataFrame) the data
        raw: (bool) a boolean representing whether or not the data is raw

    Args:
        path: (str) the path to the data directory relative to the current working directory
        params: (dict) a dictionary of key word arguments:
            start_day: the name of the column containing start study day information
            end_day: the name of the column containing end study day information
            start_date: the name of the column containing start date information
            end_date: the name of the column containing end date information
            subject_column: the name of the column containing subject ID keys
            study_column: the name of the column containing study ID keys
    """

    def __init__(
        self, dataset=pd.DataFrame(), connector=None, study=None, table=None, params={}
    ):
        self.dataset = dataset
        self.connector = connector
        self.study = study
        self.table = table

        self.raw = True

        for key, value in params.items():
            setattr(self, key, value)

        if connector:
            self.load_from_connector()

    def load_from_connector(self):
        """
        Load data from an SQL_connector.

        Args:
            None
        Retuns:
            None
        """
        if not self.study:
            ValueError("Must define study name to use connector")
        elif not self.table:
            ValueError("Must define table name to use connector")
        else:
            sql_string = f"SELECT * FROM {self.study}.{self.table}"
            self.dataset = pd.read_sql(sql_string, self.connector.connector)

    def get_time_column_name(self, time_range):
        """
        Get the name of the appropriate time column.

        Args:
            time_range: (Time_Range) the time range
        Returns:
            a string: if time_range is int type, returns start_day, else returns start_date
        """
        return getattr(self, time_range.start_column)

    def get_site_array(self):
        """
        Get array of sites.

        Args:
            None
        Returns:
            np.array of all Sites in the data
        """
        if self.raw:
            logging.error("Data must be preprocessed before site extraction")
            raise ValueError
        else:
            return self.dataset["Site"].unique()

    def get_data_from_time_range(self, time_range):
        """
        Get the data which falls in the given time range.

        Args:
            time_range: (Time_Range) the time range
        Returns:
            a pandas DataFrame of relevant data
        """

        time_column_name = self.get_time_column_name(time_range)

        left_bool = time_range.start_time <= self.dataset[time_column_name]
        right_bool = self.dataset[time_column_name] < time_range.end_time

        return self.dataset[left_bool & right_bool]
