import pandas as pd
import numpy as np

import warnings
import logging

warnings.filterwarnings("ignore")


class Time_Range:
    """
    A time range objects for handling study day and calendar date flexibility.

    Class Variables:
        start_time: (int, pd.Timestamp) the starting time value
        end_time: (int, pd.Timestamp) the ending time value
        time_step: (int, pd.Timedelta) the time step to take when iterating
        type: (str) a string representing the type, if "int" then the time range is a study day range,
            if "datetime" the time_range is a calendar day range
        iterable: (bool) represents whether the time range is iterable
        start_column: (str)the key for the Dataset start column parameter
        end_column: (str) the key for the Dataset end column parameter

    """

    def __init__(
        self, start_time, end_time, time_step=0, params={"single_score": False}
    ):
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)
        if isinstance(time_step, str):
            time_step = pd.Timedelta(time_step)
        if isinstance(start_time, np.int64):
            start_time = int(start_time)
        if isinstance(end_time, np.int64):
            end_time = int(end_time)

        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step

        for key, value in params.items():
            setattr(self, key, value)

        self.type_check()

    def type_check(self):
        """
        Type checks and sets default values:

        Args:
            None
        Returns:
            None
        """
        if not self.time_step:
            if isinstance(self.start_time, int) and isinstance(self.end_time, int):
                self.type = "int"
                self.iterable = False
                self.start_column = "start_day"
                self.end_column = "end_day"
            elif isinstance(self.start_time, pd.Timestamp) and isinstance(
                self.end_time, pd.Timestamp
            ):
                self.type = "datetime"
                self.iterable = False
                self.start_column = "start_date"
                self.end_column = "end_date"
            else:
                logging.error("Start Time and End Time must both be int or datetime")
                raise TypeError
        else:
            if (
                isinstance(self.start_time, int)
                and isinstance(self.end_time, int)
                and isinstance(self.time_step, int)
            ):
                self.type = "int"
                self.iterable = True
                self.start_column = "start_day"
                self.end_column = "end_day"
            elif (
                isinstance(self.start_time, pd.Timestamp)
                and isinstance(self.end_time, pd.Timestamp)
                and isinstance(self.time_step, pd.Timedelta)
            ):
                self.type = "datetime"
                self.iterable = True
                self.start_column = "start_date"
                self.end_column = "end_date"
            else:
                logging.error(
                    "Start Time, End Time, and Timestep must all be int or datetime"
                )
                raise TypeError

    def __iter__(self):
        """
        An smart iterator, if the type is "int" the iterator is a range, else it is a pandas date_range.
        Args:
            None
        Returns:
            None
        """
        if self.iterable:
            if self.type == "int":
                yield from range(self.start_time, self.end_time, self.time_step)
            else:
                if self.start_time == self.end_time - self.time_step:
                    yield from [self.start_time]
                else:
                    yield from pd.date_range(
                        self.start_time, self.end_time, freq=self.time_step
                    )
        else:
            logging.error("Not iterable")
            raise ValueError
