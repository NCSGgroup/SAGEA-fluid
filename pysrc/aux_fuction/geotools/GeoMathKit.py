import calendar
import datetime
import gzip
import math

import numpy as np

class GeoMathKit:

    def __init__(self):
        pass

    @staticmethod
    def generate_month_range(start_str, end_str):
        """
        input:2009-01,2010-12
        """
        start = datetime.datetime.strptime(start_str, "%Y-%m")
        end = datetime.datetime.strptime(end_str, "%Y-%m")

        months = []
        current = start
        while current <= end:
            months.append(current.strftime("%Y-%m"))
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return months
