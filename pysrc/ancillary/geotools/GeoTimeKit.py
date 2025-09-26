import pandas as pd

class GeoTimeKit:
    def __init__(self):
        pass
    @staticmethod
    def date_range_to_mjd(begin_date,end_date,freq="D"):
        date_range = pd.date_range(start=begin_date,end=end_date,freq=freq)
        mjds = date_range.to_julian_date()-2400000.5
        mjds_list = mjds.tolist()

        return mjds_list
    @staticmethod
    def date_to_mjd(date):
        date_epoch = pd.to_datetime(date)
        mjd = date_epoch.to_julian_date()-2400000.5

        mjd_float = format(mjd,'.3f')
        return mjd_float


if __name__ == '__main__':
    # mjd = GeoTimeKit().date_range_to_mjd(begin_date='2001-01-01',end_date='2001-01-10',freq='D')
    # print(mjd)
    mjd = GeoTimeKit().date_to_mjd(date='2000-01-01')
    print(type(mjd))