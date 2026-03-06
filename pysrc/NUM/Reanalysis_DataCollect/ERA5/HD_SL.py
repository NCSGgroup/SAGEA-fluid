import cdsapi
import os
import pandas as pd
from Setting_ERA5 import HDSL_Type

class ERA5_DataCollect():
    """
    HD_SL means ERA5 hourly data on single levels
    HD: Hourly data
    SL: single levels
    Details see: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
    """

    def __init__(self,path='I:/ERA5/HD_SL/'):
        self.path = path
        self.variable = "surface_pressure"
        self.filename = 'sp'
        self.grid = 0.5
        self.daylist = None
        self.timeepoch = None

    def setvariable(self,variable:HDSL_Type):
        if variable == HDSL_Type.SurPressure:
            self.variable = "surface_pressure"
            self.filename = 'sp'
        elif variable == HDSL_Type.Geopotential:
            self.variable = "geopotential"
            self.filename = 's_geop'
        elif variable == HDSL_Type.U_wind:
            self.variable = "10m_u_component_of_wind"
            self.filename = '10m_u_wind'
        elif variable == HDSL_Type.V_wind:
            self.variable = "10m_v_component_of_wind"
            self.filename = '10m_v_wind'
        elif variable == HDSL_Type.Temperature:
            self.variable = "2m_temperature"
            self.filename = '2m_temp'
        return self

    def setTime(self,begin,end):
        self.daylist = pd.date_range(begin,end).strftime("%Y-%m-%d").tolist()
        return self

    def setInterval(self,interval=3, begin=0, end=24):
        timeepoch = []
        for i in range(begin,end,interval):
            timeepoch.append('{}:00'.format(str(i).rjust(2,'0')))
        self.timeepoch = timeepoch
        return self

    def ERA5_Download(self):
        for i in self.daylist:
            year = i.split('-')[0]
            month = i.split('-')[1]
            day = i.split('-')[2]
            main_path = os.path.join(self.path, year)
            main_path = os.path.join(main_path,f"{year}{month}{day}")
            # print(main_path)
            if not os.path.exists(main_path):
                os.makedirs(main_path)
            for epoch in self.timeepoch:
                dataset = "reanalysis-era5-single-levels"
                request = {
                    "product_type": ["reanalysis"],
                    "variable": [self.variable],
                    "year": [f"{year}"],
                    "month": [f"{month}"],
                    "day": [f"{day}"],
                    "time": [f"{epoch}"],
                    "grid":[self.grid,self.grid],
                    "data_format": "netcdf",
                    # "download_format": "zip"
                }
                suffix = epoch.split(':')[0]
                client = cdsapi.Client()
                client.retrieve(dataset, request).download(f"{main_path}/{self.filename}-{year}{month}{day}{suffix}.nc")
                print(f"Save path in {main_path}/{self.filename}-{year}{month}{day}{suffix}.nc")

if __name__ == '__main__':
    a = ERA5_DataCollect()
    a.setvariable(variable=HDSL_Type.SurPressure)
    a.setTime(begin='2010-11-02',end='2010-12-31')
    a.setInterval(interval=3)
    a.ERA5_Download()
