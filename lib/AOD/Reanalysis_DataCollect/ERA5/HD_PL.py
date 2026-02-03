import cdsapi
import os
import pandas as pd
from Setting_ERA5 import HDPL_Type
class ERA5_DataCollect():
    """
    HD_PL means ERA5 hourly data on pressure levels
    HD: Hourly data
    PL: Pressure levels
    Details see: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
    """

    def __init__(self,path='I:/ERA5/HD_PL/'):
        self.path = path
        self.variable = "u_component_of_wind"
        self.filename = 'u_wind'
        self.grid = 0.5
        self.daylist = None
        self.timeepoch = None

    def setvariable(self,variable:HDPL_Type):
        if variable == HDPL_Type.Geopotential:
            self.variable="geopotential"
            self.filename="geop"
        elif variable == HDPL_Type.Temporature:
            self.variable = "temperature"
            self.filename = "temp"
        elif variable == HDPL_Type.Humidity:
            self.variable = "specific_humidity"
            self.filename = "shum"
        elif variable == HDPL_Type.U_wind:
            self.variable = "u_component_of_wind"
            self.filename = "u_wind"
        elif variable == HDPL_Type.V_wind:
            self.variable = "v_component_of_wind"
            self.filename = "v_wind"
        elif variable == HDPL_Type.Vertical_Velocity:
            self.variable = "vertical_velocity"
            self.filename = "vervloc"
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
                dataset = "reanalysis-era5-pressure-levels"
                request = {
                    "product_type": ["reanalysis"],
                    "variable": [self.variable],
                    "year": [f"{year}"],
                    "month": [f"{month}"],
                    "day": [f"{day}"],
                    "time": [f"{epoch}"],
                    "pressure_level": [
                        "1", "2", "3",
                        "5", "7", "10",
                        "20", "30", "50",
                        "70", "100", "125",
                        "150", "175", "200",
                        "225", "250", "300",
                        "350", "400", "450",
                        "500", "550", "600",
                        "650", "700", "750",
                        "775", "800", "825",
                        "850", "875", "900",
                        "925", "950", "975",
                        "1000"
                    ],
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
    # a.setvariable(variable="u_component_of_wind", filename="u_wind")
    # a.setTime(begin='2010-12-18',end='2010-12-31')
    # a.setInterval(interval=3)
    # a.ERA5_Download()

    a.setvariable(variable=HDPL_Type.Geopotential)
    a.setTime(begin='2010-12-13',end='2010-12-31')
    a.setInterval(interval=3)
    a.ERA5_Download()
