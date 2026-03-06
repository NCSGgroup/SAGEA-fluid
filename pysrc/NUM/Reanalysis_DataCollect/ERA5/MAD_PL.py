import cdsapi
import os
import pandas as pd
from Setting_ERA5 import MADPL_Type


class ERA5_DataCollect():
    """
    MAD_PL means ERA5 monthly averaged data on pressure levels
    MAD: Monthly averaged data
    PL: Pressure levels
    Details see: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means?tab=download
    """

    def __init__(self,path='I:/ERA5/MAD_PL/'):
        self.variable = "u_component_of_wind"
        self.filename = 'u_wind'
        self.path = path
        self.grid = 0.5
        self.daylist = None

    def setvariable(self,variable:MADPL_Type):
        if variable == MADPL_Type.Geopotential:
            self.variable="geopotential"
            self.filename="geop"
        elif variable == MADPL_Type.U_wind:
            self.variable = "u_component_of_wind"
            self.filename = "u_wind"
        elif variable == MADPL_Type.V_wind:
            self.variable = "v_component_of_wind"
            self.filename = "v_wind"
        elif variable == MADPL_Type.Temporature:
            self.variable = "temperature"
            self.filename = "temp"
        elif variable == MADPL_Type.Humidity:
            self.variable = "specific_humidity"
            self.filename = "shum"
        return self


    def setGrid(self,grid=0.5):
        self.grid = grid
        return self

    def setTime(self,begin='2010-01',end='2010-03'):
        self.daylist = pd.date_range(begin,end,freq='MS').strftime("%Y-%m").tolist()
        return self


    def ERA5_Download(self):
        for i in self.daylist:
            year = i.split('-')[0]
            month = i.split('-')[1]
            main_path = os.path.join(self.path,year)
            if not os.path.exists(main_path):
                os.makedirs(main_path)
            dataset = "reanalysis-era5-pressure-levels-monthly-means"
            request = {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": [self.variable],
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
                "year": [f"{year}"],
                "month": [f"{month}"],
                "time": ["00:00"],
                "grid":[self.grid,self.grid],
                "data_format": "netcdf",
                # "download_format": "zip"
            }

            client = cdsapi.Client()
            client.retrieve(dataset, request).download(f"{main_path}/{self.filename}-{year}{month}.nc")
            print(f"Save file in: {main_path}/{self.filename}-{year}{month}.nc")

def demo1():
    a = ERA5_DataCollect()
    a.setvariable(variable=MADPL_Type.U_wind)
    a.setTime(begin='2000-07',end='2005-12')
    a.setGrid(grid=0.5)
    a.ERA5_Download()

    a.setvariable(variable=MADPL_Type.Geopotential)
    a.setTime(begin='2000-01',end='2005-12')
    a.setGrid(grid=0.5)
    a.ERA5_Download()

    b = ERA5_DataCollect(path='I:/ERA5/MAD_PL_1/')
    b.setvariable(variable=MADPL_Type.V_wind)
    b.setTime(begin='2000-01',end='2005-12')
    b.setGrid(grid=1)
    b.ERA5_Download()


if __name__ == "__main__":
    demo1()






