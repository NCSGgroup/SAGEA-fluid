import cdsapi
import os
import pandas as pd
from Setting_ERA5 import MADSL_Type

class ERA5_DataCollect():
    """
    MAD_SL means ERA5 monthly averaged data on single levels
    MAD: Monthly averaged data
    SL: Single levels
    Details see: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download
    """

    def __init__(self,path='I:/ERA5/MAD_SL/'):
        self.variable = MADSL_Type.SurPressure
        self.filename = 'sp'
        self.path = path
        self.grid = 0.5
        self.daylist = None

    def setvariable(self,variable:MADSL_Type):
        if variable == MADSL_Type.Geopotential:
            self.variable="geopotential"
            self.filename="s_geop"
        elif variable == MADSL_Type.SurPressure:
            self.variable = "surface_pressure"
            self.filename = "sp"
        elif variable == MADSL_Type.V_wind:
            self.variable = "10m_v_component_of_wind"
            self.filename = "10m_v_wind"
        elif variable == MADSL_Type.U_wind:
            self.variable = "10m_u_component_of_wind"
            self.filename = "10m_u_wind"
        return self

    def setGrid(self,grid=0.5):
        self.grid=grid
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
            dataset = "reanalysis-era5-single-levels-monthly-means"
            request = {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": [self.variable],
                "year": [f"{year}"],
                "month": [f"{month}"],
                "time":["00:00"],
                "grid":[self.grid,self.grid],
                "data_format": "netcdf",
                # "download_format": "zip"
            }

            client = cdsapi.Client()
            client.retrieve(dataset, request).download(f"{main_path}/{self.filename}-{year}{month}.nc")
            print(f"Save file in: {main_path}/{self.filename}-{year}{month}.nc")


def demo1():
    a = ERA5_DataCollect(path='I:/ERA5/MAD_SL_1/')
    a.setvariable(variable=MADSL_Type.SurPressure)
    a.setTime(begin='2000-01', end='2005-12')
    a.setGrid(grid=1)
    a.ERA5_Download()

if __name__ == "__main__":
    demo1()

