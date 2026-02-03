
import pandas as pd
import os
import requests
import numpy as np
from Setting_ERAInterim import HDML_Type

class ERA_Interim_HD_ML:
    """
    you can find EAR-Interim monthly average data via: https://gdex.ucar.edu/datasets/d627001/dataaccess/#
    """
    def __init__(self,path="I:/ERA_Interim/HD_ML/"):
        self.request_address = "https://osdf-data.gdex.ucar.edu/ncar/gdex/d627000/"
        self.root_path = path
        self.daylist = None
        self.variable = HDML_Type.U_wind
        self.filename = 'u_wind'
        self.timeepoch = None


    def setTime(self,begin,end):
        self.daylist = pd.date_range(begin,end).strftime("%Y-%m-%d").tolist()
        return self

    def setInterval(self,interval=6, begin=0, end=24):
        timeepoch = []
        for i in range(begin,end,interval):
            timeepoch.append('{}:00'.format(str(i).rjust(2,'0')))
        self.timeepoch = timeepoch
        return self

    def setVariable(self,variable:HDML_Type):
        if variable == HDML_Type.U_wind:
            self.variable = 'ei.oper.an.ml/200001/ei.oper.an.ml.regn128uv.'
            self.filename = 'uv_wind'
        elif variable == HDML_Type.V_wind:
            self.variable = "ei.oper.an.ml/200001/ei.oper.an.ml.regn128uv."
            self.filename = 'uv_wind'
            self.request_address = "https://osdf-data.gdex.ucar.edu/ncar/gdex/d627000/"
        elif variable == HDML_Type.Geopotential:
            self.variable = "ei.oper.an.ml/200001/ei.oper.an.ml.regn128sc"
            self.filename = "sc"
            self.request_address = "https://osdf-data.gdex.ucar.edu/ncar/gdex/d627000/"
        elif variable == HDML_Type.Temporature:
            self.variable = "ei.oper.an.ml/200001/ei.oper.an.ml.regn128sc"
            self.filename = "sc"
            self.request_address = "https://osdf-data.gdex.ucar.edu/ncar/gdex/d627000/"
        return self


    def Download(self):
        for i in self.daylist:
            year = i.split('-')[0]
            month = i.split('-')[1]
            day = i.split('-')[2]
            main_path = os.path.join(self.root_path, year)
            main_path = os.path.join(main_path, f"{year}{month}{day}")
            # print(main_path)
            if not os.path.exists(main_path):
                os.makedirs(main_path)
            for epoch in self.timeepoch:
                suffix = epoch.split(':')[0]
                custom_filename = f"{self.filename}-{year}{month}{day}{suffix}.grib"
                full_path = os.path.join(main_path,custom_filename)
                print(full_path)
                response = requests.get(f"{self.request_address}" + f"{self.variable}{year}{month}{day}{suffix}")
                with open(full_path,"wb") as f:
                    f.write(response.content)



if __name__ == "__main__":
    a = ERA_Interim_HD_ML(path='I:/ERA_Interim/HD_ML/')
    a.setTime(begin='2000-01-01',end='2000-01-31')
    a.setVariable(variable=HDML_Type.U_wind)
    a.setInterval(interval=6)
    a.Download()


