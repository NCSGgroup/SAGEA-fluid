
import pandas as pd
import os
import requests
import numpy as np
from Setting_ERAInterim import MADML_Type

class ERA_Interim_MAD_ML:
    """
       you can find EAR-Interim monthly average data via: https://gdex.ucar.edu/datasets/d627001/dataaccess/#
    """
    def __init__(self,path="I:/ERA_Interim/MAD_ML/"):
        self.request_address = "https://osdf-data.gdex.ucar.edu/ncar/gdex/d627001/"
        self.root_path = path
        self.remote_time = None
        self.local_time = None
        self.year_file = None
        self.variable = MADML_Type.U_wind
        self.filename = 'u_wind'


    def setTime(self,begin='2000-01-01',end='2010-12-01'):
        hour = '00'
        dates_range = pd.date_range(start=begin,end=end,freq='MS')
        self.remote_time = [d.strftime(f"%Y%m%d{hour}") for d in dates_range]
        self.local_time = dates_range.strftime("%Y%m").tolist()
        self.year_file = dates_range.strftime("%Y").tolist()
        return self

    def setVariable(self,variable:MADML_Type):
        if variable == MADML_Type.U_wind:
            self.variable = 'ei.moda.an.ml/ei.moda.an.ml.regn128uv.'
            self.filename = 'uv_wind'
        elif variable == MADML_Type.V_wind:
            self.variable = "ei.moda.an.ml/ei.moda.an.ml.regn128uv."
            self.filename = 'uv_wind'

        elif variable == MADML_Type.Geopotential:
            self.variable = "ei.moda.an.ml/ei.moda.an.ml.regn128sc."
            self.filename = "sc"

        elif variable == MADML_Type.Temporature:
            self.variable = "ei.moda.an.ml/ei.moda.an.ml.regn128sc."
            self.filename = "sc"

        return self


    def Download(self):
        files, custom_names, root_path = [], [], []
        for k in np.arange(len(self.remote_time)):
            temp_file = f"{self.variable}{self.remote_time[k]}"
            temp_name = f"{self.filename}-{self.local_time[k]}.grib"
            temp_path = f"{self.root_path}/{self.year_file[k]}"
            os.makedirs(temp_path, exist_ok=True)
            files.append(temp_file)
            custom_names.append(temp_name)
            root_path.append(temp_path)

        for i, file in enumerate(files):
            idx = file.rfind("/")
            print(i,file)
            if (idx > 0):
                ofile = file[idx + 1:]
            else:
                ofile = file
            custom_filename = custom_names[i]
            down_path = root_path[i]
            full_path = os.path.join(down_path, custom_filename)
            print(full_path)
            response = requests.get(f"{self.request_address}" + file)
            with open(full_path, "wb") as f:
                f.write(response.content)


if __name__ == "__main__":
    a = ERA_Interim_MAD_ML(path='I:/ERA_Interim/MAD_ML/')
    a.setTime(begin='2000-01-01',end='2005-12-01')
    a.setVariable(variable=MADML_Type.U_wind)
    a.Download()


