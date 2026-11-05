# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : ProjectionEN.py
# @Create  : 11/3/26 15:59
# @Project : PySWOT
# @Software: PyCharm
"""
Description:
"""
from pathlib import Path
import os
from pysrc.SWOT.Process.PreProcess import data_filter,data_unfliter
import numpy as np
import pygmt
from tqdm import tqdm
from pysrc.SWOT.Settings.SWOTClasses import TrackType

def projection_EN(cNo=30,TotalPass=584,tracktype=TrackType.Along_Track, region=None):
    all_x,all_y,all_z = [],[],[]
    OutputPath = f"../../result/tempdata/"
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    OutputFile = os.path.join(OutputPath,f"delta_ssh_{cNo:03d}_{TotalPass}_{tracktype.name}.nc")
    # region = [55, 65, 10, 20]  # west, east, south, north
    # region = [-90,90,0,360]
    if region is None:
        region = [0,360,-90,90]
    else:
        region = region

    for i in tqdm(np.arange(TotalPass),desc=f" > > Loading total pass {TotalPass} of cycle {cNo} Now ..."):
        datatemp = data_filter(cNo=cNo,pNo=i+1,InputPath="/Users/zhang.17371/Documents/")
        if datatemp is None:
            continue
        else:
            x_value = datatemp['lon'].flatten()
            y_value = datatemp['lat'].flatten()
            if tracktype == TrackType.Along_Track:
                z_value = (datatemp['ssh_da']-datatemp['ged_da']-datatemp['mdt_da']).flatten()
            elif tracktype == TrackType.Cross_Track:
                z_value = (datatemp['ssh_dc']-datatemp['ged_dc']-datatemp['mdt_da']).flatten()
            else:
                z_value = datatemp['ssh'].flatten()
            all_x.extend(x_value)
            all_y.extend(y_value)
            all_z.extend(z_value)
    print(f"length of all cycle{cNo} x/y/z: {len(all_x)}, {len(all_y)}, {len(all_z)}")
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)

    valid = (
            np.isfinite(all_x)
            & np.isfinite(all_y)
            & np.isfinite(all_z)
            & (all_x >= region[0]) & (all_x <= region[1])
            & (all_y >= region[2]) & (all_y <= region[3])
    )
    x,y,v = all_x[valid],all_y[valid],all_z[valid]
    print(f"============================================================\n")
    grid = pygmt.xyz2grd(
        x=x,
        y=y,
        z=v,
        region=region,
        spacing="2k/2k",
        registration="p",
        coltypes="g",
        duplicate = "m",
        # outgrid=OutputFile,
    )
    # print(grid)
    print(f"Grid saved to {OutputFile}")
    return grid

if __name__ == "__main__":
    projection_EN(cNo=30,TotalPass=50)