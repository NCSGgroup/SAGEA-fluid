# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : ProcessSecDrv.py
# @Create  : 11/7/26 18:25
# @Project : SAGEA-fluid
# @Software: PyCharm
"""
Description:
"""

import os
import pygmt
import numpy as np
import xarray as xr
from pysrc.SWOT.SWOTKits.Derivation import derivative_2D


def Second_DRV(BeginCycle=30,EndCycle=30,
               InputPath="../../../result/SWOT/YuF2b/",
               OutputPath = "../../../result/SWOT/Fig_Nov26/"):

    if not os.path.exists(OutputPath):
        os.mkdir(OutputPath)

    all_proj = []
    for i in np.arange(BeginCycle,EndCycle+1):
        # print(InputPath+f"dN_{i:03d}_584_Along_Track.nc")
        da = xr.open_dataset(InputPath+f"dN_{i:03d}_584_Along_Track.nc")
        dc = xr.open_dataset(InputPath+f"dN_{i:03d}_584_Cross_Track.nc")
        proj = da['z'].values+dc['z'].values
        all_proj.append(proj)
        # all_da.append(da['z'].values)
        # all_dc.append(dc['z'].values)
        lat,lon = da['lat'].values,da['lon'].values
    all_proj = np.array(all_proj)
    # print(f"all_proj.shape:{all_proj.shape}")
    stack_proj = np.nanmean(all_proj,axis=0)
    # print(f"stack_proj.shape:{stack_proj.shape}")
    dn, de = derivative_2D(data=stack_proj,spacing=2000,pNo=1)
    # print(f"second derivations on North-direction, East-direction:{dn.shape},{de.shape}")
    return dn,de




def demo():
    a = Second_DRV(BeginCycle=30,EndCycle=30)

if __name__ == "__main__":
    demo()