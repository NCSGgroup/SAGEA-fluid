# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : ProcessGed.py
# @Create  : 11/9/26 17:32
# @Project : SAGEA-fluid
# @Software: PyCharm
"""
Description:
"""
import os
import pygmt
import numpy as np
import xarray as xr
from tqdm import tqdm
from pysrc.SWOT.KaRIn.L2LRSSH.SSH_Expert import Load_SSH
from pysrc.SWOT.SWOTKits.Derivation import derivative_2D,derivative_1D
from pysrc.SWOT.Settings.SWOTClasses import TrackType

res_hor = 2000
# InputPath = "/Users/zhang.17371/Documents/"
OutputPath = f"../../../result/SWOT/YuF2b/"
if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)

def first_derivation_one_pass(cNo=30,pNo=10,InputPath='/Users/zhang.17371/Documents/Datasets/SWOT/'):
    res = res_hor
    datasets = Load_SSH(InputPath=InputPath).get_ssh_pass(cycle_No=cNo, pass_No=pNo)
    if datasets is None:
        return None
    else:
        mss, mdt = datasets['mss_cnescls'], datasets['mdt']
        tides, dac = datasets['tides'], datasets['dac']
        ssh, ssh_qual = datasets['ssh'], datasets['ssh_qual']
        swh, ged = datasets['swh'], datasets['geoid']
        xover, x_qual = datasets['height_cor_xover'], datasets['height_cor_xover_qual']
        ssh_clean = ssh - tides - dac
        lat, lon = datasets['lat'], datasets['lon']

        ssh_value = ssh_clean
        mss_value = mss
        mdt_value = mdt
        ged_value = ged
        xover_value = xover

        ssh_da, ssh_dc = derivative_2D(data=ssh_value, spacing=res, pNo=pNo)
        mss_da, mss_dc = derivative_2D(data=mss_value, spacing=res, pNo=pNo)
        mdt_da, mdt_dc = derivative_2D(data=mdt_value, spacing=res, pNo=pNo)
        ged_da, ged_dc = derivative_2D(data=ged_value, spacing=res, pNo=pNo)
        results = {"ssh": ssh_value,
                   "mss": mss_value,
                   "mdt": mdt_value,
                   "geoid": ged_value,
                   "ssh_da": ssh_da, "ssh_dc": ssh_dc,
                   "mss_da": mss_da, "mss_dc": mss_dc,
                   "mdt_da": mdt_da, "mdt_dc": mdt_dc,
                   "ged_da": ged_da, "ged_dc": ged_dc,
                   "lat": lat, "lon": lon}
        return results
    pass

def first_derivation_one_region(cNo=30,TotalPass=584,region=None,
                                TrackType=TrackType.Along_Track,
                                InputPath='/Users/zhang.17371/Documents/Datasets/SWOT/'):
    all_x, all_y, all_z = [], [], []
    OutputFile = os.path.join(OutputPath, f"Geoid_{cNo:03d}_{TotalPass}_{TrackType.name}.nc")
    if region is None:
        region = [0, 360, -90, 90]
    else:
        region = region

    for i in tqdm(np.arange(TotalPass), desc=f" > > Loading total pass {TotalPass} of cycle {cNo} Now ..."):
        datatemp = first_derivation_one_pass(cNo=cNo, pNo=i + 1, InputPath=InputPath)
        if datatemp is None:
            continue
        else:
            x_value = datatemp['lon'].flatten()
            y_value = datatemp['lat'].flatten()
            if TrackType == TrackType.Along_Track:
                z_value = datatemp['ged_da'].flatten()

            elif TrackType == TrackType.Cross_Track:
                z_value = datatemp['ged_dc'].flatten()

            else:
                z_value = datatemp['geoid'].flatten()

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
    x, y, v = all_x[valid], all_y[valid], all_z[valid]
    print(f"============================================================\n")
    grid = pygmt.xyz2grd(
        x=x,
        y=y,
        z=v,
        region=region,
        spacing="2k/2k",
        registration="p",
        coltypes="g",
        duplicate="m",
        outgrid=OutputFile,
    )
    # print(grid)
    print(f"Grid saved to {OutputFile}")
    return grid

def second_derivation_region(BeginCycle=26,EndCycle=31,region=None,
                             InputPath="../../../result/SWOT/YuF2b/"):
    g_wmo = 9.80665
    all_model = []
    for i in np.arange(BeginCycle, EndCycle + 1):
        # print(InputPath+f"dN_{i:03d}_584_Along_Track.nc")
        da_model = xr.open_dataset(InputPath + f"Geoid_{i:03d}_584_Along_Track.nc")
        dc_model = xr.open_dataset(InputPath + f"Geoid_{i:03d}_584_Cross_Track.nc")
        # da_resid = xr.open_dataset(InputPath + f"dN_{i:03d}_584_Along_Track.nc")
        # dc_resid = xr.open_dataset(InputPath + f"dN_{i:03d}_584_Cross_Track.nc")

        mol = da_model['z'].values + dc_model['z'].values
        # red = da_resid['z'].values + dc_resid['z'].values
        # print(f"mod/red:{mol.shape} and {red.shape}")
        all_model.append(mol)
        # all_resid.append(red)

        lat, lon = da_model['lat'].values, da_model['lon'].values

