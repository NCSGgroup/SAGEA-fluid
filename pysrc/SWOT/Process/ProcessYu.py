# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : ProcessYu.py
# @Create  : 11/11/26 16:37
# @Project : SAGEA-fluid
# @Software: PyCharm
"""
Description:
"""
import os

import numpy as np
import xarray as xr
import pandas as pd
import pygmt
from tqdm import tqdm
from pysrc.SWOT.KaRIn.L2LRSSH.SSH_Expert import Load_SSH
from matplotlib import pyplot as plt

def swath_unit_vectors(lon, lat):

    """
    compute the unit direction vectors for each cell in a swath
    Parameters
    ----------
    lat : array_like
        2D array of latitude values with shape (na, nc).
    lon : array_like
        2D array of longitude values with shape (na, nc).
    Returns
    -------
    uea : array_like
        2D array of the east component of the init vector in the along-track direction
    una : array_like
        2D array of the north component of the init vector in the along-track direction
    uec : array_like
        2D array of the east component of the init vector in the cross-track
    unc : array_like
        2D array of the north component of the init vector in the cross-track
    """
    d2r = np.pi/180.0
    cost = np.cos(lat*d2r)
    lon = np.unwrap(lon, period = 360)

    dna, dnc = np.gradient(lat,1,1)
    dea, dec = np.gradient(lon,1,1)*cost
    dsa = (dea*dea + dna*dna)**0.5
    uea = dea/dsa
    una = dna/dsa
    dsc = (dec*dec + dnc*dnc)**0.5
    uec = dec/dsc
    unc = dnc/dsc

    return uea, una, uec, unc

def external_model(InputPath,cNo,pNo):
    pass

def swot_data_variables(ds_expert:xr.Dataset):
    ssh = ds_expert.ssh_karin
    ssh_qual = ds_expert.ssh_karin_qual
    ssha = ds_expert.ssh_karin
    dac = ds_expert.dac
    swh = ds_expert.swh_karin
    distance = ds_expert.cross_track_distance.values
    cor_xover = ds_expert.height_cor_xover
    cor_xover_qual = ds_expert.height_cor_xover_qual
    mss = ds_expert.mean_sea_surface_cnescls
    mdt = ds_expert.mean_dynamic_topography
    geoid = ds_expert.geoid

    tides = (ds_expert['ocean_tide_fes'].values +
             ds_expert['ocean_tide_non_eq'].values +
             ds_expert['pole_tide'].values +
             ds_expert['solid_earth_tide'].values+
             ds_expert["internal_tide_hret"].values)
    num_lines = ds_expert.sizes['num_lines']
    num_pixels = ds_expert.sizes['num_pixels']
    lon, lat = ds_expert['longitude'].values, ds_expert['latitude'].values
    lon = np.unwrap(lon, period=360)

    swot_variables = {
        "ssh": ssh,
        "ssh_qual": ssh_qual,
        "ssha": ssha,
        "dac": dac,
        "cor_xover": cor_xover,
        "cor_xover_qual": cor_xover_qual,
        "tides": tides,
        "num_lines": num_lines,
        "num_pixels": num_pixels,
        "swh":swh,
        "cross_track_distance":distance,
        "mss":mss,
        "mdt":mdt,
        "geoid":geoid,
        "lat":lat,
        "lon":lon,

    }
    return swot_variables

def model_vgg():
    pass

def swot_res_vgg(InputPath="/Users/zhang.17371/Documents/Datasets/SWOT/",BegincNo=27,EndcNo=31,pNo=100):
    sa_list, sc_list, swh_list = [], [], []
    gsa_list,gsc_list,msa_list,msc_list = [],[],[],[]
    # for cNo in tqdm(np.arange(BegincNo,EndcNo+1),desc=f"> > Processing cycle from {BegincNo} to {EndcNo}"):
    for cNo in np.arange(BegincNo, EndcNo + 1):
        ds_expert_list = Load_SSH(InputPath=InputPath).return_pass_path(cNo=cNo,pNo=pNo)
        # print(ds_expert_list)
        if not ds_expert_list:
            print(f"> > No expert file on cycle_id:{cNo} and pass_id:{cNo}.")
            return None
        else:
            # print(f"Processing Cycle ID:{cNo} and Pass ID:{cNo}.")
            ds_expert = xr.open_dataset(ds_expert_list[0])

            swot_variables = swot_data_variables(ds_expert)
            num_lines = swot_variables["num_lines"]
            num_pixels = swot_variables['num_pixels']
            # print(f"rows/columns: {num_lines}/{num_pixels}")
            lon,lat = ds_expert['longitude'].values,ds_expert['latitude'].values
            # print(lon)
            lon = np.unwrap(lon,period = 360)
            ssh = swot_variables["ssh"]-swot_variables["tides"]-swot_variables["dac"]+swot_variables["cor_xover"]
            ssh = np.where(swot_variables["ssh_qual"] == 0, ssh, np.nan)
            ssh = np.where(ds_expert.ancillary_surface_classification_flag == 0, ssh, np.nan)

            swh = np.where(swot_variables['swh']==0,swot_variables['swh'],np.nan)
            swh = np.where(swh!=0,swh,np.nan)

            """Delta N = SSH-EGM2008-MDT"""
            ssh_res = ssh-swot_variables['geoid'].values-swot_variables['mdt'].values
            geo_mod = swot_variables['geoid'].values
            # mss_mod = swot_variables['mss'].values
            # mdt_mod = swot_variables['mdt'].values

            # print(ssh_res)
            data = {'lon':lon.flatten(),'lat':lat.flatten()}

            da,dc = 2000,2000
            sa_res,sc_res = np.gradient(ssh_res,da,dc)
            sa_list.append(sa_res)
            sc_list.append(sc_res)
            swh_list.append(swh)

            gsa,gsc = np.gradient(geo_mod,da,dc)
            gsa_list.append(gsa)
            gsc_list.append(gsc)

        """Yu here did something that I do not understand!"""

        weight = np.nan_to_num(0*np.array(swh_list)+1,nan=1)
        weight[np.array(swh_list)>6]=0
        mask_sa = np.ma.masked_array(np.array(sa_list),np.isnan(np.array(sa_list)))
        sa_stack = np.ma.average(mask_sa,axis=0,weights=weight).filled(np.nan)

        mask_sc = np.ma.masked_array(np.array(sc_list),np.isnan(np.array(sc_list)))
        sc_stack = np.ma.average(mask_sc,axis=0,weights=weight).filled(np.nan)

        gweight = np.nan_to_num(0*np.array(gsa_list)+1,nan=1)
        gsa_stack = np.ma.average(np.array(gsa_list),axis=0,weights=gweight).filled(np.nan)
        gsc_stack = np.ma.average(np.array(gsc_list),axis=0,weights=gweight).filled(np.nan)

        if pNo % 2 == 1: # odd is ascending, even is descending
            sa_stack = -sa_stack
            sc_stack = -sc_stack

            gsa_stack = -gsa_stack
            gsc_stack = -gsc_stack

        if pNo % 2 == 1:
            saa_res, sac_res = np.gradient(sa_stack,da,dc)
            sca_res, scc_res = np.gradient(sc_stack,da,dc)
            vgg_res = -(saa_res+scc_res)*9.80665

            gsaa,gsac = np.gradient(gsa_stack,da,dc)
            gsca,gscc = np.gradient(gsc_stack,da,dc)
            vgg_mod = -(gsaa+gscc)*9.80665

        if pNo % 2 == 0:
            saa_res, sac_res = np.gradient(sa_stack,da,dc)
            sca_res, scc_res = np.gradient(sc_stack,da,dc)
            vgg_res = (saa_res+scc_res)*9.80665

            gsaa, gsac = np.gradient(gsa_stack, da, dc)
            gsca, gscc = np.gradient(gsc_stack, da, dc)
            vgg_mod = (gsaa + gscc)*9.80665

        out = {"longitude":ds_expert["longitude"].values.flatten(),
               "latitude":ds_expert.latitude.values.flatten(),
               "vgg_res":vgg_res.flatten(),
               "vgg_mod":vgg_mod.flatten(),}
        return out


def Process_VGG(BeginC,EndC,TotalPass=30,region=None,InputPath="/Users/zhang.17371/Documents/Datasets/SWOT/",OutputPath="../../../result/SWOT/New_YuF2b/"):
    if not os.path.exists(OutputPath):
        os.mkdir(OutputPath)
    OutputVGG = os.path.join(OutputPath,f"VGG_cycle_{BeginC}_{EndC}_{TotalPass}.nc")
    OutputRES = os.path.join(OutputPath,f"RES_cycle_{BeginC}_{EndC}_{TotalPass}.nc")
    OutputMOD = os.path.join(OutputPath, f"MOD_cycle_{BeginC}_{EndC}_{TotalPass}.nc")
    all_x, all_y, all_vgg, all_res,all_mod =[],[], [], [], []
    if region==None:
        region = [0,360,-90,90]
    else:
        region = [0,360,-90,90]

    for pass_id in tqdm(np.arange(TotalPass), desc=f" > > Loading total pass {TotalPass} of cycle from {BeginC}-{EndC} Now ..."):
        output_files = swot_res_vgg(InputPath=InputPath,BegincNo=BeginC,EndcNo=EndC,pNo=pass_id)
        if output_files is None:
            continue
        else:
            x_value = output_files['longitude']
            y_value = output_files['latitude']
            mod_value = output_files['vgg_mod']
            res_value = output_files['vgg_res']
            vgg_value = mod_value+res_value

            all_x.append(x_value)
            all_y.append(y_value)
            all_vgg.append(vgg_value)
            all_mod.append(mod_value)
            all_res.append(res_value)

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_vgg = np.array(all_vgg)
    all_mod = np.array(all_mod)
    all_res = np.array(all_res)
    print(f"shapes of all x,y,vee,res,mod:{all_x.shape},{all_y.shape},{all_vgg.shape},{all_mod.shape},{all_res.shape}")

    valid = (
            np.isfinite(all_x)
            & (all_x >= region[0]) & (all_x <= region[1])
            & (all_y >= region[2]) & (all_y <= region[3])
    )
    print(f"shape of valid:{valid.shape}")
    x,y,vgg,res,mod = all_x[valid],all_y[valid],all_vgg[valid],all_res[valid],all_mod[valid]
    print(f"shapes of x,y,vgg,res,mod:{x.shape},{y.shape},{vgg.shape},{mod.shape},{res.shape}")

    grid1 = pygmt.xyz2grd(
        x=x,
        y=y,
        z=vgg,
        region=region,
        spacing="2k/2k",
        registration="p",
        coltypes="g",
        duplicate="m",
        outgrid=OutputVGG,
    )

    grid2 = pygmt.xyz2grd(
        x=x,
        y=y,
        z=res,
        region=region,
        spacing="2k/2k",
        registration="p",
        coltypes="g",
        duplicate="m",
        outgrid=OutputRES,
    )

    grid3 = pygmt.xyz2grd(
        x=x,
        y=y,
        z=mod,
        region=region,
        spacing="2k/2k",
        registration="p",
        coltypes="g",
        duplicate="m",
        outgrid=OutputMOD,
    )
    print(f"Grid saved to {OutputVGG}")

def demo():
    swot_res_vgg()

def demo2():
    Process_VGG(BeginC=27,EndC=30,TotalPass=584,region=[60,90,-45,-30])

if __name__ == "__main__":
    demo2()