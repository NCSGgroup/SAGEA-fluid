# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : Process.py
# @Create  : 11/9/26 17:32
# @Project : SAGEA-fluid
# @Software: PyCharm
"""
Description: This script compute the slope and vertical gravity gradient based on along-track (da) and cross-track (dc)
Support to compute the slope (partial derivation) based on one-pass;
Support to compute the slope (partial derivation) based on certain region;
Support to compute the certain region partial derivation's partial derivation;
"""
import os
import pygmt
import numpy as np
import xarray as xr
from tqdm import tqdm
from pysrc.SWOT.KaRIn.L2LRSSH.SSH_Expert import Load_SSH
from pysrc.SWOT.SWOTKits.Derivation import derivative_2D
from pysrc.SWOT.Settings.SWOTClasses import BGtype,TrackType

def compute_slope_pass(cycle_id=30,pass_id=10,InputPath="/Users/zhang.17371/Documents/Datasets/SWOT/"):
    spacing_res = 2000      # spatial resolution is 2km, 2000m
    pass_data = Load_SSH(InputPath=InputPath).get_ssh_pass(cycle_id=cycle_id, pass_id=pass_id)
    if pass_data is None:
        return None
    else:
        mss, mdt = pass_data['mss_cnescls'], pass_data['mdt']
        tides, dac = pass_data['tides'], pass_data['dac']
        ssh, ssh_qual = pass_data['ssh'], pass_data['ssh_qual']
        swh, ged = pass_data['swh'], pass_data['geoid']
        xover, x_qual = pass_data['height_cor_xover'], pass_data['height_cor_xover_qual']
        ssh_clean = ssh - tides - dac
        lat, lon = pass_data['lat'], pass_data['lon']

        """Get the valid index"""
        valid_1 = swh < 6  # Yu, high-quality swh<6m
        valid_2 = ssh_qual < 1  # Zhang, ssh_flag_qual=0 "good"
        valid_3 = x_qual < 1  # Zhang, height_xover, qual=0 "good"
        # print(f"original ssh_value:\n{ssh_clean}")
        # ssh_value = ssh_clean
        ssh_value = np.where(valid_1, ssh_clean, np.nan)  # Yu, filter swh>6m in ssh
        ssh_value = np.where(valid_2, ssh_value, np.nan)  # Zhang, only keep ssh_qual=0
        ssh_value = np.where(valid_3, ssh_value, np.nan)  # Zhang, only keep height_xover_qual=0
        xover_value = np.where(~np.isnan(ssh_value), xover, np.nan)

        ssh_value = ssh_value+xover_value
        mss_value = mss
        mdt_value = mdt
        ged_value = ged

        dif1_values = ssh_value-mss_value-mdt_value-ged_value
        dif2_values = ssh_value-ged_value

        ssh_da, ssh_dc = derivative_2D(data=ssh_value, spacing=spacing_res, pass_id=pass_id)
        mss_da, mss_dc = derivative_2D(data=mss_value, spacing=spacing_res, pass_id=pass_id)
        mdt_da, mdt_dc = derivative_2D(data=mdt_value, spacing=spacing_res, pass_id=pass_id)
        ged_da, ged_dc = derivative_2D(data=ged_value, spacing=spacing_res, pass_id=pass_id)
        dif1_da,dif1_dc = derivative_2D(data=dif1_values, spacing=spacing_res, pass_id=pass_id)
        dif2_da,dif2_dc = derivative_2D(data=dif2_values, spacing=spacing_res, pass_id=pass_id)
        results = {"ssh": ssh_value,
                   "mss": mss_value,
                   "mdt": mdt_value,
                   "geoid": ged_value,
                   "ssh_da": ssh_da, "ssh_dc": ssh_dc,
                   "mss_da": mss_da, "mss_dc": mss_dc,
                   "mdt_da": mdt_da, "mdt_dc": mdt_dc,
                   "ged_da": ged_da, "ged_dc": ged_dc,
                   "dif1_da": dif1_da, "dif1_dc": dif1_dc,
                   "dif2_da": dif2_da, "dif2_dc": dif2_dc,
                   "lat": lat, "lon": lon}
        return results


def compute_slope_region(cycle_id=30,pass_total=584,region=None,
                         Track=TrackType.Along_Track,
                         InputPath="/Users/zhang.17371/Documents/Datasets/SWOT/",
                         OutputPath="../../../result/SWOT/",IsSave=True):
    all_x, all_y = [], []
    all_dif1, all_dif2 = [], []
    all_ged, all_mss = [], []

    if region is None:
        region = [0, 360, -90, 90]
    else:
        region = region

    for i in tqdm(np.arange(pass_total), desc=f" > > Loading total pass {pass_total} of cycle {cycle_id} Now ..."):
        datatemp = compute_slope_pass(cycle_id=cycle_id, pass_id=i + 1, InputPath=InputPath)
        if datatemp is None:
            continue
        else:
            x_value, y_value = datatemp['lon'].flatten(), datatemp['lat'].flatten()
            if Track == TrackType.Along_Track:
                ged_d = datatemp["ged_da"].flatten()
                mss_d = datatemp["mss_da"].flatten()

                """Difference 1: SSH-MSS-MDT"""
                dif1_d = datatemp['dif1_da'].flatten()
                # dif1_dc = datatemp['dif1_dc'].flatten()

                """Difference 2: SSH-EGM2008(Geoid)"""
                dif2_d = datatemp['dif2_da'].flatten()
                # dif2_dc = datatemp['dif2_dc'].flatten()
            else:
                ged_d = datatemp["ged_dc"].flatten()
                mss_d = datatemp["mss_dc"].flatten()

                """Difference 1: SSH-MSS-MDT"""
                dif1_d = datatemp['dif1_dc'].flatten()

                """Difference 2: SSH-EGM2008(Geoid)"""
                dif2_d = datatemp['dif2_dc'].flatten()


            all_x.extend(x_value)
            all_y.extend(y_value)

            all_ged.extend(ged_d)
            all_mss.extend(mss_d)

            all_dif1.extend(dif1_d)
            all_dif2.extend(dif2_d)

    print(f"length of all cycle{cycle_id} x/y/z: "
          f"{len(all_x)}, {len(all_y)}, {len(all_dif1)}")
    all_x, all_y = np.array(all_x),np.array(all_y)
    all_ged, all_mss = np.array(all_ged),np.array(all_mss)
    all_dif1, all_dif2 = np.array(all_dif1),np.array(all_dif2)


    valid = (
            np.isfinite(all_x)
            & np.isfinite(all_y)
            # & np.isfinite(all_dif1_da)
            & (all_x >= region[0]) & (all_x <= region[1])
            & (all_y >= region[2]) & (all_y <= region[3])
    )
    x, y = all_x[valid], all_y[valid]
    dged, dmss = all_ged[valid], all_mss[valid]
    d1, d2 = all_dif1[valid], all_dif2[valid]

    print(f"============================================================\n")
    if IsSave:
        if not os.path.exists(OutputPath):
            os.makedirs(OutputPath)

        data_series = [dged,dmss,d1,d2]
        suffix = ["EGM2008","MSSCLS",
                  "dEGM2008","dMSSCLS"]
        for i in np.arange(len(data_series)):
            OutputFile = os.path.join(OutputPath, f"Slope_{suffix[i]}_{cycle_id:03d}_{pass_total}_{Track.name}.nc")
            grid = pygmt.xyz2grd(
                x=x,
                y=y,
                z=data_series[i],
                region=region,
                spacing="2k/2k",
                registration="p",
                coltypes="g",
                duplicate="m",
                outgrid=OutputFile,
            )
            print(f"Grid saved to {OutputFile}")
        return None
    else:
        slopes = {
            "lon":x,
            "lat":y,
            "slope_geoid":dged,
            "slope_mss":dmss,
            "slope_dif1":d1,
            "slope_dif2":d2,
        }
        return slopes


def compute_vgg(BeginCycle=27,EndCycle=30,
                InputPath="../../../result/SWOT/YuF2b/",
                Background=BGtype.EGM2008,
                IsResidual=True):
    """
    This function is to stack different cycle of first derivation into a whole map,
    and then derive the partial derivation along north and east direction,
    finally combined the north and east direction to get the second derivation
    :param BeginCycle:
    :param EndCycle:
    :param InputPath:
    :param Background:
    :param IsResidual:
    :return:
    """
    all_bg_da,all_bg_dc = [],[]
    all_res_da,all_res_dc = [],[]
    spacing_res = 2000

    for i in np.arange(BeginCycle,EndCycle+1):
        da_bg = xr.open_dataset(InputPath+f'Slope_{Background.name}_{i:03d}_584_Along_Track.nc')
        dc_bg = xr.open_dataset(InputPath + f'Slope_{Background.name}_{i:03d}_584_Cross_Track.nc')
        all_bg_da.append(da_bg['z'].values)
        all_bg_dc.append(dc_bg['z'].values)
        lat, lon = da_bg['lat'].values, da_bg['lon'].values

        if IsResidual:
            da_res = xr.open_dataset(InputPath +f'Slope_d{Background.name}_{i:03d}_584_Along_Track.nc')
            dc_res = xr.open_dataset(InputPath +f"Slope_d{Background.name}_{i:03d}_584_Cross_Track.nc")
            all_res_da.append(da_res['z'].values)
            all_res_dc.append(dc_res['z'].values)

    all_bg_da,all_bg_dc = np.array(all_bg_da),np.array(all_bg_dc)
    # mean_mod_da = np.mean(all_bg_da,axis=0,keepdims=True)
    # mean_mod_dc = np.mean(all_bg_dc,axis=0,keepdims=True)
    # all_mod_da = all_mod_da - mean_mod_da
    # all_mod_dc = all_mod_dc - mean_mod_dc
    # all_mod = all_mod_da+all_mod_dc
    # print(f"all models da/dc/sum :{all_mod_da.shape},{all_mod_dc.shape},{all_mod.shape}")
    stack_bg_da = np.nanmean(all_bg_da,axis=0)
    stack_bg_dc = np.nanmean(all_bg_dc,axis=0)
    bg_daa,bg_dac = derivative_2D(data=stack_bg_da,spacing=spacing_res,pass_id=1)
    bg_dca,bg_dcc = derivative_2D(data=stack_bg_dc,spacing=spacing_res,pass_id=1)
    vgg_bg = -(bg_daa+bg_dcc)*9.80665

    if IsResidual:
        all_res_da,all_res_dc = np.array(all_res_da),np.array(all_res_dc)
        """Corrected phasescreen(instrument noise)"""
        mean_res_dc = np.nanmean(all_res_dc,axis=0,keepdims=True)
        all_res_dc = all_res_dc - mean_res_dc
        # mean_resi_da = np.nanmean(all_resi_da,axis=0,keepdims=True)
        # all_resi_da = all_resi_da - mean_resi_da
        # print(f"all residuals da/dc/mean:{all_resi_dc.shape},{all_resi_dc.shape},{mean_resi_dc.shape}")
        # all_resi = all_resi_da+all_resi_dc
        stack_res_da = np.nanmean(all_res_da, axis=0)
        stack_res_dc = np.nanmean(all_res_dc, axis=0)
        res_daa,res_dac = derivative_2D(data=stack_res_da,spacing=spacing_res,pass_id=1)
        res_dca,res_dcc = derivative_2D(data=stack_res_dc,spacing=spacing_res,pass_id=1)
        vgg_res = -(res_daa+res_dcc)*9.80665


        vgg = vgg_res+vgg_bg
        result = {"vgg_bg":vgg_bg,
                  "vgg_res":vgg_res,
                  "vgg":vgg,
                  "lat":lat,
                  "lon":lon}
    else:
        result = {"vgg_bg":vgg_bg,
                  "vgg_res":None,
                  "vgg":None,
                  "lat":lat,
                  "lon":lon}
    return result

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
    d2r = np.pi / 180.0
    cost = np.cos(lat * d2r)
    lon = np.unwrap(lon, period=360)

    dna, dnc = np.gradient(lat, 1, 1)
    dea, dec = np.gradient(lon, 1, 1) * cost
    dsa = (dea * dea + dna * dna) ** 0.5
    uea = dea / dsa
    una = dna / dsa
    dsc = (dec * dec + dnc * dnc) ** 0.5
    uec = dec / dsc
    unc = dnc / dsc

    return uea, una, uec, unc





    pass

def demo_vgg():
    """
    Before run this please make sure the relative Slopes data have been generated.
    BGtype.EGM2008 is to get VGG from EGM2008 & dEGM2008;
    BGtype.MSS is to get VGG from MSS & dMSS;
    """
    compute_vgg(BeginCycle=25,EndCycle=31,InputPath="../../../result/SWOT/SlopeSEIR/",
                Background=BGtype.EGM2008,IsResidual=True)

def demo_slope():
    """
    This is demo to compute certain region slopes,
    these slopes are based on pure Geoid model, MSS model, and
    residual slopes based on SSH-EGM2008 & SSH-MSS-MDT,
    final results will save in OutputPath with file name like:
    /Slope_EGM2008_{cycle_id}_{pass_numbers}_{track_direction}.nc/
    /Slope_MSS_{cycle_id}_{pass_numbers}_{track_direction}.nc/
    /Slope_dEGM2008_{cycle_id}_{pass_numbers}_{track_direction}.nc/
    /Slope_dMSS_{cycle_id}_{pass_numbers}_{track_direction}.nc/
    """
    begin_cycle, end_cycle = 26, 31
    pass_total_number = 584
    research_area = [60,90,-45,-30]
    for i in np.arange(begin_cycle, end_cycle+1):
        for track_direction in [TrackType.Along_Track,TrackType.Cross_Track]:
            print(f"> > Computing {i}th cycle,{track_direction}")
            compute_slope_region(cycle_id=i, pass_total=pass_total_number, region=research_area,
                                 InputPath="/Users/zhang.17371/Documents/Datasets/SWOT/L2_LR_SSH/",
                                 OutputPath="../../../result/SWOT/SlopeSEIR/",
                                 IsSave=True,
                                 Track=track_direction)

if __name__ == "__main__":
    demo_slope()