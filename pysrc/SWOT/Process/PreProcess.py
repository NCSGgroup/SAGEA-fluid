# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : PreProcess.py
# @Create  : 10/24/26 21:17
# @Project : PySWOT
# @Software: PyCharm
"""
Description:
"""
import numpy as np
from pysrc.SWOT.KaRIn.L2LRSSH.SSH_Expert import Load_SSH
from pysrc.SWOT.SWOTKits.Derivation import derivative_2D,derivative_1D
from pysrc.SWOT.Settings.SWOTClasses import TrackType

def data_filter(cNo=30,pNo=10,InputPath='/Users/zhang.17371/Documents/Datasets/SWOT/'):
    max_swh = 6
    res = 0.02
    # cross_indices = [4521, 4674, 4778]
    # along_indices = [13, 27]
    datasets = Load_SSH(InputPath=InputPath).get_ssh_pass(cycle_No=cNo, pass_No=pNo)
    if datasets is None:
        return None
    else:
        mss, mdt = datasets['mss_cnescls'], datasets['mdt']
        tides, dac = datasets['tides'], datasets['dac']
        ssh, ssh_qual = datasets['ssh'], datasets['ssh_qual']
        swh, ged = datasets['swh'],datasets['geoid']
        xover, x_qual = datasets['height_cor_xover'], datasets['height_cor_xover_qual']
        ssh_clean = ssh - tides - dac
        lat, lon = datasets['lat'], datasets['lon']

        """Get the valid index"""
        valid_1 = swh < max_swh  # Yu, high-quality swh<6m
        valid_2 = ssh_qual < 1  # Zhang, ssh_flag_qual=0 "good"
        valid_3 = x_qual < 1  # Zhang, height_xover, qual=0 "good"
        # print(f"original ssh_value:\n{ssh_clean}")
        # ssh_value = ssh_clean
        ssh_value = np.where(valid_1, ssh_clean, np.nan)  # Yu, filter swh>6m in ssh
        ssh_value = np.where(valid_2, ssh_value, np.nan)  # Zhang, only keep ssh_qual=0
        ssh_value = np.where(valid_3, ssh_value, np.nan)  # Zhang, only keep height_xover_qual=0
        # print(ssh_value)
        mss_value = np.where(~np.isnan(ssh_value), mss, np.nan)
        mdt_value = np.where(~np.isnan(ssh_value), mdt, np.nan)
        xover_value = np.where(~np.isnan(ssh_value), xover, np.nan)
        ged_value = np.where(~np.isnan(ssh_value), ged, np.nan)

        ssh_value = ssh_value + xover_value  # Zhang, add the height_xover_correction into ssh

        ssh_da,ssh_dc = derivative_2D(data=ssh_value,spacing=res,pNo=pNo)
        mss_da,mss_dc = derivative_2D(data=mss_value,spacing=res,pNo=pNo)
        mdt_da,mdt_dc = derivative_2D(data=mdt_value,spacing=res,pNo=pNo)
        ged_da,ged_dc = derivative_2D(data=ged_value,spacing=res,pNo=pNo)
        results = {"ssh":ssh_value,
                   "mss":mss_value,
                   "mdt":mdt_value,
                   "geoid":ged_value,
                   "ssh_da":ssh_da,"ssh_dc":ssh_dc,
                   "mss_da":mss_da,"mss_dc":mss_dc,
                   "mdt_da":mdt_da,"mdt_dc":mdt_dc,
                   "ged_da":ged_da,"ged_dc":ged_dc,
                   "lat":lat,"lon":lon}
        return results

def data_unfliter(cNo=30,pNo=10,InputPath='/Users/zhang.17371/Documents/Datasets/SWOT/'):
    max_swh = 6
    res = 0.02
    # cross_indices = [4521, 4674, 4778]
    # along_indices = [13, 27]
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

        """Get the valid index"""
        # valid_1 = swh < max_swh  # Yu, high-quality swh<6m
        # valid_2 = ssh_qual < 1  # Zhang, ssh_flag_qual=0 "good"
        # valid_3 = x_qual < 1  # Zhang, height_xover, qual=0 "good"
        # # print(f"original ssh_value:\n{ssh_clean}")
        # # ssh_value = ssh_clean
        # ssh_value = np.where(valid_1, ssh_clean, np.nan)  # Yu, filter swh>6m in ssh
        # ssh_value = np.where(valid_2, ssh_value, np.nan)  # Zhang, only keep ssh_qual=0
        # ssh_value = np.where(valid_3, ssh_value, np.nan)  # Zhang, only keep height_xover_qual=0
        # # print(ssh_value)
        # mss_value = np.where(~np.isnan(ssh_value), mss, np.nan)
        # mdt_value = np.where(~np.isnan(ssh_value), mdt, np.nan)
        # xover_value = np.where(~np.isnan(ssh_value), xover, np.nan)
        #
        # ssh_value = ssh_value + xover_value  # Zhang, add the height_xover_correction into ssh
        # ssh_value

        ssh_da, ssh_dc = derivative_2D(data=ssh_value, spacing=res,pNo=pNo)
        mss_da, mss_dc = derivative_2D(data=mss_value, spacing=res,pNo=pNo)
        mdt_da, mdt_dc = derivative_2D(data=mdt_value, spacing=res,pNo=pNo)
        ged_da, ged_dc = derivative_2D(data=ged_value, spacing=res,pNo=pNo)
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

def ssh_geoid(cNo=30,pNo=10,InputPath='/Users/zhang.17371/Documents/Datasets/SWOT/'):
    max_swh = 6 # maximum significant wave height is 6 m
    res = 0.02  # interval distance along-track/cross-track is 2km=2000m
    datasets = (Load_SSH(InputPath=InputPath).
                get_ssh_pass(cycle_No=cNo,pass_No=pNo)) # get the SWOT a specific Expert pass data
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

        valid_1 = swh < max_swh
        print(f"valid_1 information: {valid_1.shape}\n{valid_1}\n")
        valid_2 = ssh_qual < 1
        print(f"valid_2 information: {valid_2.shape}\n{valid_2}\n")
        valid_3 = x_qual < 1
        print(f"valid_3 information: {valid_3.shape}\n{valid_3}\n")

def data_second(cNo=30,InputPath='/Users/zhang.17371/Documents/Datasets/SWOT/',filter=True):

    datasets = Load_SSH(InputPath=InputPath).get_ssh_cycle(cycle_No=cNo)
    ssh = datasets['ssh']
    lat,lon = datasets['lat'],datasets['lon']
    print(ssh.shape,lat.shape,lon.shape)




    pass


if __name__ == "__main__":
    # ssh_mss_mdt()
    # ssh_geoid()
    data_second()

