# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : SSH_Refined.py
# @Create  : 11/4/26 22:01
# @Project : PySWOT
# @Software: PyCharm
"""
Description:
"""
import os
import numpy as np
import pygmt
from tqdm import tqdm
from pysrc.KaRIn.L2LRSSH.SSH_Expert import Load_SSH
from pysrc.GeoMathKits.Derivation import derivative_2D
from pysrc.Settings.EnumClasses import TrackType,RefineMethod

class SSH_Refined:
    def __init__(self,InputPath='/Users/zhang.17371/Documents/Datasets/SWOT/'):
        self.InputPath = InputPath
        self.OutputPath = None
        self.res = 0.02          # SSH resolution is 2km, 2000m, and 0.02*10^5 km;
        self.cNp = None          # 21-day & 1-day cycle;
        self.pNo = None          # 1-584 for 21-day cycle, odd num is ascending, even num is descending;
        self.Refine_Method = RefineMethod.Zhang
        self.max_swh = 6
        pass
    def setOutputPath(self,OutputPath):
        self.OutputPath = OutputPath
        if not os.path.exists(self.OutputPath):
            os.makedirs(self.OutputPath)
        return self

    def setOrbit(self,cNo,pNo):
        self.cNo = cNo
        self.pNo = pNo
        return self

    def __setMethod(self,RefineType=RefineMethod.Yu):
        self.Refine_Method = RefineType
        if self.Refine_Method == RefineMethod.Yu:
            self.max_swh = 0
        elif self.Refine_Method == RefineMethod.Zhang:
            self.max_swh = 6
        return self

    def refine_data(self,cNo,pNo,RefineType=RefineMethod.Zhang):
        # self.__setMethod(RefineType=RefineType)
        # print(self.max_swh)
        datasets = Load_SSH(InputPath=self.InputPath).get_ssh_pass(cycle_No=cNo,pass_No=pNo)
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
            if RefineType == RefineMethod.Zhang:
                valid_1 = swh < self.max_swh
                valid_2 = ssh_qual < 1
                valid_3 = x_qual < 1
                ssh_value = np.where(valid_1, ssh_clean, np.nan)        # Yu, filter swh>6m in ssh
                ssh_value = np.where(valid_2, ssh_value, np.nan)        # Zhang, only keep ssh_qual=0
                ssh_value = np.where(valid_3, ssh_value, np.nan)        # Zhang, only keep height_xover_qual=0
                mss_value = np.where(~np.isnan(ssh_value), mss, np.nan)
                mdt_value = np.where(~np.isnan(ssh_value), mdt, np.nan)
                xover_value = np.where(~np.isnan(ssh_value), xover, np.nan)
                ged_value = np.where(~np.isnan(ssh_value), ged, np.nan)
                ssh_value = ssh_value + xover_value                     # Zhang, add the height_xover_correction into ssh
                delta_vle = ssh_value-ged_value-mdt_value               # Delta is SSH-Geoid-MDT
                de_da, de_dc = derivative_2D(data=delta_vle, spacing=self.res, pNo=self.pNo)
            elif RefineType == RefineMethod.Yu:
                valid_1 = swh < self.max_swh
                ssh_value = np.where(valid_1, ssh_clean, np.nan)  # Yu, filter swh>6m in ssh
                mss_value = np.where(~np.isnan(ssh_value), mss, np.nan)
                mdt_value = np.where(~np.isnan(ssh_value), mdt, np.nan)
                # xover_value = np.where(~np.isnan(ssh_value), xover, np.nan)
                ged_value = np.where(~np.isnan(ssh_value), ged, np.nan)
                delta_vle = ssh_value-mdt_value-mss_value               # Delta is SSH-MSS-MDT
                de_da,de_dc = derivative_2D(data=delta_vle, spacing=self.res, pNo=self.pNo)
            else:
                ssh_value,mss_value = ssh,mss
                mdt_value,ged_value = mdt,ged
                de_da,de_dc = None,None

            ssh_da, ssh_dc = derivative_2D(data=ssh_value, spacing=self.res, pNo=self.pNo)
            mss_da, mss_dc = derivative_2D(data=mss_value, spacing=self.res, pNo=self.pNo)
            mdt_da, mdt_dc = derivative_2D(data=mdt_value, spacing=self.res, pNo=self.pNo)
            ged_da, ged_dc = derivative_2D(data=ged_value, spacing=self.res, pNo=self.pNo)
            results = {"ssh": ssh_value,
                       "mss": mss_value,
                       "mdt": mdt_value,
                       "geoid": ged_value,
                       "ssh_da": ssh_da, "ssh_dc": ssh_dc,
                       "mss_da": mss_da, "mss_dc": mss_dc,
                       "mdt_da": mdt_da, "mdt_dc": mdt_dc,
                       "ged_da": ged_da, "ged_dc": ged_dc,
                       "de_da": de_da, "de_dc": de_dc,
                       "lat": lat, "lon": lon}
            return results

    def projection(self,cNo=30,TotalPass=584,region=None,
                   tracktype=TrackType.Cross_Track,
                   RefineMethod=RefineMethod.Zhang,):
        all_x, all_y, all_z = [], [], []
        if region is None:
            region = [0, 360, -90, 90]
        else:
            region = region

        for i in tqdm(np.arange(TotalPass), desc=f" > > Loading total pass {TotalPass} of cycle {cNo} Now ..."):
            datatemp = self.refine_data(cNo=cNo,pNo=i + 1,RefineType=RefineMethod)
            if datatemp is None:
                continue
            else:
                x_value = datatemp['lon'].flatten()
                y_value = datatemp['lat'].flatten()
                if tracktype == TrackType.Along_Track:
                    z_value = (datatemp['ssh_da'] - datatemp['ged_da'] - datatemp['mdt_da']).flatten()
                elif tracktype == TrackType.Cross_Track:
                    z_value = (datatemp['ssh_dc'] - datatemp['ged_dc'] - datatemp['mdt_da']).flatten()
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
        x, y, v = all_x[valid], all_y[valid], all_z[valid]
        print(f"============================================================\n")
        if self.OutputPath is None:
            grid = pygmt.xyz2grd(
                x=x,
                y=y,
                z=v,
                region=region,
                spacing="2k/2k",
                registration="p",
                coltypes="g",
                duplicate="m",
                # outgrid=OutputFile,
            )
            return grid
        else:
            OutputFile = os.path.join(self.OutputPath, f"delta_ssh_{cNo:03d}_{TotalPass}_{tracktype.name}.nc")
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
            return None

        pass
def demo():
    A = SSH_Refined(InputPath='/Users/zhang.17371/Documents/Datasets/SWOT/')
    A.setOrbit(cNo=30,pNo=100)
    # results = A.refine_data(RefineMethodefineMethod.Zhang)
    # print(results['de_da'].shape)
    r1 = A.projection(cNo=30,TotalPass=584)

if __name__ == "__main__":
    demo()
