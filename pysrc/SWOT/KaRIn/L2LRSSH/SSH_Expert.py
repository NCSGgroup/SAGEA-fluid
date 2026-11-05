# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : SSH_Expert.py
# @Create  : 10/2/2026 10:58 PM
# @Project : PySWOT
# @Software: PyCharm
"""
Description: This script is used to load the SWOT L2 LR SSH, mainly for Expert file,
since the Expert file including the Basic and WindWave files.
"""
import numpy as np
import xarray as xr
import glob
from pysrc.SWOT.Settings.SWOTClasses import KaRInL2LRSSH
from tqdm import tqdm

class Load_SSH:
    def __init__(self,InputPath):
        self.load_path = InputPath
        self.cycle_No = None
        self.pass_No = None
        self.FileType = KaRInL2LRSSH.Expert

    def set_Orbit(self,cycle_No,pass_No):
        self.cycle_No = cycle_No
        self.pass_No = pass_No
        return self

    def set_FileType(self,FileType):
        self.FileType = FileType
        return self

    def _return_pass_path(self,cNo,pNo):
        cycle_str = f"{cNo:03d}"
        pass_str = f"{pNo:03d}"
        karin_type = self.FileType.name
        target_str = f"{karin_type}_{cycle_str}_{pass_str}"
        # print(target_str)
        file_pattern = f"{self.load_path}/**/*{target_str}*.nc"
        nc_file = glob.glob(file_pattern,recursive=True)
        # print(nc_file)
        return nc_file

    def _return_ssh(self,data_path:list):
        xr_data = xr.open_dataset(data_path[0])
        time_array = xr_data['time'].values
        t = np.asarray(time_array, dtype="datetime64[ns]")
        dt = (t - t[0]) / np.timedelta64(1, "s")
        if self.FileType == KaRInL2LRSSH.Basic:
            data_dic = {"ssh": xr_data['ssh_karin'].values,
                        "ssha": xr_data['ssha'].values, }
        elif self.FileType == KaRInL2LRSSH.WaveWind:
            data_dic = {"swh": xr_data['swh_karin'].values}
        elif self.FileType == KaRInL2LRSSH.Unsmooth:
            data_dic = None
            print(f" > > Currently doesn't support to load Unsmoothed data!")
        else:
            data_dic = {"ssh": xr_data['ssh_karin'].values,
                        "ssh_qual": xr_data['ssh_karin_qual'].values,
                        "ssha": xr_data['ssha_karin'].values,
                        "lat": xr_data['latitude'].values,
                        "lon": xr_data['longitude'].values,
                        "tides": xr_data['ocean_tide_fes'].values +
                                 xr_data['ocean_tide_non_eq'].values +
                                 xr_data['pole_tide'].values +
                                 xr_data['solid_earth_tide'].values +
                                 xr_data["internal_tide_hret"].values,
                        "dac": xr_data['dac'].values,
                        "swh": xr_data['swh_karin'].values,
                        "mss_cnescls": xr_data['mean_sea_surface_cnescls'].values,
                        "mss_dtu": xr_data['mean_sea_surface_dtu'].values,
                        "geoid": xr_data['geoid'].values,
                        "mdt": xr_data['mean_dynamic_topography'].values,
                        "time": xr_data['time'].values,
                        "height_cor_xover": xr_data['height_cor_xover'].values,
                        "height_cor_xover_qual": xr_data['height_cor_xover_qual'].values,
                        "dt": dt,
                        }
        return data_dic

    def _return_cycle_paths(self,cNo):
        cycle_str = f"{cNo:03}"
        # print(f"cycle_No is:{cycle_str}")
        target_str = f"_{self.FileType.name}_{cycle_str}_"
        # print(target_str)
        file_pattern = f"{self.load_path}/**/*{target_str}*.nc"
        nc_files = glob.glob(file_pattern, recursive=True)
        # print(f"{nc_files},\nTotal {len(nc_files)} files!")

        ds_files = []
        for f in tqdm(nc_files,desc=f"> > Loading total {len(nc_files)} in {file_pattern}"):
            ds_temp = xr.open_dataset(f)
            ds_files.append(ds_temp)

        return ds_files

    def _return_whole_cycle(self,file_path:list):
        all_lon, all_lat, all_ssh = [], [], []
        all_ssh_qual, all_ssha = [],[]
        all_tides, all_dac, all_swh = [], [], []
        all_mssc, all_mssd, all_mdt = [],[],[]
        all_geo = []
        all_hcx,all_hcx_qual = [],[]

        for grid_data in tqdm(file_path,desc=f"> > Integrating total {len(file_path)} passes"):
            lon = grid_data['longitude'].values.flatten()
            lat = grid_data['latitude'].values.flatten()
            ssh = grid_data['ssh_karin'].values.flatten()
            ssh_qual = grid_data['ssh_karin_qual'].values.flatten()
            ssha = grid_data['ssha_karin'].values.flatten()
            tides = (grid_data['ocean_tide_fes'].values.flatten() +
                     grid_data['ocean_tide_non_eq'].values.flatten() +
                     grid_data['pole_tide'].values.flatten() +
                     grid_data['solid_earth_tide'].values.flatten() +
                     grid_data['internal_tide_hret'].values.flatten())
            dac = grid_data['dac'].values.flatten()
            swh = grid_data['swh_karin'].values.flatten()
            mss_cnescls = grid_data['mean_sea_surface_cnescls'].values.flatten()
            mss_dtu = grid_data['mean_sea_surface_dtu'].values.flatten()
            ged = grid_data['geoid'].values.flatten()
            mdt = grid_data['mean_dynamic_topography'].values.flatten()
            hcx = grid_data['height_cor_xover'].values.flatten()
            hcx_qual = grid_data['height_cor_xover_qual'].values.flatten()

            all_lon.append(lon)
            all_lat.append(lat)
            all_ssh.append(ssh)
            all_ssh_qual.append(ssh_qual)
            all_ssha.append(ssha)
            all_tides.append(tides)
            all_dac.append(dac)
            all_swh.append(swh)
            all_mssc.append(mss_cnescls)
            all_mssd.append(mss_dtu)
            all_geo.append(ged)
            all_mdt.append(mdt)
            all_hcx.append(hcx)
            all_hcx_qual.append(hcx_qual)
        # 合并为numpy数组（轻量）
        all_lon = np.concatenate(all_lon)
        all_lat = np.concatenate(all_lat)
        all_ssh = np.concatenate(all_ssh)
        all_ssh_qual = np.concatenate(all_ssh_qual)
        all_ssha = np.concatenate(all_ssha)
        all_tides = np.concatenate(all_tides)
        all_dac = np.concatenate(all_dac)
        all_swh = np.concatenate(all_swh)
        all_mssc = np.concatenate(all_mssc)
        all_mssd = np.concatenate(all_mssd)
        all_geo = np.concatenate(all_geo)
        all_hcx = np.concatenate(all_hcx)
        all_hcx_qual = np.concatenate(all_hcx_qual)
        # print(f"all_tides:{len(all_tides)},all_ssh:{len(all_ssh)}")
        data_dic = {"lat":all_lat,
                    "lon":all_lon,
                    "ssh":all_ssh,
                    "ssh_qual":all_ssh_qual,
                    "ssha":all_ssha,
                    "tides":all_tides,
                    "dac":all_dac,
                    "swh":all_swh,
                    "mss_cnescls":all_mssc,
                    "mss_dtu":all_mssd,
                    "geoid":all_geo,
                    "mdt":all_mdt,
                    "height_cor_xover":all_hcx,
                    "height_cor_xover_qual":all_hcx_qual}
        return data_dic


    def get_ssh(self):
        data_path = self._return_pass_path(cNo=self.cycle_No,pNo=self.pass_No)
        # print(data_path)
        if not data_path:
            print(f" > > There is no SWOT {self.FileType.name} data on cycle {self.cycle_No} and pass {self.pass_No}")
            return None

        else:
            print(f" > > Existing data_path as: {data_path[0]}")
            data_dict = self._return_ssh(data_path=data_path)
            return data_dict

    def get_ssh_pass(self,cycle_No=30,pass_No=10):
        # self.cycle_No = cycle_No
        # self.pass_No = pass_No
        data_path = self._return_pass_path(cNo=cycle_No,pNo=pass_No)
        if not data_path:
            print(f" > > There is no SWOT {self.FileType.name} data on cycle {cycle_No} and pass {pass_No}")
            return None

        else:
            # print(f" > > Existing data_path as: {data_path[0]}")
            data_dict = self._return_ssh(data_path=data_path)
            return data_dict

        pass

    def get_ssh_cycle(self,cycle_No=30):
        # self.cycle_No = cycle_No
        ds_files = self._return_cycle_paths(cNo=cycle_No)

        data_dic = self._return_whole_cycle(file_path=ds_files)
        return data_dic



def get_ssh_original(InputPath="D:/",cycle_No=30,pass_No=10,FileType=KaRInL2LRSSH.Expert):
    cycle_str = f"{cycle_No:03d}"   # set the cycle_number format as _xxx_
    orbit_str = f"{pass_No:03d}"    # set the pass_number format as _xxx_
    karin_type = FileType.name      # give the granules files that need to load, usually is Expert

    target_str = (f"{karin_type}_"
                  f"{cycle_str}_"
                  f"{orbit_str}_")  # This can set the identification like "Expert_xxx_xxx_" for better loading the file needed
    # print(target_str)
    file_pattern = (f"{InputPath}"
                    f"/**/"
                    f"*{target_str}*.nc")           # Set the pattern of the target path
    # print(file_pattern)
    nc_file = glob.glob(file_pattern,recursive=True)# To get the paths list which has the identification like {target_str}
    print(f"{nc_file}")
    xr_data = xr.open_dataset(nc_file[0])
    return xr_data

def get_ssh_quick_start(InputPath="D:/",cycle_No=30,pass_No=10,FileType=KaRInL2LRSSH.Expert):
    cycle_str = f"{cycle_No:03d}"   # set the cycle_number format as _xxx_
    orbit_str = f"{pass_No:03d}"    # set the pass_number format as _xxx_
    karin_type = FileType.name      # give the granules files that need to load, usually is Expert

    target_str = (f"{karin_type}_"
                  f"{cycle_str}_"
                  f"{orbit_str}_")  # This can set the identification like "Expert_xxx_xxx_" for better loading the file needed
    # print(target_str)
    file_pattern = (f"{InputPath}"
                    f"/**/"
                    f"*{target_str}*.nc")           # Set the pattern of the target path
    # print(file_pattern)
    nc_file = glob.glob(file_pattern,recursive=True)# To get the paths list which has the identification like {target_str}
    # print(f"{nc_file}")
    xr_data = xr.open_dataset(nc_file[0])
    # print(xr_data.variables)
    time_array = xr_data['time'].values
    # print((xr_data['time'].values).dtype)
    t = np.asarray(time_array,dtype="datetime64[ns]")
    dt = (t-t[0])/np.timedelta64(1,"s")
    # print(dt,dt.shape)
    print(f" > > Loading SWOT L2 LR SSH "
          f"{KaRInL2LRSSH.Expert.name} with cycle "
          f"{cycle_No} and pass {pass_No}...")

    data_dic = {"ssh":xr_data['ssh_karin'].values,
                "ssh_qual":xr_data['ssh_karin_qual'].values,
                "ssha":xr_data['ssha_karin'].values,
                "lat":xr_data['latitude'].values,
                "lon":xr_data['longitude'].values,
                "tides":xr_data['ocean_tide_fes'].values+
                        xr_data['ocean_tide_non_eq'].values+
                        xr_data['pole_tide'].values+
                        xr_data['solid_earth_tide'].values+
                        xr_data["internal_tide_hret"].values,
                "dac":xr_data['dac'].values,
                "swh":xr_data['swh_karin'].values,
                "mss_cnescls":xr_data['mean_sea_surface_cnescls'].values,
                "mss_dtu": xr_data['mean_sea_surface_dtu'].values,
                "geoid": xr_data['geoid'].values,
                "mdt":xr_data['mean_dynamic_topography'].values,
                "time":xr_data['time'].values,
                "height_cor_xover":xr_data['height_cor_xover'].values,
                "height_cor_xover_qual":xr_data['height_cor_xover_qual'].values,
                "dt":dt,
                }
    return data_dic

def demo():
    a = Load_SSH(InputPath='D:/Documents/Datasets/SWOT/').get_ssh_pass(cycle_No=30,pass_No=100)
    # a.set_FileType(FileType=KaRInL2LRSSH.Basic)
    # data_dic = a.get_ssh()
    # print(data_dic['ssh'])

def demo1():
    a = Load_SSH(InputPath='D:/Documents/Datasets/',)
    data = a.get_ssh_cycle(cycle_No=7)
    print(f"lat:{data['lat'].shape}\nlon:{data['lon'].shape}\n"
          f"ssh:{data['ssh'].shape}\nmss:{data['mss_cnescls'].shape}\n"
          f"tides:{data['tides'].shape}\ndac:{data['dac'].shape}")

if __name__ == "__main__":
    # get_ssh_quick_start(load_path="/Users/zhang.17371/Documents/Dataset/SWOT/",)
    # demo()
    demo1()