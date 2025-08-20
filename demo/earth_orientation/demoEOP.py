import numpy as np
import xarray as xr
import pandas as pd
from datetime import date

from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
import SaGEA.auxiliary.preference.EnumClasses as Enums

from pysrc.earth_rotation.EarthOrientaition import EOP


def demo1():
    """this demo shows an example of polar motion (both mass term and motion term) estimation, excitation is from atmosphere"""

    from tqdm import tqdm
    from pysrc.earth_rotation.PolarMotion import AAM_motion_term
    lmax=60
    begin_date, end_date = date(2009,1,1),date(2009,12,31)
    begin, end = begin_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d")
    gaa_dir,gaa_key = FileTool.get_project_dir("data/L2_SH_products/GAA/GFZ/RL06/BC01/"),"GRCOF2"
    shc_gaa = load_SHC(gaa_dir,key=gaa_key,lmax=lmax,begin_date=begin_date,end_date=end_date)
    shc_gaa.de_background()
    shc_gaa.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,to_type=Enums.PhysicalDimensions.EWH)

    date_range = pd.date_range(start=begin,end=end,freq="MS").strftime("%Y%m").tolist()
    u,v = [],[]
    for i in tqdm(date_range):
        u_wind_temp = xr.open_dataset(f"../../data/ERA5/pressure level/u_wind-{i}.nc")
        v_wind_temp = xr.open_dataset(f"../../data/ERA5/pressure level/v_wind-{i}.nc")
        u_wind = u_wind_temp['u'].values[0]
        v_wind = v_wind_temp['v'].values[0]
        u.append(u_wind)
        v.append(v_wind)
    pressure = u_wind_temp['pressure_level'].values*100
    lats = u_wind_temp['latitude'].values
    lons = u_wind_temp['longitude'].values
    u = np.array(u)
    v = np.array(v)
    mean_u = np.mean(u,axis=0)
    mean_v = np.mean(v,axis=0)
    u = u - mean_u[None,:,:,:]
    v = v - mean_v[None,:,:,:]


    AOM_mass = EOP().PM_mass_term(SH=shc_gaa.value,isMas=True)
    AOM_motion = EOP().PM_motion_term(u_speed=u,v_speed=v,lat=lats,lon=lons,pressure=pressure,isMas=True)
    a = AAM_motion_term(u=u,v=v,lat=lats,lon=lons,pl=pressure,isMas=True)
    mass_chi1 = AOM_mass['chi1']
    motion_chi1 = AOM_motion['chi1']
    a_motion = a['chi1']
    print(f"mass term:\n{mass_chi1}\nmotion term:\n{motion_chi1}\na:\n{a_motion}")

def demo2():
    """this demo shows an example of length-of-day (both mass term and motion term) estimation, excitation is from atmosphere"""

    lmax = 60
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    begin, end = begin_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    gaa_dir, gaa_key = FileTool.get_project_dir("data/L2_SH_products/GAA/GFZ/RL06/BC01/"), "GRCOF2"
    shc_gaa = load_SHC(gaa_dir, key=gaa_key, lmax=lmax, begin_date=begin_date, end_date=end_date)
    shc_gaa.de_background()
    # shc_gaa.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)

    date_range = pd.date_range(start=begin,end=end,freq='MS').strftime("%Y%m").tolist()
    u_set = []
    for i in date_range:
        u_temp = xr.open_dataset(f"../../data/ERA5/pressure level/u_wind-{i}.nc")
        u_set.append(u_temp['u'].values[0])
    lats = u_temp['latitude'].values
    lons = u_temp['longitude'].values
    pressure = u_temp['pressure_level'].values*100
    u_set = np.array(u_set)
    u_mean = np.mean(u_set,axis=0)
    u_set = u_set-u_mean


    AOM_LOD_mass = EOP().LOD_mass_term(SH=shc_gaa.value,isMas=True)
    AOM_LOD_motion = EOP().LOD_motion_term(u_speed=u_set,lat=lats,lon=lons,pressure=pressure,isMas=True)

    print(f"{AOM_LOD_mass['LOD']}\n{AOM_LOD_mass['chi3']}")
    print('-------------------------------')
    print(f"{AOM_LOD_motion['LOD']}\n{AOM_LOD_motion['chi3']}")


if __name__ == '__main__':
    demo2()