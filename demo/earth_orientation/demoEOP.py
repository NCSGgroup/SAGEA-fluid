import os
import numpy as np
import xarray as xr
import pandas as pd
from datetime import date

from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from SaGEA.auxiliary.aux_tool.TimeTool import TimeTool
from SaGEA.auxiliary.load_file.LoadL2LowDeg import load_low_degs
from SaGEA.post_processing.harmonic.Harmonic import Harmonic
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
import SaGEA.auxiliary.preference.EnumClasses as Enums

from pysrc.aliasing_model.specify.IBcorrection import IBcorrection
from pysrc.ancillary.load_file.DataClass import GRID, SHC
from pysrc.earth_rotation.EarthOrientaition import EOP
from pysrc.ancillary.constant.Setting import EAMType
from tqdm import tqdm

from pysrc.sealevel_equation.SeaLevelEquation import PseudoSpectralSLE


def demo_PM_mass_term():
    """this demo shows estimation of mass term of polar motion by SAGEA-fluid,
       taking the numerical model and calculation of AAM and OAM as an example """
    lmax,res,grace_lmax = 180,0.5,60
    begin_date,end_date = date(2009,1,1),date(2009,12,31)

    begin_str,end_str = begin_date.strftime("%Y-%m"),end_date.strftime("%Y-%m")
    date_range = pd.date_range(start=begin_str,end=end_str,freq='MS').strftime("%Y-%m").tolist()
    OBP_SH,ASP_SH = [],[]
    shift_amount = int(-360*0.5/res)
    for i in tqdm(date_range):
        temp_ocean = xr.open_dataset(f"../../data/ECCO/OBP/OCEAN_BOTTOM_PRESSURE_mon_mean_{i}_ECCO_V4r4b_latlon_0p50deg.nc")
        temp_obp = temp_ocean['OBP'].values
        temp_obp = np.nan_to_num(temp_obp,nan=0)
        temp_obp = GRID(grid=temp_obp,lat=temp_ocean['latitude'].values,lon=temp_ocean['longitude'].values).to_SHC(lmax=lmax)
        OBP_SH.append(temp_obp.value[0])

        year,month = i.split('-')[0],i.split('-')[1]
        temp_atmos = xr.open_dataset(f"../../data/ERA5/pressure level/sp-{year}{month}.nc")
        temp_asp = np.roll(temp_atmos['sp'].values,shift=shift_amount,axis=2)
        atmos_lat,atmos_lon = temp_atmos['latitude'].values,temp_atmos['longitude'].values-180
        ib = IBcorrection(lat=atmos_lat, lon=atmos_lon)
        asp_ib_f = ib.correct(grids=temp_asp.flatten())
        asp_ib = asp_ib_f.reshape(len(temp_asp[:,0,0]),len(temp_asp[0,:,0]),len(temp_asp[0,0,:]))

        har = Harmonic(lat=atmos_lat,lon=atmos_lon,lmax=lmax,option=1)
        C,S = har.analysis(gqij=asp_ib)
        sh_asp = SHC(c=C,s=S).convert_type(from_type=Enums.PhysicalDimensions.Pressure,to_type=Enums.PhysicalDimensions.EWH)
        ASP_SH.append(sh_asp.value[0])

    ASP_SH = np.array(ASP_SH)
    ASP_SH = SHC(c=ASP_SH)
    ASP_SH.de_background()
    AAM_ERA5 = EOP().PM_mass_term(SH=ASP_SH.value,isMas=False)


    OBP_SH_N = np.array(OBP_SH)
    OBP_SH_N = SHC(c=OBP_SH_N)
    OBP_SH_N.de_background()
    OBP_SH = OBP_SH_N.value
    OBP_SH[:,0] = 0
    OAM_ECCO = EOP().PM_mass_term(SH=OBP_SH,isMas=False)

    print(f"==============Mass term of Polar motion==============\n\n"
          f"AAM chi1 (mas):\n{AAM_ERA5['chi1']}\n"
          f"AAM chi2 (mas):\n{AAM_ERA5['chi2']}\n"
          f"OAM chi2 (mas):\n{OAM_ECCO['chi1']}\n"
          f"OAM chi2 (mas):\n{OAM_ECCO['chi2']}\n")

    save_path = '../../result/EOP/'
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    np.savez(f"{save_path}/PM_AAM_mass_term.npz",chi1=AAM_ERA5['chi1'],chi2=AAM_ERA5['chi2'])
    np.savez(f"{save_path}/PM_OAM_mass_term.npz", chi1=OAM_ECCO['chi1'], chi2=OAM_ECCO['chi2'])

def demo_PM_motion_term():
    """this demo shows estimation of motion term of polar motion by SAGEA-fluid,
       taking the calculation of AAM and OAM as an example """

    begin_date,end_date = date(2009,1,1),date(2009,12,31)
    begin_str, end_str = begin_date.strftime("%Y-%m"), end_date.strftime("%Y-%m")
    date_range = pd.date_range(start=begin_str, end=end_str, freq='MS').strftime("%Y-%m").tolist()
    ATM_SP,ATM_U,ATM_V, OCN_U,OCN_V,OCN_SSH =[],[],[], [],[],[]

    for i in tqdm(date_range):
        year, month = i.split('-')[0], i.split('-')[1]
        atmos_sp = xr.open_dataset(f"../../data/ERA5/pressure level/sp-{year}{month}.nc")
        atmos_u = xr.open_dataset(f"../../data/ERA5/pressure level/u_wind-{year}{month}.nc")
        atmos_v = xr.open_dataset(f"../../data/ERA5/pressure level/v_wind-{year}{month}.nc")
        atm_lat = atmos_u['latitude'].values
        atm_lon = atmos_u['longitude'].values
        sp_atm = atmos_sp['sp'].values
        u_atm = atmos_u['u'].values
        v_atm = atmos_v['v'].values
        pressure = atmos_u['pressure_level'].values*100
        ATM_U.append(u_atm[0])
        ATM_V.append(v_atm[0])
        ATM_SP.append(sp_atm[0])

        ocean_vel = xr.open_dataset(f"../../data/ECCO/VEL/OCEAN_VELOCITY_mon_mean_{i}_ECCO_V4r4_latlon_0p50deg.nc")
        ocean_ssh = xr.open_dataset(f"../../data/ECCO/SSH/SEA_SURFACE_HEIGHT_mon_mean_{i}_ECCO_V4r4b_latlon_0p50deg.nc")
        ssh = ocean_ssh['SSH'].values
        ssh = np.nan_to_num(ssh, nan=0)
        z_length = ocean_vel['Z'].values
        u_ocean = ocean_vel['EVEL'].values
        v_ocean = ocean_vel['WVEL'].values
        ocn_lat, ocn_lon = ocean_ssh['latitude'].values, ocean_ssh['longitude'].values
        u_ocean = np.nan_to_num(u_ocean, nan=0)
        v_ocean = np.nan_to_num(v_ocean, nan=0)
        OCN_SSH.append(ssh[0])
        OCN_V.append(v_ocean[0])
        OCN_U.append(u_ocean[0])



    ATM_U,ATM_V,ATM_SP = np.array(ATM_U),np.array(ATM_V),np.array(ATM_SP)
    atm_u_mean,atm_v_mean = np.mean(ATM_U,axis=0),np.mean(ATM_V,axis=0)
    ATM_U,ATM_V = ATM_U-atm_u_mean,ATM_V-atm_v_mean
    AAM_motion_term = EOP().PM_motion_term(u_speed=ATM_U,v_speed=ATM_V,layer=pressure,surf=ATM_SP,
                                           lat=atm_lat,lon=atm_lon,type=EAMType.AAM,isMas=False)

    OCN_U, OCN_V, OCN_SSH = np.array(OCN_U), np.array(OCN_V), np.array(OCN_SSH)
    ocn_u_mean, ocn_v_mean = np.mean(OCN_U, axis=0), np.mean(OCN_V, axis=0)
    OCN_U, OCN_V = OCN_U - ocn_u_mean, OCN_V - ocn_v_mean
    OAM_motion_term = EOP().PM_motion_term(u_speed=OCN_U, v_speed=OCN_V, lat=ocn_lat, lon=ocn_lon,
                                           type=EAMType.OAM, layer=z_length, surf=OCN_SSH, isMas=False)

    print(f"==============Motion term of Polar motion==============\n"
          f"AAM chi1 (mas):\n{AAM_motion_term['chi1']}\n"
          f"AAM chi2 (mas):\n{AAM_motion_term['chi2']}\n"
          f"OAM chi2 (mas):\n{OAM_motion_term['chi1']}\n"
          f"OAM chi2 (mas):\n{OAM_motion_term['chi2']}\n")

    save_path = '../../result/EOP/'
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    np.savez(f"{save_path}/PM_AAM_motion_term.npz",chi1=AAM_motion_term['chi1'],chi2=AAM_motion_term['chi2'])
    np.savez(f"{save_path}/PM_OAM_motion_term.npz", chi1=OAM_motion_term['chi1'], chi2=OAM_motion_term['chi2'])


def demo_LOD_mass_term():
    """this demo shows estimation of mass term of  length of day by SAGEA-fluid,
       taking the calculation of HIAM and SLAM as an example """

    lmax, res, grace_lmax = 180, 0.5, 60
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    gsm_dir, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    low_deg_dir = FileTool.get_project_dir("data/L2_low_degrees/")
    rep_deg1, rep_c20, rep_c30 = True, True, True
    gia_filepath = FileTool.get_GIA_path(gia_type=Enums.GIAModel.Caron2018)
    filter_method, filter_params = Enums.SHCFilterType.DDK, (3,)
    shc, dates_begin, dates_end = load_SHC(gsm_dir, key=gsm_key, lmax=grace_lmax, begin_date=begin_date,
                                           end_date=end_date,
                                           get_dates=True, )  # load GSM and dates
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)  # get average dates

    low_degs = load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN11, )  # load c20
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN13,
                                  institute=Enums.L2InstituteType.CSR, release=Enums.L2Release.RL06))  # load degree-1
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN14))  # load c20 (update) and c30
    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=grace_lmax)  # load GIA trend
    shc_gia = shc_gia_trend.expand(dates_ave)  #
    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=rep_deg1, c20=rep_c20,
                         c30=rep_c30)  # replace SHC low_degrees

    shc.subtract(shc_gia)  # subtracting gia model
    shc.de_background()  # de-average if non-input
    shc.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,
                     to_type=Enums.PhysicalDimensions.EWH)  # convert physical dimension
    shc.filter(method=filter_method, param=filter_params)

    '''load ocean mask'''
    basin_path_SH = FileTool.get_project_dir("data/basin_mask/SH/Ocean_maskSH.dat")
    shc_basin = load_SHC(basin_path_SH, key='', lmax=grace_lmax)
    grid_basin = shc_basin.to_grid(grid_space=res)
    grid_basin.limiter(threshold=0.5)
    mask_ocean = grid_basin.value[0]

    lat, lon = MathTool.get_global_lat_lon_range(resolution=res)

    '''Get RSL'''
    SAL = PseudoSpectralSLE(SH=shc.value,lmax=lmax)
    SAL.setLoveNumber(lmax=lmax,method=Enums.LLN_Data.Wang,frame=Enums.Frame.CM)
    SAL.setLatLon(lat=lat,lon=lon)
    RSL = SAL.SLE(mask=mask_ocean,rotation=True,isLand=True)['RSL_SH']
    RSL_SH = SHC(c=RSL).convert_type(from_type=Enums.PhysicalDimensions.EWH,to_type=Enums.PhysicalDimensions.Dimensionless)

    GSM_grid = shc.to_grid(res)
    GSM_grid = GSM_grid.value * (1-mask_ocean)
    GSM_SH = GRID(grid=GSM_grid, lat=lat, lon=lon).to_SHC(lmax=grace_lmax)
    GSM_Stokes = GSM_SH.convert_type(from_type=Enums.PhysicalDimensions.EWH,to_type=Enums.PhysicalDimensions.Dimensionless)


    HIAM_LOD_GRACE = EOP().LOD_mass_term(SH=GSM_Stokes.value,isMs=False)
    SLAM_LOD = EOP().LOD_mass_term(SH=RSL_SH.value,isMs=False)

    print(f"==============Mass term of Polar motion==============\n\n"
          f"HIAM chi3 (mas):\n{HIAM_LOD_GRACE['chi3']}\n"
          f"SLAM chi3 (mas):\n{SLAM_LOD['chi3']}\n")

    save_path = '../../result/EOP/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(f"{save_path}/LOD_HIAM_mass_term.npz", chi3=HIAM_LOD_GRACE['chi3'], LOD=HIAM_LOD_GRACE['LOD'])
    np.savez(f"{save_path}/LOD_SLAM_mass_term.npz", chi3=SLAM_LOD['chi3'], LOD=SLAM_LOD['LOD'])

def demo_LOD_motion_term():
    """this demo shows estimation of motion term of length of day by SAGEA-fluid,
       taking the calculation of AAM and OAM as an example """
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    begin_str, end_str = begin_date.strftime("%Y-%m"), end_date.strftime("%Y-%m")
    date_range = pd.date_range(start=begin_str, end=end_str, freq='MS').strftime("%Y-%m").tolist()
    ATM_SP, ATM_U, ASP_Grid, OCN_U,OCN_V, OCN_SSH =[],[],[],[],[],[]
    for i in tqdm(date_range):
        year, month = i.split('-')[0], i.split('-')[1]
        atmos_sp = xr.open_dataset(f"../../data/ERA5\pressure level/sp-{year}{month}.nc")
        atmos_u = xr.open_dataset(f"../../data/ERA5\pressure level/u_wind-{year}{month}.nc")
        atm_lat = atmos_u['latitude'].values
        atm_lon = atmos_u['longitude'].values-180
        sp_atm = atmos_sp['sp'].values
        u_atm = atmos_u['u'].values
        pressure = atmos_u['pressure_level'].values * 100
        ATM_U.append(u_atm[0])
        ATM_SP.append(sp_atm[0])

        ocean_vel = xr.open_dataset(f"../../data/ECCO\VEL/OCEAN_VELOCITY_mon_mean_{i}_ECCO_V4r4_latlon_0p50deg.nc")
        ocean_ssh = xr.open_dataset(f"../../data/ECCO/SSH/SEA_SURFACE_HEIGHT_mon_mean_{i}_ECCO_V4r4b_latlon_0p50deg.nc")
        ssh = ocean_ssh['SSH'].values
        ssh = np.nan_to_num(ssh, nan=0)
        z_length = ocean_vel['Z'].values
        u_ocean = ocean_vel['EVEL'].values
        ocn_lat, ocn_lon = ocean_ssh['latitude'].values, ocean_ssh['longitude'].values
        u_ocean = np.nan_to_num(u_ocean, nan=0)
        OCN_SSH.append(ssh[0])
        OCN_U.append(u_ocean[0])



    ATM_U, ATM_SP = np.array(ATM_U), np.array(ATM_SP)
    u_mean = np.mean(ATM_U)
    ATM_U = ATM_U - u_mean
    AAM_motion_term = EOP().LOD_motion_term(u_speed=ATM_U, surf=ATM_SP, layer=pressure,
                                        lat=atm_lat, lon=atm_lon, type=EAMType.AAM, isMs=False)
    OCN_U, OCN_SSH = np.array(OCN_U), np.array(OCN_SSH)
    ocn_u_mean= np.mean(OCN_U, axis=0)
    OCN_U= OCN_U - ocn_u_mean
    OAM_motion_term = EOP().LOD_motion_term(u_speed=OCN_U, lat=ocn_lat, lon=ocn_lon,
                                            type=EAMType.OAM, layer=z_length, surf=OCN_SSH, isMs=False)

    print(f"==============Motion term of Polar motion==============\n\n"
          f"AAM chi3 (mas):\n{AAM_motion_term['chi3']}\n"
          f"OAM chi3 (mas):\n{OAM_motion_term['chi3']}\n")


    save_path = '../../result/EOP/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(f"{save_path}/LOD_AAM_motion_term.npz", chi3=AAM_motion_term['chi3'], LOD=AAM_motion_term['LOD'])
    np.savez(f"{save_path}/LOD_OAM_motion_term.npz", chi3=OAM_motion_term['chi3'], LOD=OAM_motion_term['LOD'])



if __name__ == '__main__':
    # demo_PM_mass_term()
    # demo_PM_motion_term()
    # demo_LOD_mass_term()
    demo_LOD_motion_term()