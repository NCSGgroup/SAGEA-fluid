import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import date
import lib.SaGEA.auxiliary.preference.EnumClasses as Enums
from lib.SaGEA.auxiliary.aux_tool.FileTool import FileTool
from lib.SaGEA.auxiliary.aux_tool.TimeTool import TimeTool
from lib.SaGEA.auxiliary.aux_tool.MathTool import MathTool
from lib.SaGEA.auxiliary.load_file.LoadL2LowDeg import load_low_degs
from lib.SaGEA.auxiliary.load_file.LoadL2SH import load_SHC

from pysrc.AD.specify.IBcorrection import IBcorrection
from lib.SaGEA.data_class.DataClass import GRID, SHC
from pysrc.SAL.SeaLevelEquation import PseudoSpectralSLE
from tqdm import tqdm
from lib.SaGEA.post_processing.harmonic.Harmonic import Harmonic
def demo_NM():
    """This is an example for computing SAL effect using numerical model, i.e., ERA5 surface pressure and ECCO ocean bottom pressure"""

    '''Setting configuration for loading numerical models'''
    lmax, res, grace_lmax = 180, 0.5, 60
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    begin_str, end_str = begin_date.strftime("%Y-%m"), end_date.strftime("%Y-%m")
    date_range = pd.date_range(start=begin_str, end=end_str, freq='MS').strftime("%Y-%m").tolist()
    OBP_SH, ASP_SH = [], []
    shift_amount = int(-360 * 0.5 / res)
    for i in tqdm(date_range):
        temp_ocean = xr.open_dataset(f"../../data/ECCO/OBP/OCEAN_BOTTOM_PRESSURE_mon_mean_{i}_ECCO_V4r4b_latlon_0p50deg.nc")
        temp_obp = temp_ocean['OBP'].values
        temp_obp = np.nan_to_num(temp_obp, nan=0)
        ocean_lat,ocean_lon = temp_ocean['latitude'].values,temp_ocean['longitude'].values
        temp_obp = GRID(grid=temp_obp, lat=ocean_lat, lon=ocean_lon).to_SHC(lmax=lmax)
        OBP_SH.append(temp_obp.value[0])

        year, month = i.split('-')[0], i.split('-')[1]
        temp_atmos = xr.open_dataset(f"../../data/ERA5/pressure level/sp-{year}{month}.nc")
        temp_asp = np.roll(temp_atmos['sp'].values, shift=shift_amount, axis=2)
        atmos_lat, atmos_lon = temp_atmos['latitude'].values, temp_atmos['longitude'].values - 180
        ib = IBcorrection(lat=atmos_lat, lon=atmos_lon)
        asp_ib_f = ib.correct(grids=temp_asp.flatten())
        asp_ib = asp_ib_f.reshape(len(temp_asp[:, 0, 0]), len(temp_asp[0, :, 0]), len(temp_asp[0, 0, :]))
        har = Harmonic(lat=atmos_lat, lon=atmos_lon, lmax=lmax, option=1)
        C, S = har.analysis(gqij=asp_ib)
        sh_asp = SHC(c=C, s=S).convert_type(from_type=Enums.PhysicalDimensions.Pressure,
                                            to_type=Enums.PhysicalDimensions.EWH)
        ASP_SH.append(sh_asp.value[0])

    ASP_SH = np.array(ASP_SH)
    ASP_SH = SHC(c=ASP_SH)
    ASP_SH.de_background()


    OBP_SH_N = np.array(OBP_SH)
    OBP_SH_N = SHC(c=OBP_SH_N)
    OBP_SH_N.de_background()

    '''Run SAL module'''

    lat, lon = MathTool.get_global_lat_lon_range(resolution=res)
    ATM_SAL = PseudoSpectralSLE(SH=ASP_SH.value, lmax=lmax)
    ATM_SAL.setLoveNumber(lmax=lmax, method=Enums.LLN_Data.Wang, frame=Enums.Frame.CM)
    ATM_SAL.setLatLon(lat=lat, lon=lon)
    ATM_SAL_results = ATM_SAL.SLE(mask=None, rotation=True, isLand=False)

    OCN_SAL = PseudoSpectralSLE(SH=OBP_SH_N.value, lmax=lmax)
    OCN_SAL.setLoveNumber(lmax=lmax, method=Enums.LLN_Data.Wang, frame=Enums.Frame.CM)
    OCN_SAL.setLatLon(lat=lat, lon=lon)
    OCN_SAL_results = OCN_SAL.SLE(mask=None, rotation=True, isLand=False)

    ATM_RSL = SHC(c=ATM_SAL_results['RSL_SH']).to_grid(res).value
    ATM_GHC = SHC(c=ATM_SAL_results['GHC']).to_grid(res).value
    ATM_VLM = SHC(c=ATM_SAL_results['VLM']).to_grid(res).value



    OCN_RSL = SHC(c=OCN_SAL_results['RSL_SH']).to_grid(res).value
    OCN_GHC = SHC(c=OCN_SAL_results['GHC']).to_grid(res).value
    OCN_VLM = SHC(c=OCN_SAL_results['VLM']).to_grid(res).value

    print(f"shape of atmosphere RSL, GHC, VLM is:{ATM_RSL.shape},{ATM_GHC.shape},{ATM_VLM.shape}")
    print(f"shape of ocean RSL, GHC, VLM is:{OCN_RSL.shape},{OCN_GHC.shape},{OCN_VLM.shape}")

    save_path = '../../result/SAL/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(f"{save_path}/ATM_RSL_grid.npz", data=ATM_RSL,lat=lat,lon=lon)
    np.savez(f"{save_path}/ATM_GHC_grid.npz", data=ATM_GHC,lat=lat,lon=lon)
    np.savez(f"{save_path}/ATM_VLM_grid.npz", data=ATM_VLM,lat=lat,lon=lon)

    np.savez(f"{save_path}/OCN_RSL_grid.npz", data=OCN_RSL,lat=lat,lon=lon)
    np.savez(f"{save_path}/OCN_GHC_grid.npz", data=OCN_GHC,lat=lat,lon=lon)
    np.savez(f"{save_path}/OCN_VLM_grid.npz", data=OCN_VLM,lat=lat,lon=lon)


def demo_GO():
    """This is an example for computing SAL effect using GRACE GSM, i.e., CSR RL06"""
    lmax, res = 60, 0.5
    begin_date, end_date = date(2009,1,1), date(2009,12,31)
    gsm_dir, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    low_deg_dir = FileTool.get_project_dir("data/L2_low_degrees/")
    rep_deg1, rep_c20, rep_c30 = True, True, True
    gia_filepath = FileTool.get_GIA_path(gia_type=Enums.GIAModel.Caron2018)
    filter_method, filter_params = Enums.SHCFilterType.DDK, (3,)

    '''load ocean mask'''
    basin_path_SH = FileTool.get_project_dir("data/basin_mask/SH/Ocean_maskSH.dat")
    shc_basin = load_SHC(basin_path_SH, key='', lmax=lmax)
    grid_basin = shc_basin.to_grid(grid_space=res)
    grid_basin.limiter(threshold=0.5)
    mask_ocean = grid_basin.value[0]
    '''load Antarctica mask'''
    Antarc_path_SH = FileTool.get_project_dir("data/basin_mask/SH/Antarctica_maskSH.dat")
    shc_Antarc = load_SHC(Antarc_path_SH,key='',lmax=lmax)
    grid_Antarc = shc_Antarc.to_grid(grid_space=res)
    grid_Antarc.limiter(threshold=0.5)
    mask_Antarc = grid_Antarc.value[0]
    '''load Greenland mask'''
    Grelan_path_SH = FileTool.get_project_dir("data/basin_mask/SH/Greenland_maskSH.dat")
    shc_Grelan = load_SHC(Grelan_path_SH,key='',lmax=lmax)
    grid_Grelan = shc_Grelan.to_grid(grid_space=res)
    grid_Grelan.limiter(threshold=0.5)
    mask_Grelan = grid_Grelan.value[0]

    '''load GSM and ancillary data'''
    shc, dates_begin, dates_end = load_SHC(gsm_dir, key=gsm_key, lmax=lmax, begin_date=begin_date, end_date=end_date,
                                           get_dates=True, )  # load GSM and dates
    dates_ave = TimeTool.get_average_dates(dates_begin,dates_end)
    low_degs = load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN11, )  # load c20
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN13,
                                  institute=Enums.L2InstituteType.CSR, release=Enums.L2Release.RL06))  # load degree-1
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN14))  # load c20 (update) and c30
    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax)  # load GIA trend
    shc_gia = shc_gia_trend.expand(dates_ave)  # project GIA trend into (monthly) signals along with GRACE times
    '''post-processing GSM'''
    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=rep_deg1, c20=rep_c20,
                         c30=rep_c30)  # replace SHC low_degrees
    shc.subtract(shc_gia)  # subtracting gia model
    shc.de_background()  # de-average if non-input
    shc.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,
                     to_type=Enums.PhysicalDimensions.EWH)
    shc.filter(method=filter_method, param=filter_params)  # average filter

    lat, lon = MathTool.get_global_lat_lon_range(resolution=res)
    '''Run SAL module'''
    SAL = PseudoSpectralSLE(SH=shc.value, lmax=lmax)
    SAL.setLoveNumber(lmax=lmax, method=Enums.LLN_Data.Wang, frame=Enums.Frame.CM)
    SAL.setLatLon(lat=lat, lon=lon)
    SAL_results = SAL.SLE(mask=mask_ocean,rotation=True,isLand=True)

    RSL = SHC(c=SAL_results['RSL_SH']).to_grid(res).value
    GHC = SHC(c=SAL_results['GHC']).to_grid(res).value
    VLM = SHC(c=SAL_results['VLM']).to_grid(res).value
    print(f"shape of RSL, GHC, VLM is:{RSL.shape},{GHC.shape},{VLM.shape}")

    save_path = '../../result/SAL/'
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    np.savez(f"{save_path}/GRACE_RSL_grid.npz", data=RSL)
    np.savez(f"{save_path}/GRACE_GHC_grid.npz", data=GHC)
    np.savez(f"{save_path}/GRACE_VLM_grid.npz", data=VLM)



if __name__ == '__main__':
    demo_GO()
    demo_NM()