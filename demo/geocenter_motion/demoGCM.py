import os
import numpy as np
import xarray as xr
from datetime import date
import SaGEA.auxiliary.preference.EnumClasses as Enums
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.aux_tool.TimeTool import TimeTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.ancillary.geotools.GeoMathKit import GeoMathKit
from pysrc.ancillary.load_file.DataClass import GRID
from pysrc.geocenter_motion.GeocenterMotion import GeocenterMotion
from pysrc.geocenter_motion.EarthOblateness import J2


def demo_GCM():
    """this demo shows an example of GRACE GCM estimation"""
    lmax, res, buffer_width = 60, 1, 0
    filter_method, filter_params = Enums.SHCFilterType.DDK, (3,)
    begin_date, end_date = date(2009,1,1),date(2009,12,31)
    begins, ends = begin_date.strftime("%Y-%m-%d").split('-'),end_date.strftime('%Y-%m-%d').split('-')
    '''Load GSM'''
    gsm_dir, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"),"GRCOF2"
    shc, dates_begin, dates_end = load_SHC(gsm_dir,key=gsm_key,lmax=lmax,begin_date=begin_date,end_date=end_date,
                                           get_dates=True)
    '''Load GIA'''
    dates_ave = TimeTool.get_average_dates(dates_begin,dates_end)
    gia_filepath = FileTool.get_GIA_path(gia_type=Enums.GIAModel.Caron2018)
    shc_gia_trend = load_SHC(gia_filepath,key='',lmax=lmax)
    shc_gia = shc_gia_trend.expand(dates_ave)
    '''Load GAD'''
    gad_dir, gad_key = FileTool.get_project_dir("data/L2_SH_products/GAD/GFZ/RL06/BC01/"),"GRCOF2"
    shc_gad = load_SHC(gad_dir,key=gad_key,lmax=lmax,begin_date=begin_date,end_date=end_date)
    '''Load GAC'''
    gac_dir, gac_key = FileTool.get_project_dir("data/L2_SH_products/GAC/GFZ/RL06/BC01/"), "GRCOF2"
    shc_gac = load_SHC(gac_dir,key=gac_key,lmax=lmax,begin_date=begin_date,end_date=end_date)
    '''Load ECCO_OBP'''
    dates = GeoMathKit().generate_month_range(start_str=f"{int(begins[0])}-{int(begins[1]):02d}",end_str=f"{int(ends[0])}-{int(ends[1]):02d}")
    OceanGrid = []
    for i in dates:
        temp = xr.open_dataset(f"../../data/ECCO/OBP/OCEAN_BOTTOM_PRESSURE_mon_mean_{i}_ECCO_V4r4b_latlon_0p50deg.nc")
        if i in ['2002-01', '2002-02', '2002-03', '2002-06', '2002-07', '2003-06',
                 '2011-01', '2011-06', '2012-05', '2012-10', '2013-03', '2013-08', '2013-09',
                 '2014-02', '2014-07', '2014-12', '2015-06', '2015-10', '2015-11',
                 '2016-04', '2016-09', '2016-10',
                 '2017-02', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12']:
            continue
        else:
            OceanGrid.append(temp['OBP'].values[0,int(res)::int(res*2),int(res)::int(res*2)])
    OceanGrid = np.array(OceanGrid)
    mask = np.where(np.isnan(OceanGrid[0]),np.nan,1)
    mask = np.nan_to_num(mask,nan=0)
    OceanGrid = np.nan_to_num(OceanGrid,nan=0)
    oceanlat = temp['latitude'].values[int(res)::int(res*2)]
    oceanlon = temp['longitude'].values[int(res)::int(res*2)]

    OceanSH = GRID(grid=OceanGrid,lat=oceanlat,lon=oceanlon).to_SHC(lmax=lmax)
    OceanSH.convert_type(from_type=Enums.PhysicalDimensions.EWH,to_type=Enums.PhysicalDimensions.Density)
    '''Processing'''
    shc.subtract(shc_gia)
    shc.de_background()
    shc_gac.de_background()
    shc_gad.de_background()
    OceanSH.de_background()

    shc.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,to_type=Enums.PhysicalDimensions.Density)

    shc.filter(method=filter_method,param=filter_params)

    GCM = GeocenterMotion(GRACE=shc.value,OceanSH=OceanSH.value,GAD=shc_gac.value,lmax=lmax)
    GCM.setResolution(resolution=res)
    GSM_like = GCM.GSM_Like(mask=mask,GRD=False,rotation=False,buffer=buffer_width)
    full_geocenter = GCM.Full_Geocenter(GAC=shc_gad.value,mask=mask,GRD=False)

    GCM_like_X,GCM_like_Y,GCM_like_Z = GSM_like['X']*1000,GSM_like['Y']*1000,GSM_like['Y']*1000
    GCM_full_X,GCM_full_Y,GCM_full_Z = full_geocenter['X']*1000,full_geocenter['Y']*1000,full_geocenter['Z']*1000

    print(f"X axis:\nOnly:{GCM_like_X}\nFull:{GCM_full_X}")
    print(f"Y axis:\nOnly:{GCM_like_Y}\nFull:{GCM_full_Y}")
    print(f"Z axis:\nOnly:{GCM_like_Z}\nFull:{GCM_full_Z}")

    save_path = '../../result/GCM/'
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    np.savez(f"{save_path}/GCM_GRACE_like_X.npz",data=GCM_full_X)
    np.savez(f"{save_path}/GCM_GRACE_like_Y.npz", data=GCM_full_Y)
    np.savez(f"{save_path}/GCM_GRACE_like_Z.npz", data=GCM_full_Z)
    np.savez(f"{save_path}/GCM_FULL_X.npz", data=GCM_full_X)
    np.savez(f"{save_path}/GCM_FULL_Y.npz", data=GCM_full_X)
    np.savez(f"{save_path}/GCM_FULL_Z.npz", data=GCM_full_X)


def demo_J2():
    """this demo shows an example of low-dergee terms, i.e., C10,S11,C11,C20,C30 estimation"""
    lmax, res, buffer_width = 60, 1, 0
    filter_method, filter_params = Enums.SHCFilterType.DDK, (3,)
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    begins, ends = begin_date.strftime("%Y-%m-%d").split('-'), end_date.strftime('%Y-%m-%d').split('-')
    '''Load GSM'''
    gsm_dir, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    shc, dates_begin, dates_end = load_SHC(gsm_dir, key=gsm_key, lmax=lmax, begin_date=begin_date, end_date=end_date,
                                           get_dates=True)
    '''Load GIA'''
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)
    gia_filepath = FileTool.get_GIA_path(gia_type=Enums.GIAModel.Caron2018)
    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax)
    shc_gia = shc_gia_trend.expand(dates_ave)
    '''Load GAD'''
    gad_dir, gad_key = FileTool.get_project_dir("data/L2_SH_products/GAD/GFZ/RL06/BC01/"), "GRCOF2"
    shc_gad = load_SHC(gad_dir, key=gad_key, lmax=lmax, begin_date=begin_date, end_date=end_date)
    '''Load GAC'''
    gac_dir, gac_key = FileTool.get_project_dir("data/L2_SH_products/GAC/GFZ/RL06/BC01/"), "GRCOF2"
    shc_gac = load_SHC(gac_dir, key=gac_key, lmax=lmax, begin_date=begin_date, end_date=end_date)
    '''Load ECCO_OBP'''
    dates = GeoMathKit().generate_month_range(start_str=f"{int(begins[0])}-{int(begins[1]):02d}",
                                              end_str=f"{int(ends[0])}-{int(ends[1]):02d}")
    OceanGrid = []
    for i in dates:
        temp = xr.open_dataset(f"../../data/ECCO/OBP/OCEAN_BOTTOM_PRESSURE_mon_mean_{i}_ECCO_V4r4b_latlon_0p50deg.nc")
        if i in ['2002-01', '2002-02', '2002-03', '2002-06', '2002-07', '2003-06',
                 '2011-01', '2011-06', '2012-05', '2012-10', '2013-03', '2013-08', '2013-09',
                 '2014-02', '2014-07', '2014-12', '2015-06', '2015-10', '2015-11',
                 '2016-04', '2016-09', '2016-10',
                 '2017-02', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12']:
            continue
        else:
            OceanGrid.append(temp['OBP'].values[0, int(res)::int(res * 2), int(res)::int(res * 2)])
    OceanGrid = np.array(OceanGrid)
    mask = np.where(np.isnan(OceanGrid[0]), np.nan, 1)
    mask = np.nan_to_num(mask, nan=0)
    OceanGrid = np.nan_to_num(OceanGrid, nan=0)
    oceanlat = temp['latitude'].values[int(res)::int(res * 2)]
    oceanlon = temp['longitude'].values[int(res)::int(res * 2)]

    OceanSH = GRID(grid=OceanGrid, lat=oceanlat, lon=oceanlon).to_SHC(lmax=lmax)
    OceanSH.convert_type(from_type=Enums.PhysicalDimensions.EWH, to_type=Enums.PhysicalDimensions.Density)
    '''Processing'''
    shc.subtract(shc_gia)
    shc.de_background()
    shc_gac.de_background()
    shc_gad.de_background()
    OceanSH.de_background()

    shc.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.Density)

    shc.filter(method=filter_method,param=filter_params)

    LD= J2(GRACE=shc.value, OceanSH=OceanSH.value, GAD=shc_gac.value, lmax=lmax)
    LD.setResolution(resolution=res)
    LD.setLoveNumber(method=Enums.LLN_Data.Wang,frame=Enums.Frame.CM)
    C20 = LD.Low_Degree_Term(mask=mask, GRD=True,rotation=False ,buffer=buffer_width)['Stokes']['C20']


    print(f"The C20 estimated by SAGEA-fluid is:\n{C20}")

    save_path = '../../result/GCM/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(f"{save_path}/C20_SH.npz", data=C20)

if __name__ == '__main__':
    demo_GCM()
    demo_J2()

