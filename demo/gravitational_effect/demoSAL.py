from datetime import date
import SaGEA.auxiliary.preference.EnumClasses as Enums
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.aux_tool.TimeTool import TimeTool
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from SaGEA.auxiliary.load_file.LoadL2LowDeg import load_low_degs
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC

from pysrc.sealevel_equation.SeaLevelEquation import PseudoSpectralSLE

def demo1():
    lmax, res = 60, 0.5
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    gaa_dir, gaa_key = FileTool.get_project_dir("data/L2_SH_products/GAA/GFZ/RL06/BC01/"), "GRCOF2"
    shc_gaa = load_SHC(gaa_dir, key=gaa_key, lmax=lmax, begin_date=begin_date, end_date=end_date)  # load GAA
    shc_gaa.de_background()
    shc_gaa.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,to_type=Enums.PhysicalDimensions.EWH)

    lat,lon = MathTool.get_global_lat_lon_range(resolution=res)

    SAL = PseudoSpectralSLE(SH=shc_gaa.value,lmax=lmax)
    SAL.setLoveNumber(lmax=lmax,method=Enums.LLN_Data.Wang,frame=Enums.Frame.CM)
    SAL.setLatLon(lat=lat,lon=lon)

    ATM_SAL = SAL.SLE(rotation=True,mask=None,isOnlyTWS=False)
    print(ATM_SAL['RSL_SH'].shape)


    print(shc_gaa.value.shape)
def demo3():
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

    Antarc_path_SH = FileTool.get_project_dir("data/basin_mask/SH/Antarctica_maskSH.dat")
    shc_Antarc = load_SHC(Antarc_path_SH,key='',lmax=lmax)
    grid_Antarc = shc_Antarc.to_grid(grid_space=res)
    grid_Antarc.limiter(threshold=0.5)
    mask_Antarc = grid_Antarc.value[0]

    Grelan_path_SH = FileTool.get_project_dir("data/basin_mask/SH/Greenland_maskSH.dat")
    shc_Grelan = load_SHC(Grelan_path_SH,key='',lmax=lmax)
    grid_Grelan = shc_Grelan.to_grid(grid_space=res)
    grid_Grelan.limiter(threshold=0.5)
    mask_Grelan = grid_Grelan.value[0]

    '''load GSM and aux_fuction data'''
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

if __name__ == '__main__':
    demo1()