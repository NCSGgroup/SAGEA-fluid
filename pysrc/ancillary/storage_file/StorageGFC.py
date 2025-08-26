import os
import numpy as np
from datetime import date
from pysrc.ancillary.load_file.DataClass import SHC


class CnmSnm:
    def __init__(self,begin_date:date,end_date:date,Nmax:int):
        self.SH = {}
        self.maxDegree = Nmax
        self.begin_date = begin_date
        self.end_date = end_date
    def add(self,SH:np.ndarray,keys:list):
        assert np.ndim(SH) == 3
        for i in np.arange(len(keys)):
            self.SH[keys[i]]=SH[i]

        return self
class StorageGFC:
    def __init__(self):
        self.__fileDir = None
        self.__SH = None
        self.__fileFullPath = None
        self.orderFirst = False
        self.input_source = "*GFZ* Level-2 Release-06"
        self.comment1 = "Rotational feedbacks *INCLUDED*"
        self.comment2 = "Center-of-mass (CM)* reference frame"
        self.comment3 = "Solutions are relative to 2003-01 - 2013-11 means "
        self.Prefix = None
        self.Suffix = None
        pass

    def setRootDir(self,fileDir):
        self.__fileDir = fileDir
        assert os.path.exists(fileDir),f"The path {fileDir} is not exist"
        return self
    def setSLEComments(self,rotation=False,meanfieldtime='2003-01 2014-12'):
        if rotation:
            self.comment1 = "Rotational feedbacks *INCLUDED*"
        else:
            self.comment1 = "Rotational feedbacks *EXCLUDED*"
        self.comment3 = f"Solutions are relative to {meanfieldtime} means "
        return self
    def setPreSufFix(self,prefix,suffix):
        self.Prefix = prefix
        self.Suffix = suffix
        return self
    def setSH(self,SH:CnmSnm):
        self.__SH = SH
        # res = CS.date.split('-')
        begin_year = SH.begin_date.year
        day_of_begin_year = SH.begin_date.timetuple().tm_yday
        formatted_begin_date = f"{begin_year}{day_of_begin_year:03d}"

        end_year = SH.end_date.year
        day_of_end_year = SH.end_date.timetuple().tm_yday
        formatted_end_date = f"{end_year}{day_of_end_year:03d}"

        subdir = str(begin_year)

        subdir = os.path.join(self.__fileDir,subdir)

        if not os.path.exists(subdir):
            os.makedirs(subdir)

        self.__fileFullPath = subdir+os.sep+f'{self.Prefix}{formatted_begin_date}-{formatted_end_date}{self.Suffix}.gfc'
        return self
    def SLEStyle(self):
        with open(self.__fileFullPath,'w') as file:
            file.write('%-31s%3s \n' % ('header: ', ' '))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'product_id ', ': ', 'SLF'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'long_name ', ': ', 'Sea-level fingerprints'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'model input ', ': ', f'{self.input_source}'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'comment #1 ', ': ', f'{self.comment1}'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'comment #2 ', ': ', f'{self.comment2}'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'comment #3 ', ': ', f'{self.comment3}'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'units ', ': ', 'meters of water height equivalent'))
            file.write('=========================================================================================================================\n')
            file.write('%-31s%3s \n' % ('variables: ', ' '))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column1 ', ': ', 'spherical harmonic degree'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column2 ', ': ', 'spherical harmonic order'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column3 ', ': ', 'cosine coefficients for land load function '))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column4 ', ': ', 'sine coefficients for land load function '))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column5 ', ': ', 'cosine coefficients for relative sea-level '))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column6 ', ': ', 'sine coefficients for relative sea-level '))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column7 ', ': ', 'cosine coefficients for geoid height change'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column8 ', ': ', 'sine coefficients for geoid height change'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column9 ', ': ', 'cosine coefficients for bedrock motion'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column10 ', ': ', 'sine coefficients for bedrock motion'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column11 ', ': ', 'cosine coefficients for ocean mask'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'column12 ', ': ', 'sine coefficients for ocean mask'))
            file.write('=========================================================================================================================\n')
            file.write('%-31s%3s \n' % ('reference: ', ' '))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'authors ', ': ', 'Zhang Weihang, Yang Fan, Liu Shuhao;'))
            file.write('%3s%-20s%3s%-31s \n' % (' ', 'contact ', ': ', 'zwh_cge@hust.edu.cn; yfan_cge@hust.edu.cn'))
            file.write('=========================================================================================================================\n')
            file.write('%-5s %-5s  %-15s  %-15s  %-15s  %-15s %-15s  %-15s %-15s %-15s %-15s %-15s\n'
                       % ('L', 'M', 'Input_C', 'Input_S', 'rsl_C', 'rsl_S', 'ghc_C', 'ghc_S','vlm_C', 'vlm_S','OceanMask_C','OceanMask_S'))
            file.write('=========================================================================================================================\n')

            keys = list(self.__SH.SH.keys())
            # keys.sort()
            # print(keys)

            # for key in keys:
            #     print(f"key is {key}")
            SH = self.__SH.SH
            Nmax = self.__SH.maxDegree
            self.SLE_mainContent(SH_Assemble=SH,keys=keys,Nmax=Nmax,file=file)
        print(f"Already save file in {self.__fileFullPath}")



    def SLE_mainContent(self, SH_Assemble,keys,Nmax,file):

        LCnm, LSnm = SHC(c=SH_Assemble[keys[0]]).get_cs2d()
        RCnm, RSnm = SHC(c=SH_Assemble[keys[1]]).get_cs2d()
        GCnm, GSnm = SHC(c=SH_Assemble[keys[2]]).get_cs2d()
        VCnm, VSnm = SHC(c=SH_Assemble[keys[3]]).get_cs2d()
        OCnm, OSnm = SHC(c=SH_Assemble[keys[4]]).get_cs2d()
        if self.orderFirst:
            for i in range(Nmax + 1):
                for j in range(i + 1):
                    file.write('%5i %5i %+15.10E %+15.10E %+15.10E %+15.10E %+15.10E %+15.10E %+15.10E %+15.10E %+15.10E %+15.10E\n'
                               % (i, j, LCnm[0, i, j], LSnm[0, i, j],RCnm[0,i,j],RSnm[0,i,j],GCnm[0,i,j],GSnm[0,i,j],
                                  VCnm[0,i,j],VSnm[0,i,j],OCnm[0,i,j],OSnm[0,i,j]))
        else:
            for j in range(Nmax + 1):
                for i in range(j, Nmax + 1):
                    file.write('%5i %5i  %+15.10E  %+15.10E %+15.10E  %+15.10E %+15.10E %+15.10E %+15.10E  %+15.10E %+15.10E %+15.10E\n'
                               % (i, j, LCnm[0, i, j], LSnm[0, i, j],RCnm[0,i,j],RSnm[0,i,j],GCnm[0,i,j],GSnm[0,i,j],
                                  VCnm[0,i,j],VSnm[0,i,j],OCnm[0,i,j],OSnm[0,i,j]))


def demo_SLE():
    import netCDF4 as nc
    from SaGEA.auxiliary.aux_tool.FileTool import FileTool
    from pysrc.ancillary.load_file.LoadCS import LoadCS
    from SaGEA.auxiliary.aux_tool.MathTool import MathTool
    from pysrc.sealevel_equation.SpectralSeaLevel import PseudoSpectralSLE
    from pysrc.ancillary.geotools.LLN import LLN_Data,Frame
    res = 0.5
    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")['ocean_mask'][:]
    filepath = FileTool.get_project_dir('data/ref_sealevel/SLFsh_coefficients/GFZOP/CM/WOUTrotation/')
    begin_date, end_date = date(2003, 1, 1), date(2010, 2, 1)
    Load_SH, begins, ends = LoadCS().get_CS(filepath, begin_date=begin_date, end_date=end_date,
                                            lmcs_in_queue=np.array([0, 1, 2, 4]), get_dates=True)

    lat, lon = MathTool.get_global_lat_lon_range(res)
    A = PseudoSpectralSLE(SH=Load_SH.value, lmax=60).setLatLon(lat=lat, lon=lon)
    A.setLoveNumber(lmax=60,method=LLN_Data.PREM,frame=Frame.CM)

    RSLwout = A.SLE(rotation=False, mask=ocean_mask)
    Input_SH = RSLwout['Input']
    Quasi_SH = RSLwout['Quasi_RSL_SH']
    GHC_SH = RSLwout['GHC']
    VLM_SH = RSLwout['VLM']
    OCE_SH = RSLwout['mask']
    print(f"SLE Information: {Input_SH.shape},{Quasi_SH.shape},{GHC_SH.shape},{VLM_SH.shape}")
    SH = []
    SH.append(Input_SH)
    SH.append(Quasi_SH)
    SH.append(GHC_SH)
    SH.append(VLM_SH)
    SH.append(OCE_SH)
    SH = np.array(SH)

    print(SH.shape)

    fm = StorageGFC().setRootDir(fileDir='../../result/SLFsh_coefficients/GFZOP/CM/WOUTrotation/')
    fm.setPreSufFix(prefix="SLF-2_", suffix="_GRAC_GFZOP_BA01_0600")
    for dayindex in np.arange(len(begins)):
        sh_fm = CnmSnm(begin_date=begins[dayindex], end_date=ends[dayindex], Nmax=60)
        sh_fm.add(SH=SH, keys=['Input', 'RSL', 'GHC', 'VLM','Ocean_mask'])
        fm.setSH(SH=sh_fm).SLEStyle()

if __name__ == '__main__':
    demo_SLE()
