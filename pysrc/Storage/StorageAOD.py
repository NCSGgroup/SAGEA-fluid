"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/7/1 下午6:55
@Description:
"""
import os
import numpy as np
import time
from datetime import datetime, timedelta
from pysrc.Auxiliary.GeoMathKit import GeoMathKit
from pysrc.Auxiliary.MathTool import MathTool
import time as ti
import xarray as xr
from tqdm import tqdm
from pysrc.LoadFile.LoadAOD import AOD_GFZ,AODtype
from pysrc.AliasingModel.Specify.Harmonic import Harmonic
from pysrc.PostProcessing.old.LoveNumber import LoveNumber,LoveNumberType
from pysrc.PostProcessing.old.Setting import HarAnalysisType,SynthesisType



class CnmSnm:

    def __init__(self, date: str, Nmax: int):
        self.Cnm = {}
        self.Snm = {}
        self.maxDegree = Nmax
        self.date = date
        self._other()
        pass

    def add(self, Cnm, Snm, epoch: str, date: str, attribute: str):
        assert date == self.date

        self.Cnm[epoch+'/'+attribute] = Cnm

        self.Snm[epoch+'/'+attribute] = Snm

        pass

    def _other(self):
        self.producer = 'HUST'
        self.product_type = 'Atmosphere Dealiasing'
        # self.product = 'HUST'
        self.version = '01'
        self.author='Yang F'+','+'Zhang W.H'
        self.start_year =''
        self.end_year =''

class FormatWrite:

    def __init__(self):
        self.__fileDir = None
        self.__CS = None
        self.__fileFullPath = None
        self.orderFirst = True
        self.__fileErrPath = None
        pass

    def setRootDir(self, fileDir):
        self.__fileDir = fileDir
        assert os.path.exists(fileDir)
        return self
    def day_of_year_to_date(self,ordinal):
        year = int(str(ordinal)[:4])
        day_of_year = int(str(ordinal)[4:])
        date = datetime(year,1,1) + timedelta(days=day_of_year-1)
        return date.strftime('%Y-%m-%d')

    def setCS(self, CS: CnmSnm):
        self.__CS = CS
        res = CS.date.split('-')
        subdir_year = res[0]

        try:
            subdir = res[0] + '-' + res[1]

        except:
            return self

        subdir = os.path.join(self.__fileDir, subdir_year)

        if not os.path.exists(subdir):
            os.makedirs(subdir)

        self.__fileFullPath = subdir + os.sep + 'AOD1B_'+CS.date + '.asc'
        self.__fileGraceL2B = subdir + os.sep



        return self

    def AODstyle(self,date='2020-01-01'):
        with open(self.__fileFullPath, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY  ', ': ', 'HUST'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER INSTITUTION ', ': ', 'HUST-PGMF'))
            file.write('%-31s%3s%-31s \n' % ('FILE TYPE ipAOD1BF ', ': ', '999'))
            file.write('%-31s%3s%-31s \n' % ('FILE FORMAT 0=BINARY 1=ASCII', ': ', '1'))
            file.write('%-31s%3s%-31s \n' % ('NUMBER OF HEADER RECORDS', ': ', '29'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION  ', ': ', 'atm_ocean_dealise.06'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE LINK TIME   ', ': ', 'Not Applicable'))
            file.write('%-31s%3s%-31s \n' % ('REFERENCE DOCUMENTATION  ', ': ', 'GRACE AOD1B PDD'))
            file.write('%-31s%3s%-31s \n' % ('SATELLITE NAME', ': ', 'GRACE X'))
            file.write('%-31s%3s%-31s \n' % ('SENSOR NAME', ': ', 'Not Applicable'))
            file.write('%-31s%3s%-31s \n' % ('TIME EPOCH (GPS TIME)  ', ': ', '{}'.format(date)))
            file.write('%-31s%3s%-31s \n' % ('TIME FIRST OBS(SEC PAST EPOCH) ', ': ', '{} 00:00:00'.format(date)))
            file.write('%-31s%3s%-31s \n' % ('TIME LAST OBS(SEC PAST EPOCH)', ': ', '{} 21:00:00'.format(date)))
            file.write('%-31s%3s%-31s \n' % ('NUMBER OF DATA RECORDS', ': ', '527072'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCT START CREATE TIME(UTC)', ': ', datetime.now()))
            file.write('%-31s%3s%-31s \n' % ('PRODUCT END CREATE TIME(UTC)', ': ', datetime.now()))
            file.write('%-31s%3s%-31s \n' % ('FILESIZE (BYTES)', ': ', '21086441'))
            file.write('%-31s%3s%-31s \n' % ('FILENAME      ', ': ', 'AOD1B_{}_X_06.asc'.format(date)))
            file.write('%-31s%3s%-31s \n' % ('PROCESS LEVEL (1A OR 1B) ', ': ', '1B'))
            file.write('%-31s%3s%-31s \n' % ('PRESSURE TYPE (SP OR VI) ', ': ', 'VI'))
            file.write('%-31s%3s%-31s \n' % ('MAXIMUM DEGREE ', ': ', '{}'.format(self.__CS.maxDegree)))
            file.write('%-31s%3s%-31s \n' % ('COEFFICIENTS ERRORS (YES/NO)', ': ', 'NO'))
            file.write('%-31s%3s%-31s \n' % ('COEFF. NORMALIZED (YES/NO)  ', ': ', 'YES'))
            file.write('%-31s%3s%-31s \n' % ('CONSTANT GM [M^3/S^2]     ', ': ', '0.39860044180000E+15'))
            file.write('%-31s%3s%-31s \n' % ('CONSTANT A [M]     ', ': ', '0.63781366000000E+07'))
            file.write('%-31s%3s%-31s \n' % ('CONSTANT FLAT [-] ', ': ', '0.29825642000000E+03'))
            file.write('%-31s%3s%-31s \n' % ('CONSTANT OMEGA [RAD/S]', ': ', '0.72921150000000E-04'))
            file.write('%-31s%3s%-31s \n' % ('NUMBER OF DATA SETS', ': ', '32'))
            file.write('%-31s%3s%-31s \n' % ('DATA FORMAT (N,M,C,S)  ', ': ', '(2(I3,x),E15.9,X,E15.9)'))
            file.write('END OF HEADER \n')

            keys = list(self.__CS.Cnm.keys())
            keys.sort()

            for key in keys:
                Cnm, Snm = self.__CS.Cnm[key], self.__CS.Snm[key]
                Nmax = self.__CS.maxDegree
                file.write('DATA SET %2i:   %s COEFFICIENTS FOR %s %s OF TYPE %s \n'
                           % (keys.index(key), int((Nmax + 2) * (Nmax + 1) / 2), self.__CS.date, key.split('/')[0],
                              key.split('/')[1].lower()))
                self._mainContent(Cnm, Snm, Nmax, file)

        pass

    def GRACE_L2B(self,kind,start,end):
        type_name = ''
        if kind == 'ATM' or kind == 'atm':
            type_name = 'GAA'

        elif kind == 'OCN' or kind == 'ocn':
            type_name = 'GAB'

        elif kind == 'GLO' or kind == 'glo':
            type_name = 'GAC'

        elif kind == 'OBA' or kind == 'oba':
            type_name = 'GAD'
        print(f'{kind},{type_name}')
        save_file = self.__fileGraceL2B+f'{type_name}-2_{start}-{end}_GRFO_HUST_IAP_BC01_0600.gfc'
        self.orderFirst = False
        print(save_file)
        start = self.day_of_year_to_date(ordinal=start)
        end = self.day_of_year_to_date(ordinal=end)
        starts = start.split('-')
        ends = end.split('-')
        current_time = time.strftime("%a %b %d %H:%M:%S %Y",time.localtime())
        with open(save_file, 'w') as file:
            file.write('**************************************************************\n')
            file.write(f'model converted into ICGEM-format at: {current_time}\n')
            file.write('**************************************************************\n')
            file.write('\n')
            file.write('**** some information from original YAML header ****\n')
            file.write('summary             : Spherical harmonic coefficients '
                       'that represent the sum of the ATM (or GAA) and OCN (or GAB) '
                       'coefficients during the specified timespan. These coefficients '
                       'represent anomalous contributions of the non-tidal dynamic ocean '
                       'to ocean bottom pressure, the non-tidal atmospheric surface pressure '
                       'over the continents, the static contribution of atmospheric pressure '
                       'to ocean bottom pressure, and the upper-air density anomalies above '
                       'both the continents and the oceans. The anomalous signals are relative '
                       'to the mean field from 2003-2014.\n')
            file.write('history             : GRACE Level-2 Data created at HUST and IAP\n')
            file.write('acknowledgement     : GRACE is a joint mission of NASA (USA) and DLR (Germany).\n')
            file.write('license             : None\n')
            file.write('references          : None\n')
            file.write(f'time_coverage_start : {start}\n')
            file.write(f'time_coverage_end   : {end}\n')
            file.write('unused_days         : Not listed\n')
            file.write('**********  end of original YAML header  ***********\n')
            file.write('\n')
            file.write(f'time_period_of_data:    {starts[0]+starts[1]+start[2]}-{ends[0]+ends[1]+ends[2]}   \n')
            file.write('generating_institute   HUST and IAP\n')
            file.write('\n')
            file.write('begin_of_head ===================================================\n')
            file.write('product_type           gravity_field\n')
            file.write(f'modelname              {type_name}-2_{starts[0]+starts[1]}_GRFO_HUST_IAP_BC01_0600\n')
            file.write(f'radius                 6.3781366000e+06\n')
            file.write('earth_gravity_constant 3.9860044180e+14\n')
            file.write('max_degree             180\n')
            file.write('norm                   fully_normalized\n')
            file.write('errors                 formal\n')
            file.write('%5s %5s %5s  %5s  %5s\n' % ('key','L','M','C','S'))

            file.write('end_of_head ================================================================== \n')

            keys = list(self.__CS.Cnm.keys())
            keys.sort()

            for key in keys:
                Cnm, Snm = self.__CS.Cnm[key], self.__CS.Snm[key]
                Nmax = self.__CS.maxDegree
                self._ErrmainContent(Cnm, Snm, Nmax, file)

    def CRALICOMstyle(self):
        with open(self.__fileFullPath, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY  ', ': ', 'IAP'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCT AUTHOR ', ': ', 'Yang F., Bai J., Liu H., Zhang W'))
            file.write('%-31s%3s%-31s \n' % ('CONTACT    ', ': ', 'lhl@lasg.iap.ac.cn'))
            file.write('END OF HEADER \n')

            keys = list(self.__CS.Cnm.keys())

            # keys.sort()


            for key in keys:
                Cnm, Snm = self.__CS.Cnm[key], self.__CS.Snm[key]
                Nmax = self.__CS.maxDegree
                file.write('DATA SET %02i:   %s COEFFICIENTS FOR %s %s OF TYPE %s \n'
                           % (keys.index(key)+1, int((Nmax + 2) * (Nmax + 1) / 2), self.__CS.date, key.split('/')[0],
                              key.split('/')[1].lower()))
                # print(f"{keys.index(key)+1} {int((Nmax + 2) * (Nmax + 1) / 2)} {self.__CS.date} {key.split('/')[0]} {key.split('/')[1].lower()}")
                self._mainContent(Cnm, Snm, Nmax, file)

        pass

    def TideStyle(self, tide:str, range='2007'):
        fileFullPath = self.__fileDir + os.sep + 'ATM_'+tide+'.asc'

        with open(fileFullPath, 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('PRODUCER AGENCY ', ': ', 'HUST'))
            file.write('%-31s%3s%-31s \n' % ('PRODUCER INSTITUTION', ': ', 'HUST-PGMF'))
            file.write('%-31s%3s%-31s \n' % ('FILE TYPE ipAOD1BF', ': ', '999'))
            file.write('%-31s%3s%-31s \n' % ('FILE FORMAT 0=BINARY 1=ASCII ', ': ', '1'))
            file.write('%-31s%3s%-31s \n' % ('NUMBER OF HEADER RECORDS  ', ': ', '26'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE VERSION     ', ': ', 'atm ocean dealise.06'))
            file.write('%-31s%3s%-31s \n' % ('SOFTWARE LINK TIME   ', ': ', 'Not Applicable'))
            file.write('%-31s%3s%-31s \n' % ('REFERENCE DOCUMENTATION   ', ': ', 'GRACE AOD1B PDD, version 06'))
            file.write('%-31s%3s%-31s \n' % ('SATELLITE NAME   ', ': ', 'GRACE X'))
            file.write('%-31s%3s%-31s \n' % ('SENSOR NAME     ', ': ', 'Not Applicable'))
            file.write('%-31s%3s%-31s \n' % ('PARTIAL TIDE  ', ': ', tide))
            file.write('%-31s%3s%-31s \n' % ('TIME FIRST OBS (YEAR START) ', ': ', '2007'))
            file.write('%-31s%3s%-31s \n' % ('TIME LAST OBS (YEAR END)  ', ': ', range))
            file.write('%-31s%3s%-31s \n' % ('NUMBER OF DATA RECORDS   ', ': ','32942'))
            file.write('%-31s%3s%-31s \n' % ('FILENAME     ', ': ', 'AOD1B_ATM_{}_06.asc'.format(tide)))
            file.write('%-31s%3s%-31s \n' % ('PROCESS LEVEL (1A OR 1B)   ', ': ', '1B'))
            file.write('%-31s%3s%-31s \n' % ('PRESSURE TYPE (ATM OR OCN)  ', ': ', 'ATM'))
            file.write('%-31s%3s%-31s \n' % ('MAXIMUM DEGREE     ', ': ', '180'))
            file.write('%-31s%3s%-31s \n' % ('COEFFICIENTS ERRORS (YES/NO)  ', ': ', 'NO'))
            file.write('%-31s%3s%-31s \n' % ('COEFF. NORMALIZED (YES/NO)  ', ': ', 'YES'))
            file.write('%-31s%3s%-31s \n' % ('CONSTANT GM [M^3/S^2]   ', ': ', '0.39860044180000E+15'))
            file.write('%-31s%3s%-31s \n' % ('CONSTANT A [M]    ', ': ', '0.63781366000000E+07'))
            file.write('%-31s%3s%-31s \n' % ('CONSTANT FLAT [-]     ', ': ', '0.29825642000000E+03'))
            file.write('%-31s%3s%-31s \n' % ('CONSTANT OMEGA [RAD/S]  ', ': ', '0.72921150000000E-04'))
            file.write('%-31s%3s%-31s \n' % ('NUMBER OF DATA SETS  ', ': ', '2'))
            file.write('%-31s%3s%-31s \n' % ('DATA FORMAT (N,M,C,S)   ', ': ', '(2(I3,x),E15.9,X,E15.9)'))
            file.write('END OF HEADER \n')

            keys = list(self.__CS.Cnm.keys())
            keys.sort()

            for key in keys:
                Cnm, Snm = self.__CS.Cnm[key], self.__CS.Snm[key]
                Nmax = self.__CS.maxDegree
                file.write('DATA SET %2i:   %s COEFFICIENTS OF TYPE %s \n'
                           % (keys.index(key), int((Nmax + 2) * (Nmax + 1) / 2), key.split('/')[1].lower()))
                self._mainContent(Cnm, Snm, Nmax, file)

        pass

    def HUSTerrStyle(self, time='00:00:00'):
        print(self.__fileErrPath+'{}.asc'.format(time.split(':')[0]))
        with open(self.__fileErrPath+'{}.asc'.format(time.split(':')[0]), 'w') as file:
            file.write('%-31s%3s%-31s \n' % ('product type   ', ': ', 'anomalous gravity potential'))
            file.write('%-31s%3s%-31s \n' % ('modelname ', ': ', 'Improved Mass Transport Model'))
            file.write('%-31s%3s%-31s \n' % ('model content   ', ': ', 'HUST_Aerr'))
            file.write('%-31s%3s%-31s \n' % ('version   ', ': ', '1.0'))
            file.write('%-31s%3s%-31s \n' % ('earth_gravity_constant ', ': ', float('0.39860050000000E+15')))
            file.write('%-31s%3s%-31s \n' % ('radius    ', ': ', '6378137.0000'))
            file.write('%-31s%3s%-31s \n' % ('max_degree  ', ': ', int('180')))
            file.write('%-31s%3s%-31s \n' % ('error   ', ': ', 'no'))
            file.write('%-31s%3s%-31s \n' % ('norm  ', ': ', 'fully_normalized'))
            file.write('%-31s%3s%-31s \n' % ('tide_system ', ': ', 'does-not-apply'))
            file.write('end_of_head \n')

            keys = list(self.__CS.Cnm.keys())
            keys.sort()

            for key in keys:
                Cnm, Snm = self.__CS.Cnm[key], self.__CS.Snm[key]
                Nmax = self.__CS.maxDegree
                self._ErrmainContent(Cnm, Snm, Nmax, file)

        pass

    def _mainContent(self, Cnm, Snm, Nmax, file):

        if np.ndim(Cnm) == 1:
            Cnm = GeoMathKit.CS_1dTo2d(Cnm)
            Snm = GeoMathKit.CS_1dTo2d(Snm)

        if self.orderFirst:
            for i in range(Nmax + 1):
                for j in range(i + 1):
                    file.write('%5i %5i  %+15.10E  %+15.10E\n' % (i, j, Cnm[i, j], Snm[i, j]))
        else:
            for j in range(Nmax + 1):
                for i in range(j, Nmax + 1):
                    file.write('%5i %5i  %15.10g  %15.10g\n' % (i, j, Cnm[i, j], Snm[i, j]))

        pass

    def _ErrmainContent(self,Cnm, Snm, Nmax, file):
        pre = 'gfc'
        if np.ndim(Cnm) == 1:
            Cnm = GeoMathKit.CS_1dTo2d(Cnm)
            Snm = GeoMathKit.CS_1dTo2d(Snm)

        if self.orderFirst:
            for i in range(Nmax + 1):
                for j in range(i + 1):
                    # print(i,j,Cnm[i,j],Snm[i,j])
                    file.write('%5s %5i %5i  %+15.10E  %+15.10E\n' % (pre, i, j, Cnm[i, j], Snm[i, j]))
        else:
            for j in range(Nmax + 1):
                for i in range(j, Nmax + 1):
                    file.write('%5s %5i %5i  %15.10g  %15.10g\n' % (pre, i, j, Cnm[i, j], Snm[i, j]))

        pass


class SyntheticAOD:
    def __init__(self):
        self.mask_file='I:\CRALICOM\data\Auxiliary/mask_720X361.nc'
        self.Nmax = 180
        self.SavePath = "I:\CRALICOM/result\Licom_AOD/"
        self.daylist = GeoMathKit.dayListByDay('2024-03-01','2024-04-30')
        self.TimeEpoch = ['00:00:00','06:00:00','12:00:00','18:00:00']
        self.lat,self.lon = MathTool.get_global_lat_lon_range_V2(resolution=0.5)
        self.LN = LoveNumber('../../data/auxiliary/')
        self.HM = Harmonic(self.LN).setLoveNumMethod(LoveNumberType.Wang)
        self.ATM_path = 'I:\CRALICOM/result\ATM/'
        self.OCN_path = 'I:\CRALICOM/result\OCN_pso/'
        self.upperair_path = 'I:\CRALICOM/result/upper_demean/'

    def setMask(self,file):
        self.mask_file = file
        return self

    def Synthetic(self):
        Nmax = self.Nmax
        mask = xr.open_dataset(self.mask_file)
        ocean_mask = mask['mask'].values
        ocean_mask = np.nan_to_num(ocean_mask, nan=1)
        land_mask = 1 - ocean_mask
        PnmMat = GeoMathKit.getPnmMatrix(self.lat, Nmax, 2)
        fm = FormatWrite().setRootDir(self.SavePath)
        begin_time = ti.time()
        for date in tqdm(self.daylist):
            date = date.strftime('%Y-%m-%d')
            cs_file = CnmSnm(date=date, Nmax=Nmax)
            for epoch in self.TimeEpoch:
                CS_ATM = np.array(
                    AOD_GFZ().load(self.ATM_path).setType(AODtype.ATM).setTime(date, epoch).getCS_old(Nmax))
                cs_file.add(Cnm=CS_ATM[0], Snm=CS_ATM[1], epoch=epoch, date=date,
                            attribute=AODtype.ATM.name)
                CS_OCN = np.array(
                    AOD_GFZ().load(self.OCN_path).setType(AODtype.OCN).setTime(date, epoch).getCS_old(Nmax))
                cs_file.add(Cnm=CS_OCN[0], Snm=CS_OCN[1], epoch=epoch, date=date,
                            attribute=AODtype.OCN.name)
                CS_GLO = CS_ATM + CS_OCN
                cs_file.add(Cnm=CS_GLO[0], Snm=CS_GLO[1], epoch=epoch, date=date,
                            attribute=AODtype.GLO.name)
                ###########Method 1:
                # upperair = np.array(AOD_GFZ().load(self.upperair_path).setType(AODtype.ATM).setTime(date, epoch).getCS_old(Nmax))
                # CS_OBA = CS_GLO - upperair
                # Space = self.HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(CS_OBA[0]), Sqlm=GeoMathKit.CS_1dTo2d(CS_OBA[1]),
                #                           PnmMat=PnmMat, lat=self.lat, lon=self.lon, Nmax=Nmax,
                #                           kind=SynthesisType.synthesis)
                # Space[land_mask == 0] = 0
                # C_mask, S_mask = self.HM.analysis(Nmax=Nmax, Gqij=Space, lat=self.lat, lon=self.lon,
                #                                   PnmMat=PnmMat, kind=HarAnalysisType.analysis)
                # cs_file.add(Cnm=C_mask, Snm=S_mask, epoch=epoch, date=date,
                #             attribute=AODtype.OBA.name)
                ############Method 2:
                upperair = np.array(AOD_GFZ().load(self.upperair_path).setType(AODtype.ATM).setTime(date, epoch).getCS_old(Nmax))
                Space_upper = self.HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(upperair[0]), Sqlm=GeoMathKit.CS_1dTo2d(upperair[1]),
                                                PnmMat=PnmMat, lat=self.lat, lon=self.lon, Nmax=Nmax,kind=SynthesisType.Pressure)
                Space_GLO = self.HM.synthesis(Cqlm=GeoMathKit.CS_1dTo2d(CS_GLO[0]), Sqlm=GeoMathKit.CS_1dTo2d(CS_GLO[1]),
                                              PnmMat=PnmMat, lat=self.lat, lon=self.lon, Nmax=Nmax,kind=SynthesisType.Pressure)
                Space = Space_GLO-Space_upper
                Space[land_mask == 0] = 0
                C_mask, S_mask = self.HM.analysis(Nmax=Nmax, Gqij=Space, lat=self.lat, lon=self.lon,
                                                  PnmMat=PnmMat, kind=HarAnalysisType.Pressure)
                cs_file.add(Cnm=C_mask, Snm=S_mask, epoch=epoch, date=date,
                            attribute=AODtype.OBA.name)
            fm.setCS(cs_file).CRALICOMstyle()
        print(f'Cost time: {ti.time() - begin_time} s')



class SyntheticGAX:
    def __init__(self):
        self.TimeEpoch = ['00:00:00','06:00:00','12:00:00','18:00:00']
        self.LoadPath = 'I:\CRALICOM/result\Licom_AOD/'
        self.SavePath = 'I:\CRALICOM/result/'
        self.Month = 12


        self.begin_grace = \
            ['2002095','2002122','2002213','2002244','2002274','2002305','2002335',
             '2003001','2003032','2003060','2003091','2003121','2003182','2003213','2003244','2003274','2003305','2003335',
             '2004001','2004035','2004061','2004092','2004122','2004153','2004183','2004214','2004245','2004275','2004306','2004336',
             '2005001','2005032','2005060','2005091','2005121','2005152','2005182','2005213','2005244','2005274','2005305','2005335',
             '2006001','2006032','2006060','2006091','2006121','2006152','2006182','2006213','2006244','2006274','2006305','2006335',
             '2007001','2007032','2007060','2007091','2007121','2007152','2007182','2007213','2007244','2007274','2007305','2007335',
             '2008001','2008032','2008061','2008092','2008122','2008153','2008183','2008214','2008245','2008275','2008306','2008336',
             '2009001','2009032','2009060','2009091','2009121','2009152','2009182','2009213','2009244','2009274','2009305','2009335',
             '2010001','2010032','2010060','2010091','2010121','2010152','2010182','2010213','2010244','2010274','2010305','2010335',
             '2011039','2011060','2011091','2011121','2011186','2011213','2011244','2011274','2011289','2011347',
             '2012001','2012032','2012061','2012080','2012153','2012183','2012214','2012245','2012311','2012336',
             '2013001', '2013032', '2013101', '2013121', '2013152', '2013182', '2013274', '2013305','2013335',
             '2014001', '2014062', '2014091', '2014121', '2014152', '2014213', '2014244', '2014274','2014305',
             '2015013', '2015032', '2015060', '2015091', '2015102', '2015180', '2015213', '2015244','2015345',
             '2016004', '2016029', '2016061', '2016129', '2016153', '2016183', '2016221', '2016319','2016346',
             '2017007', '2017076', '2017100', '2017123', '2017143', ]
        self.begin_gfo = \
            ['2018152', '2018182', '2018295', '2018305', '2018335',
             '2019001', '2019026', '2019060', '2019091', '2019121', '2019152', '2019182', '2019213','2019244', '2019274', '2019305', '2019335',
             '2020001', '2020032', '2020061', '2020092', '2020122', '2020153', '2020183', '2020214','2020245', '2020275', '2020306', '2020336',
             '2021001', '2021032', '2021060', '2021091', '2021121', '2021152', '2021182', '2021213','2021244', '2021274', '2021305', '2021335',
             '2022001', '2022032', '2022060', '2022091', '2022121', '2022152', '2022182', '2022213','2022244', '2022274', '2022305', '2022335',
             '2023001', '2023032', '2023060', '2023091', '2023121', '2023152', '2023182', '2023213','2023244', '2023274', '2023305', '2023335',
             '2024001', '2024032', '2024061', '2024092']
        self.end_grace = \
            ['2002120','2002137','2002243','2002273','2002304','2002334','2002365',
             '2003031','2003059','2003090','2003120','2003141','2003212','2003243','2003273','2003304','2003334','2003365',
             '2004013','2004060','2004091','2004121','2004152','2004182','2004213','2004244','2004274','2004305','2004335','2004366',
             '2005031','2005059','2005090','2005120','2005151','2005181','2005212','2005243','2005273','2005304','2005334','2005365',
             '2006031','2006059','2006090','2006120','2006151','2006181','2006212','2006243','2006273','2006304','2006334','2006365',
             '2007031','2007058','2007090','2007120','2007151','2007181','2007212','2007243','2007273','2007304','2007334','2007365',
             '2008031','2008060','2008091','2008121','2008152','2008182','2008213','2008244','2008274','2008305','2008335','2008366',
             '2009031','2009059','2009090','2009120','2009151','2009181','2009212','2009243','2009273','2009304','2009334','2009365',
             '2010031','2010059','2010090','2010120','2010151','2010181','2010212','2010243','2010273','2010304','2010334','2010361',
             '2011059','2011090','2011120','2011151','2011212','2011243','2011273','2011304','2011319','2012011',
             '2012031','2012060','2012091','2012109','2012182','2012213','2012244','2012269','2012335','2012366',
             '2013031', '2013057', '2013120', '2013151', '2013181', '2013212', '2013304', '2013334','2013365',
             '2014016', '2014090', '2014120', '2014151', '2014175', '2014243', '2014273', '2014304','2014334',
             '2015031', '2015059', '2015090', '2015120', '2015131', '2015212', '2015243', '2015270','2016003',
             '2016028', '2016060', '2016091', '2016152', '2016182', '2016211', '2016247', '2016345',
             '2017006','2017034', '2017104', '2017128', '2017142', '2017179']
        self.end_gfo = \
            ['2018181', '2018199', '2018313', '2018334', '2018365',
             '2019031', '2019066', '2019090', '2019120', '2019151', '2019181', '2019212', '2019243',
             '2019273', '2019304', '2019334', '2019365',
             '2020031', '2020060', '2020091', '2020121', '2020152', '2020182', '2020213', '2020244',
             '2020274', '2020305', '2020335', '2020366',
             '2021031', '2021059', '2021090', '2021120', '2021151', '2021181', '2021212', '2021243',
             '2021273', '2021304', '2021334', '2021365',
             '2022031', '2022059', '2022090', '2022120', '2022151', '2022181', '2022212', '2022243',
             '2022273', '2022304', '2022334', '2022365',
             '2023031', '2023059', '2023090', '2023120', '2023151', '2023181', '2023212', '2023243',
             '2023273', '2023304', '2023334', '2023365',
             '2024031', '2024060', '2024091', '2024121']

    def setSavePath(self, path='H:/Paper3/paper_result/mean_sp/'):
        self.SavePath = path
        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)
        print(f'Save path is: {self.SavePath}')
        return self

    def Synthetic(self, kind=AODtype.ATM):
        begdays = self.begin_gfo
        enddays = self.end_gfo
        # begdays=[
        #          '2013001', '2013032', '2013101', '2013121', '2013152', '2013182', '2013274', '2013305', '2013335',
        #          '2014001', '2014062', '2014091', '2014121', '2014152', '2014213', '2014244', '2014274', '2014305',
        #          '2015013', '2015032', '2015060', '2015091', '2015102', '2015180', '2015213', '2015244', '2015345',
        #          '2016004', '2016029', '2016061', '2016129', '2016153', '2016183', '2016221', '2016319', '2016346',
        #          '2017007', '2017076', '2017100', '2017123', '2017143',
        #          ]
        #
        # enddays=[
        #          '2013031', '2013057', '2013120', '2013151', '2013181', '2013212', '2013304', '2013334', '2013365',
        #          '2014016', '2014090', '2014120', '2014151', '2014175', '2014243', '2014273', '2014304', '2014334',
        #          '2015031', '2015059', '2015090', '2015120', '2015131', '2015212', '2015243', '2015270', '2016003',
        #          '2016028', '2016060', '2016091', '2016152', '2016182', '2016211', '2016247', '2016345', '2017006',
        #          '2017034', '2017104', '2017128', '2017142', '2017179',
        #          ]
        for i in np.arange(len(begdays)):
            BeginDate = self.day_of_year_to_date(ordinal=begdays[i])
            EndDate = self.day_of_year_to_date(ordinal=enddays[i])
            print(f'BeginDate is: {self.day_of_year_to_date(ordinal=begdays[i])}')
            print(f'EndDate is: {self.day_of_year_to_date(ordinal=enddays[i])}')
            print(kind.name)
            CS_temp = []
            daylist = GeoMathKit.dayListByDay(BeginDate,EndDate)
            for date in tqdm(daylist):
                date = date.strftime('%Y-%m-%d')
                for time in self.TimeEpoch:
                    CS = AOD_GFZ().load(self.LoadPath).setType(kind).setTime(date, time).getCS_old(Nmax=180)
                    CS = np.array(CS)
                    CS_temp.append(CS)
            CS_temp = np.array(CS_temp)
            CS_mean = np.mean(CS_temp, axis=0)
            fm = FormatWrite().setRootDir(self.SavePath)
            cs_file = CnmSnm(date=BeginDate, Nmax=180)
            cs_file.add(Cnm=CS_mean[0], Snm=CS_mean[1], epoch=time, date=BeginDate,
                        attribute=kind.name)
            fm.setCS(cs_file).GRACE_L2B(start=begdays[i], end=enddays[i], kind=kind.name)

    def date_to_day_of_year(self, year, month, day):
        date = datetime(int(year), int(month), int(day))
        day_of_year = date.timetuple().tm_yday
        return day_of_year

    def day_of_year_to_date(self, ordinal):
        year = int(str(ordinal)[:4])
        day_of_year = int(str(ordinal)[4:])
        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        return date.strftime('%Y-%m-%d')

def demo1():
    from pysrc.LoadFile.LoadAOD import AOD_GFZ, AODtype
    ad = AOD_GFZ().load('../data/Products/RL05').setType(AODtype.ATM).setTime('2005-01-01', '12:00:00')
    C, S = ad.getCS(ad.maxDegree)

    cs = CnmSnm(date='2005-01-01', Nmax=5)

    cs.add(Cnm=C, Snm=S, epoch='06:00:00', date='2005-01-01', attribute=AODtype.GLO.name)
    cs.add(Cnm=C, Snm=S, epoch='06:00:00', date='2005-01-01', attribute=AODtype.OCN.name)
    cs.add(Cnm=C, Snm=S, epoch='06:00:00', date='2005-01-01', attribute=AODtype.OBA.name)
    cs.add(Cnm=C, Snm=S, epoch='06:00:00', date='2005-01-01', attribute=AODtype.ATM.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.ATM.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.GLO.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.ATM.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.OCN.name)
    cs.add(Cnm=C, Snm=S, epoch='00:00:00', date='2005-01-01', attribute=AODtype.OBA.name)

    fm = FormatWrite().setRootDir('../result/products/')
    fm.setCS(cs).AODstyle()
    pass

def demoSyntheticAOD():
    a = SyntheticAOD()
    a.Synthetic()

def demoSyntheticGAX():
    a = SyntheticGAX()
    # Save_Names = ['GAA','GAB','GAC','GAD']
    Save_Names = ['GAD']
    # Kinds = [AODtype.ATM,AODtype.OCN,AODtype.GLO,AODtype.OBA]
    Kinds = [AODtype.OBA]
    for i in np.arange(len(Save_Names)):
        a.setSavePath(path=f'I:/CRALICOM/result/{Save_Names[i]}/')
        a.Synthetic(kind=Kinds[i])



if __name__ == '__main__':
    # demoSyntheticAOD()
    demoSyntheticGAX()
