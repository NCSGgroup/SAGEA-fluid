"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/7/1 下午6:55
@Description:
"""
import os
import numpy as np
from datetime import datetime

from lib.AOD.MathKit.GeoMathKit import GeoMathKit



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

    def setCS(self, CS: CnmSnm):
        self.__CS = CS
        res = CS.date.split('-')

        try:
            subdir = res[0] + '-' + res[1]
        except:
            return self

        subdir = os.path.join(self.__fileDir, subdir)

        if not os.path.exists(subdir):
            os.makedirs(subdir)

        self.__fileFullPath = subdir + os.sep + CS.date + '.asc'
        self.__fileErrPath = subdir + '/' + res[0] + res[1] + res[2] + '_'

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





def demo1():
    from lib.AOD.LoadFile.LoadSH import AOD_GFZ, AODtype
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


if __name__ == '__main__':
    demo1()
