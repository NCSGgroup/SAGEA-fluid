"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/10 10:15
@Description:
"""
import json
import sys

sys.path.append('../Configure')

from SaGEA.post_processing.geometric_correction.old.GeoMathKit import GeoMathKit
from scipy import signal
from pysrc.aux_fuction.load_file.LoadAOD import LoadFields, DataType
from pysrc.aliasing_model.Specify.Harmonic import Harmonic, LoveNumber, LoveNumberType, HarAnalysisType
from pysrc.aux_fuction.storage_file.StorageAOD import FormatWrite, CnmSnm
from pysrc.aliasing_model.Integral.SurPres2CS import SurPres2CS, RefEllipsoid, EllipsoidType
from SaGEA.post_processing.geometric_correction.old.GeoidUndulation import GeoidUndulation
import numpy as np
import os


class TideFit:
    """Only For CRA 6 hours"""
    tideFreq = {
        'P1': 14.9589314,
        'S1': 15.0,
        'K1': 15.0410686,
        'N2': 28.4397295,
        'M2': 28.9841042,
        'L2': 29.5284789,
        'T2': 29.9589333,
        'S2': 30.0,
        'R2': 30.0410667,
        'T3': 44.9589300,
        'S3': 45.0,
        'R3': 45.0410700
    }
    ref_point = '2007-01-01, 00:00:00'  # 00:00:00
    ref_point_mjd = 54101.000754444445  # MJD of the ref point.

    def __init__(self):
        self._timeEpoch = ["00:00:00","03:00:00", "06:00:00","09:00:00" ,"12:00:00","15:00:00","18:00:00","21:00:00"]
        self._sr = 1 / (int(self._timeEpoch[1][:2]) - int(self._timeEpoch[0][:2]))

        self._daylist = None
        self._dataDirIn = None
        self._dataDirOut = None
        self._butterworth = None

    def setDuration(self, begin='2007-01-01', end='2007-12-31'):
        """
        set the duration that tide fits will last
        :param begin:
        :param end:
        :return:
        """
        self._daylist = GeoMathKit.dayListByDay(begin, end)
        return self

    def setDataDir(self, dataDirIn: str, dataDirOut: str):
        """
        set the directory of surface pressure data deployed and the output tides
        :param dataDirIn:
        :param dataDirOut:
        :return:
        """
        self._dataDirIn = dataDirIn
        self._dataDirOut = dataDirOut

        isExists = os.path.exists(self._dataDirOut)
        if not isExists:
            os.makedirs(self._dataDirOut)
        return self

    def setButterworth(self, cutoff=3, order=3):
        """
        define the butterworth highpass filter
        :param cutoff: cutoff-frequency, [days], eg. cutoff=3
        :param order: order of the butterworth filter
        :return:
        """

        cutoff = 1 / (cutoff * 24)
        wn = 2 * cutoff / self._sr
        assert wn <= 1
        self._butterworth = signal.butter(order, wn, 'high')
        return self

    def fit(self,HighSpeed=True):
        """
        :return:
        """
        Tide = []
        b, a = self._butterworth[0], self._butterworth[1]
        SR = int(self._timeEpoch[1][:2]) - int(self._timeEpoch[0][:2])

        ld = LoadFields(data_path=self._dataDirIn)
        ld.setTime_for_tide()
        Nlat, NLon = ld.getNlatNlon()

        Nlen = 0
        for date in self._daylist:
            for time in self._timeEpoch:
                Nlen += 1

        interval = np.round(1 / self._sr)

        refDate = TideFit.ref_point.split(',')[0]
        refTime = int(TideFit.ref_point.split(',')[1].split(':')[0])

        date = self._daylist[0]
        days = len(GeoMathKit.dayListByDay(begin=refDate, end=date.strftime("%Y-%m-%d")))
        '''calculate how may hours between the given time and the starting point'''
        if days == 0:
            days = len(GeoMathKit.dayListByDay(begin=date.strftime("%Y-%m-%d"), end=refDate))
            days = days * (-1)
            hours = (days + 1) * 24 - refTime
        else:
            hours = (days - 1) * 24 - refTime

        xdata = np.arange(Nlen) * interval + hours

        dm = self.designMatrix(xdata)
        if interval == 6:
            dm = dm[:, :16]
            # dm = dm[:, :18]
        if HighSpeed:
            Assemble = []
            N = 0
            Ntoll = len(self._daylist)
            for date in self._daylist:
                print('Progress: {:.1%}'.format(N / Ntoll))
                for time in self._timeEpoch:
                    ld.setTime_for_tide(date=date.strftime("%Y-%m-%d"), time=time)

                    Assemble.append(ld.getField(kind=DataType.PSFC).flatten())
                N += 1

            Assemble = np.array(Assemble)

            '''butterworth filter'''
            sf = signal.filtfilt(b, a, Assemble, axis=0)

            sf_arr = np.array(sf)
            # print("dm shape:", dm.shape)
            # print("sf_arr shape:", sf_arr.shape)
            # print("NaN in dm?", np.isnan(dm).any())
            # print("NaN in sf_arr?", np.isnan(sf_arr).any())
            # print("Inf in dm?", np.isinf(dm).any())
            # print("Inf in sf_arr?", np.isinf(sf_arr).any())
            pp = np.linalg.lstsq(dm.copy(), sf_arr)[0]
            # Tide.append(pp)

            '''Complete'''

            Tide = np.array(pp)
            Tide = Tide.reshape(len(Tide[:, 0]), Nlat, NLon).transpose(1, 0, 2)
            np.save(self._dataDirOut + 'All_grid.npy', Tide)

        else:
            for indexByLat in range(Nlat):
                print('Progress: {:.1%}'.format(indexByLat / Nlat))
                pointsBylat = []

                '''get time-series of surface pressure data'''
                for date in self._daylist:
                    for time in self._timeEpoch:

                        ld.setTime_for_tide(date=date.strftime("%Y-%m-%d"), time=time)
                        pointsBylat.append(ld.getField(kind=DataType.PSFC)[indexByLat, :])

                pointsBylat = np.array(pointsBylat)

                sf_list = []
                for index in np.arange(NLon):
                    '''butterworth filter'''
                    sf = signal.filtfilt(b, a, pointsBylat[:, index])
                    sf_list.append(sf)

                    pass



                sf_arr = np.array(sf_list).transpose()
                pp = np.linalg.lstsq(dm.copy(), sf_arr)[0]
                Tide.append(pp)

            '''Complete'''
            Tide = np.array(Tide)

            np.save(self._dataDirOut + 'All_grid.npy', Tide)

    def designMatrix(self, xdata):
        """
        use the least square to speed-up computation.
        :param x:
        :return:
        """

        p1 = np.deg2rad(self.tideFreq['P1'])
        s1 = np.deg2rad(self.tideFreq['S1'])
        k1 = np.deg2rad(self.tideFreq['K1'])

        n2 = np.deg2rad(self.tideFreq['N2'])
        m2 = np.deg2rad(self.tideFreq['M2'])
        l2 = np.deg2rad(self.tideFreq['L2'])

        t2 = np.deg2rad(self.tideFreq['T2'])
        s2 = np.deg2rad(self.tideFreq['S2'])
        r2 = np.deg2rad(self.tideFreq['R2'])

        t3 = np.deg2rad(self.tideFreq['T3'])
        s3 = np.deg2rad(self.tideFreq['S3'])
        r3 = np.deg2rad(self.tideFreq['R3'])

        a = np.ones(len(xdata))
        b = xdata
        P1_c = np.cos(xdata * p1)
        P1_s = np.sin(xdata * p1)
        S1_c = np.cos(xdata * s1)
        S1_s = np.sin(xdata * s1)
        K1_c = np.cos(xdata * k1)
        K1_s = np.sin(xdata * k1)
        N2_c = np.cos(xdata * n2)
        N2_s = np.sin(xdata * n2)
        M2_c = np.cos(xdata * m2)
        M2_s = np.sin(xdata * m2)
        L2_c = np.cos(xdata * l2)
        L2_s = np.sin(xdata * l2)
        T2_c = np.cos(xdata * t2)
        T2_s = np.sin(xdata * t2)
        S2_c = np.cos(xdata * s2)
        S2_s = np.sin(xdata * s2)
        R2_c = np.cos(xdata * r2)
        R2_s = np.sin(xdata * r2)
        T3_c = np.cos(xdata * t3)
        T3_s = np.sin(xdata * t3)
        S3_c = np.cos(xdata * s3)
        S3_s = np.sin(xdata * s3)
        R3_c = np.cos(xdata * r3)
        R3_s = np.sin(xdata * r3)

        dm = [a, b,
              P1_c, P1_s,
              S1_c, S1_s,
              K1_c, K1_s,
              N2_c, N2_s,
              M2_c, M2_s,
              L2_c, L2_s,
              T2_c, T2_s,
              S2_c, S2_s,
              R2_c, R2_s,
              T3_c, T3_s,
              S3_c, S3_s,
              R3_c, R3_s
              ]

        dm = np.array(dm).transpose()
        return dm

    def separateTidePres(self, isLessThanSix=False):

        AllTide = np.load(self._dataDirOut + 'All_grid.npy')

        out = self._dataDirOut + 'TidePressure/'
        isExists = os.path.exists(out)
        if not isExists:
            os.makedirs(out)

        p1Cos = AllTide[:, 2, :]
        p1Sin = AllTide[:, 3, :]
        np.save(out + 'P1.npy', np.array([p1Cos, p1Sin]))

        s1Cos = AllTide[:, 4, :]
        s1Sin = AllTide[:, 5, :]
        np.save(out + 'S1.npy', np.array([s1Cos, s1Sin]))

        k1Cos = AllTide[:, 6, :]
        k1Sin = AllTide[:, 7, :]
        np.save(out + 'K1.npy', np.array([k1Cos, k1Sin]))

        n2Cos = AllTide[:, 8, :]
        n2Sin = AllTide[:, 9, :]
        np.save(out + 'N2.npy', np.array([n2Cos, n2Sin]))

        m2Cos = AllTide[:, 10, :]
        m2Sin = AllTide[:, 11, :]
        np.save(out + 'M2.npy', np.array([m2Cos, m2Sin]))

        l2Cos = AllTide[:, 12, :]
        l2Sin = AllTide[:, 13, :]
        np.save(out + 'L2.npy', np.array([l2Cos, l2Sin]))

        t2Cos = AllTide[:, 14, :]
        t2Sin = AllTide[:, 15, :]
        np.save(out + 'T2.npy', np.array([t2Cos, t2Sin]))

        if isLessThanSix:
            s2Cos = AllTide[:, 16, :]
            s2Sin = AllTide[:, 17, :]
            np.save(out + 'S2.npy', np.array([s2Cos, s2Sin]))

            r2Cos = AllTide[:, 18, :]
            r2Sin = AllTide[:, 19, :]
            np.save(out + 'R2.npy', np.array([r2Cos, r2Sin]))

            t3Cos = AllTide[:, 20, :]
            t3Sin = AllTide[:, 21, :]
            np.save(out + 'T3.npy', np.array([t3Cos, t3Sin]))

            s3Cos = AllTide[:, 22, :]
            s3Sin = AllTide[:, 23, :]
            np.save(out + 'S3.npy', np.array([s3Cos, s3Sin]))

            r3Cos = AllTide[:, 24, :]
            r3Sin = AllTide[:, 25, :]
            np.save(out + 'R3.npy', np.array([r3Cos, r3Sin]))

        pass

    def generateTide_bySphere(self,end:str, tides: dict, Nmax: int, lat, lon):
        """
        The tide is generated assuming the Earth is sphere
        :param Nmax: Max degree of SH expansion for the tide
        :param tides:
        example
        tides = {
            'S1': True,
            'S2': True,
            'M2': False
        }
        :param lat: e.g., np.arange(90, -90.1, -0.5)
        :param lon: e.g., np.arange(0, 360, 0.5)
        :return:
        """

        indir = self._dataDirOut + 'TidePressure/'
        outdir = self._dataDirOut + 'TideGeoCS_sphere/'

        isExists = os.path.exists(outdir)
        if not isExists:
            os.makedirs(outdir)

        tidesInfo = {}
        tidesDoodsonMatrix = {}

        LN = LoveNumber('../../../Data/aux_fuction/')
        HM = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)

        MaxDeg = Nmax
        nutarg_first = GeoMathKit.doodsonArguments(TideFit.ref_point_mjd)
        Pnm = GeoMathKit.getPnmMatrix(lat, Nmax=MaxDeg, option=1)
        fm = FormatWrite().setRootDir(outdir)

        for tide in tides.keys():
            if tides[tide]:
                # self.__tidesInfo[tide] = np.load(tideDir + tide + '_grid' + '.npy')
                tidesInfo[tide] = np.load(indir + tide + '.npy')
                tidesDoodsonMatrix[tide] = self.__doodsonMatrix(tide)

        for tide in tidesInfo.keys():
            print(tide)
            cs = CnmSnm(date=tide, Nmax=MaxDeg)
            st = tidesInfo[tide]
            # A = st[:, :, 0].flatten()
            # B = st[:, :, 1].flatten()
            A = st[0, :, :]
            B = st[1, :, :]

            wt0 = np.matmul(tidesDoodsonMatrix[tide], nutarg_first)
            Anew = A * np.cos(wt0) - B * np.sin(wt0)
            Bnew = A * np.sin(wt0) + B * np.cos(wt0)
            cnmCos, snmCos = HM.analysis(Nmax=MaxDeg, Gqij=Anew, lat=lat, lon=lon, PnmMat=Pnm,
                                         kind=HarAnalysisType.Pressure)
            cs.add(Cnm=cnmCos, Snm=snmCos, epoch='00:00:00', date=tide, attribute='cos')

            cnmSin, snmSin = HM.analysis(Nmax=MaxDeg, Gqij=Bnew, lat=lat, lon=lon, PnmMat=Pnm,
                                         kind=HarAnalysisType.Pressure)
            cs.add(Cnm=cnmSin, Snm=snmSin, epoch='01:00:00', date=tide, attribute='sin')
            '''format writing'''
            '''write results'''
            fm.setCS(cs).TideStyle(tide,range=end)

        pass

    def generateTide_byTopography(self,end:str,tides: dict, Nmax: int, lat, lon):
        """
        The tide is generated considering an actual Earth.
        :param Nmax: Max degree of SH expansion for the tide
        :param tides:
        example
        tides = {
            'S1': True,
            'S2': True,
            'M2': False
        }
        :param lat: e.g., np.arange(90, -90.1, -0.5)
        :param lon: e.g., np.arange(0, 360, 0.5)
        :return:
        """

        indir = self._dataDirOut + 'TidePressure/'
        outdir = self._dataDirOut + 'TideGeoCS_topography/'

        '''Configure for the surface pressure integration'''
        LN = LoveNumber('../../../Data/aux_fuction/')
        ell = RefEllipsoid(EllipsoidType.GRS80_IERS2010)
        undulation = GeoidUndulation(EllipsoidType.GRS80_IERS2010).getGeoid(lat, lon).flatten()
        ld = LoadFields(data_path='H:/ERA5/model level/')
        ld.setTime_model(OnlyPressure=True)
        # orography = ld.getField(DataType.PHISFC) * Constants.g_wmo
        orography = ld.getField(DataType.PHISFC)
        sf2cs = SurPres2CS().setPar(lat=lat, lon=lon, orography=orography, undulation=undulation, elliposid=ell,
                                    loveNumber=LN)

        isExists = os.path.exists(outdir)
        if not isExists:
            os.makedirs(outdir)

        tidesInfo = {}
        tidesDoodsonMatrix = {}

        LN = LoveNumber('../../../Data/aux_fuction/')
        HM = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)

        MaxDeg = Nmax
        nutarg_first = GeoMathKit.doodsonArguments(TideFit.ref_point_mjd)
        Pnm = GeoMathKit.getPnmMatrix(lat, Nmax=MaxDeg, option=1)
        fm = FormatWrite().setRootDir(outdir)

        for tide in tides.keys():
            if tides[tide]:
                # self.__tidesInfo[tide] = np.load(tideDir + tide + '_grid' + '.npy')
                tidesInfo[tide] = np.load(indir + tide + '.npy')
                tidesDoodsonMatrix[tide] = self.__doodsonMatrix(tide)

        for tide in tidesInfo.keys():
            print(tide)
            cs = CnmSnm(date=tide, Nmax=MaxDeg)
            st = tidesInfo[tide]
            # A = st[:, :, 0].flatten()
            # B = st[:, :, 1].flatten()
            A = st[0, :, :]
            B = st[1, :, :]

            wt0 = np.matmul(tidesDoodsonMatrix[tide], nutarg_first)
            Anew = A * np.cos(wt0) - B * np.sin(wt0)
            Bnew = A * np.sin(wt0) + B * np.cos(wt0)

            cnmCos, snmCos = sf2cs.setPressure(pressure=Anew, maxDeg=MaxDeg)
            # cnmCos, snmCos = HM.analysis(Nmax=MaxDeg, Gqij=Anew, lat=lat, lon=lon, PnmMat=Pnm,
            #                              kind=HarAnalysisType.Pressure)
            cs.add(Cnm=cnmCos, Snm=snmCos, epoch='00:00:00', date=tide, attribute='cos')

            cnmSin, snmSin = sf2cs.setPressure(pressure=Bnew, maxDeg=MaxDeg)
            # cnmSin, snmSin = HM.analysis(Nmax=MaxDeg, Gqij=Bnew, lat=lat, lon=lon, PnmMat=Pnm,
            #                              kind=HarAnalysisType.Pressure)
            cs.add(Cnm=cnmSin, Snm=snmSin, epoch='01:00:00', date=tide, attribute='sin')
            '''format writing'''
            '''write results'''
            fm.setCS(cs).TideStyle(tide,range=end)

        pass

    def __doodsonMatrix(self, tide='S1'):

        DoodsonNumber = {'S1': '164.556', 'S2': '273.555', 'Sa': '056.554', 'Ssa': '057.555',
                         'P1': '163.555', 'K1': '165.555', 'N2': '245.655', 'M2': '255.555',
                         'L2': '265.455', 'T2': '272.556', 'R2': '274.554', 'T3': '381.555',
                         'S3': '382.555', 'R3': '383.555'}

        doodson = DoodsonNumber[tide]

        doodsonMatrix = np.zeros(6)

        doodsonMatrix[0] = int(doodson[0]) - 0
        doodsonMatrix[1] = int(doodson[1]) - 5
        doodsonMatrix[2] = int(doodson[2]) - 5
        doodsonMatrix[3] = int(doodson[4]) - 5
        doodsonMatrix[4] = int(doodson[5]) - 5
        doodsonMatrix[5] = int(doodson[6]) - 5

        return doodsonMatrix
    @staticmethod
    def DefaultConfig(isWrite=True):
        config = {'dataDirIn': '../../../Data/ERA5/model level/',
                  'dataDirOut': '../../../Result/Paper3/model_tide/2007_2014/',
                  'BeginDate':'2007-01-01',
                  'EndDate':'2014-12-31'}

        if isWrite:
            with open('../SetFile/TideFit.json', 'w') as f:
                f.write(json.dumps(config,indent=4))
        return config

class Detide:

    def __init__(self):

        self.__refpoint = TideFit.ref_point
        self.__tidesInfo = {}
        pass

    def setTides(self, tides: dict, tideDir: str):
        """
        :param tides:
        example
        tides = {
            'S1': True,
            'S2': True,
            'M2': False
        }
        :param tideDir:
        :return:
        """
        for tide in tides.keys():
            if tides[tide]:
                # self.__tidesInfo[tide] = np.load(tideDir + tide + '_grid' + '.npy')
                self.__tidesInfo[tide] = np.load(tideDir + tide + '.npy')

        return self

    def setRefPoint(self, refpoint):
        """
        for easy use, the tide phase are derived with self-defined time system, inferring that the
        de-tide process needs the info of start point to get the tide value at given time.
        :param refpoint: e.g., '2007-01-01,00:00:00'
        :return:
        """
        self.__refpoint = refpoint
        return self

    def remove(self, pressure, date: str, time: str):
        """
        remove tides from the given pressure
        :param pressure: input pressure with the same lat and lon with tide grids, see Tidefit.cls
        :param date: date of the pressure
        :param time: time of the pressure like 18:00:00; the basic unit is hour.
        :return: pressure after de-tides. One-dimension [N*M]
        """
        refDate = self.__refpoint.split(',')[0]
        refTime = int(self.__refpoint.split(',')[1].split(':')[0])
        endTime = int(time.split(':')[0])

        days = len(GeoMathKit.dayListByDay(begin=refDate, end=date))
        '''calculate how may hours between the given time and the starting point'''
        if days == 0:
            days = len(GeoMathKit.dayListByDay(begin=date, end=refDate))
            days = days * (-1)
            hours = (days + 1) * 24 + endTime - refTime
        else:
            hours = (days - 1) * 24 + endTime - refTime

        for tide in self.__tidesInfo.keys():
            st = self.__tidesInfo[tide]
            wt = np.deg2rad(TideFit.tideFreq[tide]) * hours
            pressure -= st[0, :, :].flatten() * np.cos(wt) + st[1, :, :].flatten() * np.sin(wt)
            # pressure -= st[:, :, 0] * np.cos(wt) + st[:, :, 1] * np.sin(wt)
            pass

        return pressure



def demo():
    cutoff = 3
    order = 3
    cutoff = 1 / (cutoff * 24)
    wn = 2 * cutoff / 6
    assert wn <= 1
    b, a = signal.butter(order, wn, 'high')

    pass

  # tf.setDataDir(dataDirIn='../data/CRA.grib2/', dataDirOut='../result/tide/2007_2014/')
    # tf.setDuration(begin='2007-01-01', end='2014-12-31')

    # tf.setDataDir(dataDirIn='../data/CRA.grib2/',dataDirOut='../result/tide/2012_2019/')
    # tf.setDuration(begin='2012-01-01', end='2019-12-31')
def demo2():
    tf = TideFit().setButterworth()
    tf.setDataDir(dataDirIn='H:/ERA5/pressure level/', dataDirOut='H:/Paper3/paper_result/pressure_tide/xxx/')
    tf.setDuration(begin='2020-01-01', end='2020-01-05')
    tf.fit()
    tf.separateTidePres(isLessThanSix=True)
    tides = {
        'P1': True,
        'S1': True,
        'K1': True,
        'N2': True,
        'M2': True,
        'L2': True,
        'T2': True,
        'S2': True,
        'R2': True,
        'T3': True,
        'S3': True,
        'R3': True
    }
    lat = np.arange(90, -90.1, -0.5)
    lon = np.arange(0, 360, 0.5)
    tf.generateTide_bySphere(end='2014',tides=tides, Nmax=180, lat=lat, lon=lon)
    tf.generateTide_byTopography(end='2014',tides=tides, Nmax=180, lat=lat, lon=lon)

def demo3():
    with open('../Settings/TideFit.json', 'r') as f:
        config = json.load(f)
    tf = TideFit().setButterworth()
    tf.setDataDir(dataDirIn=config['dataDirIn'], dataDirOut=config['dataDirOut'])
    tf.setDuration(begin=config['BeginDate'], end=config['EndDate'])
    tf.fit()
    tf.separateTidePres(isLessThanSix=True)
    tides = {
        'P1': True,
        'S1': True,
        'K1': True,
        'N2': True,
        'M2': True,
        'L2': True,
        'T2': True,
        'S2': True,
        'R2': True,
        'T3': True,
        'S3': True,
        'R3': True
    }
    lat = np.arange(90, -90.1, -0.5)
    lon = np.arange(0, 360, 0.5)
    tf.generateTide_bySphere(end='XXX', tides=tides, Nmax=180, lat=lat, lon=lon)
    # tf.generateTide_byTopography(endyear='2014', tides=tides, Nmax=180, lat=lat, lon=lon)



if __name__ == '__main__':
    demo3()


