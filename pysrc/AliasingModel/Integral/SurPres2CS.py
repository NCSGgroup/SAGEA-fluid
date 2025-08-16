"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/7/21 下午6:19
@Description:
"""
import numpy as np

from pysrc.Auxiliary.GeoMathKit import GeoMathKit
from pysrc.PostProcessing.old.GeoidUndulation import GeoidUndulation
from pysrc.AliasingModel.Specify.Harmonic import Harmonic
from pysrc.PostProcessing.old.LoveNumber import LoveNumber
from pysrc.PostProcessing.old.RefEllipsoid import RefEllipsoid
from pysrc.PostProcessing.old.Setting import EllipsoidType, LoveNumberType, DataType, Constants, HarAnalysisType


class SurPres2CS:
    """
    Transform the surface pressure to Stokes coefficients considering the orography
    For instance, get harmonic coefficients format of the tides, as the pressure from tide-fit is a gridded data
    associated with orography.
    """

    def __init__(self):
        self.__lat = None
        self.__lon = None
        self.__orography = None
        self.__undulation = None
        self.__ellipsoid = None
        self.__loveNumber = None
        # self.__surG = self.__getSurGrav()
        pass

    def setPar(self, lat, lon, orography, undulation, elliposid: RefEllipsoid, loveNumber: LoveNumber):
        """
        set parameters
        :param lat: N
        :param lon: M
        :param orography: N*M, surface geo-potential
        :param undulation: N*M
        :param elliposid:
        :param loveNumber:
        :return:
        """
        self.__ellipsoid = elliposid
        self.__lat = lat
        self.__lon = lon
        self.__orography = orography
        self.__undulation = undulation
        self.__loveNumber = loveNumber

        return self

    def setPressure(self, pressure, maxDeg: int):
        """
        Set pressure or other physical fields input and carry out surface ellipsoidal integral
        :param pressure: pressure field or EWH field or ...  [N*M]
        :param maxDeg: up to given max degree/order that output Stokes coefficients will get
        :return:
        """
        pressure = pressure.flatten()
        lonMesh, latMesh = np.meshgrid(self.__lon, self.__lat)
        sh = np.shape(lonMesh)
        lonMesh = lonMesh.flatten()
        latMesh = latMesh.flatten()
        '''reference ellipsoid + geoid undulation + orography'''
        z = self._getHeight(latMesh, self.__orography / Constants.g_wmo)
        r = self._getR(latMesh) + self.__undulation + z
        # r = self._getR(latMesh) + z
        ar = r / self.__ellipsoid.SemimajorAxis
        g = self._getG(latMesh, z)

        deltaI = []
        iniPower = ar
        for i in range(maxDeg + 1):
            I_lev = np.zeros(np.size(ar))
            iniPower = iniPower * ar
            # I_lev = np.power(ar, i + 2) * pressure / self._getG(latMesh, z)
            # I_lev = iniPower * pressure / self._getG(latMesh, z)
            I_lev = iniPower * pressure / g
            deltaI.append(I_lev)

        '''Classical one'''
        # Pnm = GeoMathKit.getPnm(self.__lat, maxDeg, 1)  # run for once is enough
        #
        # hm = Harmonic2(self.__loveNumber, Parallel=-1).setLoveNumMethod(LoveNumberType.Wang)
        # cnm2, snm2 = hm.analysis(Nmax=maxDeg, Inner=deltaI, lat=self.__lat, lon=self.__lon, Pnm=Pnm,
        #                        kind=HarAnalysisType.InnerIntegral)

        '''Einstein tensor'''
        Pnm = GeoMathKit.getPnmMatrix(self.__lat, maxDeg, 1)  # run for once is enough

        hm = Harmonic(self.__loveNumber).setLoveNumMethod(LoveNumberType.Wang)

        deltaI = np.array(deltaI).reshape((maxDeg+1, sh[0], sh[1]))
        cnm, snm = hm.analysis(Nmax=maxDeg, Gqij=deltaI, lat=self.__lat, lon=self.__lon, PnmMat=Pnm,
                               kind=HarAnalysisType.InnerIntegral)

        return cnm, snm

    def setPressure_inner(self, pressure, maxDeg: int):
        """
        Set pressure or other physical fields input and carry out surface ellipsoidal integral
        :param pressure: pressure field or EWH field or ...  [N*M]
        :param maxDeg: up to given max degree/order that output Stokes coefficients will get
        :return:
        """
        pressure = pressure.flatten()
        lonMesh, latMesh = np.meshgrid(self.__lon, self.__lat)
        sh = np.shape(lonMesh)
        lonMesh = lonMesh.flatten()
        latMesh = latMesh.flatten()
        '''reference ellipsoid + geoid undulation + orography'''
        z = self._getHeight(latMesh, self.__orography / Constants.g_wmo)
        r = self._getR(latMesh) + self.__undulation + z
        # r = self._getR(latMesh) + z
        ar = r / self.__ellipsoid.SemimajorAxis
        g = self._getG(latMesh, z)

        deltaI = []
        iniPower = ar
        for i in range(maxDeg + 1):
            I_lev = np.zeros(np.size(ar))
            iniPower = iniPower * ar
            # I_lev = np.power(ar, i + 2) * pressure / self._getG(latMesh, z)
            # I_lev = iniPower * pressure / self._getG(latMesh, z)
            I_lev = iniPower * pressure / g
            deltaI.append(I_lev)

        '''Classical one'''
        # Pnm = GeoMathKit.getPnm(self.__lat, maxDeg, 1)  # run for once is enough
        #
        # hm = Harmonic2(self.__loveNumber, Parallel=-1).setLoveNumMethod(LoveNumberType.Wang)
        # cnm2, snm2 = hm.analysis(Nmax=maxDeg, Inner=deltaI, lat=self.__lat, lon=self.__lon, Pnm=Pnm,
        #                        kind=HarAnalysisType.InnerIntegral)

        # '''Einstein tensor'''
        # Pnm = GeoMathKit.getPnmMatrix(self.__lat, maxDeg, 1)  # run for once is enough
        #
        # hm = Harmonic(self.__loveNumber).setLoveNumMethod(LoveNumberType.Wang)
        #
        # deltaI = np.array(deltaI).reshape((maxDeg+1, sh[0], sh[1]))
        # cnm, snm = hm.analysis(Nmax=maxDeg, Gqij=deltaI, lat=self.__lat, lon=self.__lon, PnmMat=Pnm,
        #                        kind=HarAnalysisType.InnerIntegral)

        return np.array(deltaI)

    # def setPressure2(self, pressure, maxDeg: int):
    #     """
    #     Set pressure or other physical fields input and carry out surface ellipsoidal integral
    #     :param pressure: pressure field or EWH field or ...  [N*M]
    #     :param maxDeg: up to given max degree/order that output Stokes coefficients will get
    #     :return:
    #     """
    #
    #     lonMesh, latMesh = np.meshgrid(self.__lon, self.__lat)
    #     lonMesh = lonMesh.flatten()
    #     latMesh = latMesh.flatten()
    #     '''reference ellipsoid + geoid undulation + orography'''
    #     z = self._getHeight(latMesh, self.__orography / Constants.g_wmo)
    #     # r = self._getR(latMesh) + self.__undulation + z
    #     r = self._getR(latMesh) + z
    #     ar = r / self.__ellipsoid.SemimajorAxis
    #     ar2 = r/self._getR(latMesh)
    #     g = self.__surG
    #
    #
    #     deltaI = []
    #     iniPower = ar* ar2**2
    #     for i in range(maxDeg + 1):
    #         I_lev = np.zeros(np.size(ar))
    #         iniPower = iniPower * ar
    #         # I_lev = np.power(ar, i + 2) * pressure / self._getG(latMesh, z)
    #         I_lev = iniPower * pressure / g
    #         deltaI.append(I_lev)
    #
    #     Pnm = GeoMathKit.getPnm(self.__lat, maxDeg, 1)  # run for once is enough
    #
    #     hm = Harmonic(self.__loveNumber, Parallel=-1).setLoveNumMethod(LoveNumberType.Wang)
    #     cnm, snm = hm.analysis(Nmax=maxDeg, Inner=deltaI, lat=self.__lat, lon=self.__lon, Pnm=Pnm,
    #                            kind=HarAnalysisType.InnerIntegral)
    #
    #     return cnm, snm

    def _getR(self, lat):
        # % Input:
        # % lat...Latitudes in degree
        # % ellipsoid...ellipsoidal
        # Parameters
        # % Output:
        # % r...Latitutde - dependent Radius

        B = lat * np.pi / 180  # latitutde in [rad]
        R = self.__ellipsoid.SemimajorAxis
        e = self.__ellipsoid.Eccentricity

        r = R * np.sqrt(1 - (e ** 2) * (np.sin(B) ** 2))

        return r

    def _getHeight(self, theta, H):
        """

        :param theta: latitude in degree
        :param H: geopoetntial height
        :return:
        """
        B = (90 - theta) * np.pi / 180  # co-latitude in [rad]
        # print('B shape:',B.shape)
        # print('H shape:',H.shape)

        '''AOD RL06 document: get the real height from geopotential height'''
        z = (1 - 0.002644 * np.cos(2 * B)) * H + (1 - 0.0089 * np.cos(2 * B)) * np.power(H, 2) / 6.245e6

        return z

    def _getG(self, theta, z):
        """

        :param theta:
        :param z:
        :return:
        """
        '''Boy and Chao (2005), Precise evaluation of atmospheric loading effects
        on Earth’s time-variable gravity field. Eq. 18 and 19 '''
        # notice: the input z must be the real height

        B = (90 - theta) * np.pi / 180  # co-latitutde in [rad]

        R = self.__ellipsoid.SemimajorAxis
        g_theta_z = self.__ellipsoid.je * (1 + 5.2885e-3 * np.power(np.cos(B), 2) - 5.9e-6 * np.power(np.cos(2 * B), 2)) \
                    * (1 - 2 * (1.006803 - 0.060706 * np.power(np.cos(B), 2)) * z / R + 3 * (z / R) ** 2)

        return g_theta_z

    # def __getSurGrav(self):
    #     elltype = EllipsoidType.GRS80_IERS2010
    #     ellipsoid = RefEllipsoid(elltype)
    #
    #     lmax = 180 - 1
    #     SH = Gif48().load('../data/Auxiliary/GIF48.gfc').getCS(lmax)
    #
    #     topo_sh = np.zeros((2, lmax + 1, lmax + 1))
    #     topo_sh[0, :, :] = GeoMathKit.CS_1dTo2d(SH[0])
    #     topo_sh[1, :, :] = GeoMathKit.CS_1dTo2d(SH[1])
    #     topo_sh = pysh.SHGravCoeffs.from_array(topo_sh, gm=0.3986004415E+15, r0=0.6378136300E+07)
    #     topo_sh.set_omega(7.2921151467e-5)
    #     grav = topo_sh.expand(a=ellipsoid.SemimajorAxis, f=ellipsoid.Flattening, lmax=lmax, lmax_calc=lmax,
    #                           normal_gravity=False, sampling=2, extend=True)
    #
    #     gg = grav.total.data[:, 0:-1].T
    #
    #     return gg.flatten()


# def CRALICOM():
#     from pysrc.ExtractNC import ReadNC
#     from pysrc.SetFile import SynthesisType
#
#     elltype = EllipsoidType.GRS80_IERS2010
#     ell = RefEllipsoid(elltype)
#     LN = LoveNumber('../data/Auxiliary/')
#
#     lat = np.arange(90, -90.1, -0.5)
#     lon = np.arange(0, 360, 0.5)
#
#     undulation = GeoidUndulation(elltype).getGeoid(lat, lon).flatten()
#
#     PHISFC = '../data/ERA_interim/PHISFC_interim_invariant.nc'
#     orography = ReadNC().setPar(PHISFC, DataType.PHISFC).read()[0].flatten()
#
#     sp = np.load('../result/temp/AS1.npy')
#
#     sf2cs = SurPres2CS().setPar(lat=lat, lon=lon, orography=orography, undulation=undulation, elliposid=ell,
#                                 loveNumber=LN)
#
#     cnm, snm = sf2cs.setPressure(pressure=sp.flatten(), maxDeg=100)
#
#     HM = Harmonic(LN)
#
#     grids = HM.synthesis(Cnm=cnm, Snm=snm, lat=lat, lon=lon, Nmax=100, kind=SynthesisType.Pressure)
#
#     np.save('../result/temp/AS1_synthesis', grids)
#
#     pass

def demo():

    pass


if __name__ == '__main__':
    demo()
