"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/8 9:35
@Description:
"""

import json

import numpy as np

from pysrc.AD.geoheight.GeoidUndulation import GeoidUndulation
from pysrc.AD.geoheight.GeopotentialHeight import GeopotentialHeight, LoadFields
from pysrc.AD.geoheight.RefEllipsoid import RefEllipsoid
from pysrc.ancillary.constant.Setting import IntegralChoice, EllipsoidType, Constants


class InnerIntegral:

    """
    InnerIntegral ~=  rhow * h = Pressure/g
    """

    def __init__(self, gh: GeopotentialHeight, ld: LoadFields):
        self._maxDeg = None
        self._ellipsoid = None
        self.__Geoid = None
        # self._ellipsoid = RefEllipsoid(EllipsoidType.GRS80_IERS2010)
        self._gh = gh
        self._ld = ld
        self._processors = -10000
        pass

    def configure(self, config: dict):
        """
        using files to configure the class
        :param config:
        :return:
        """
        self.setMaxDeg(config['MaxDegree']).setParallel(config['Parallel']).setEllipsoid(
            EllipsoidType[config['Ellipsoid']])

        return self

    def setMaxDeg(self, degree_order: int):
        self._maxDeg = degree_order
        return self

    def setParallel(self, processors: int = 5):
        self._processors = processors
        return self

    def setEllipsoid(self, ell: EllipsoidType):
        """
        define the ellipsoid
        :param ell:
        :return:
        """
        self._ellipsoid = RefEllipsoid(ell)
        return self

    def setGeoid(self, Geoid):
        self.__Geoid = Geoid
        return self

    def deltaI(self):
        pass

    def _getHeight(self, theta, H):
        """

        :param theta: latitude in degree
        :param H: geopoetntial height
        :return:
        """
        B = (90 - theta) * np.pi / 180  # co-latitude in [rad]

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

        R = self._ellipsoid.SemimajorAxis
        g_theta_z = self._ellipsoid.je * (1 + 5.2885e-3 * np.power(np.cos(B), 2) - 5.9e-6 * np.power(np.cos(2 * B), 2)) \
                    * (1 - 2 * (1.006803 - 0.060706 * np.power(np.cos(B), 2)) * z / R + 3 * (z / R) ** 2)

        return g_theta_z

    def _getR(self, lat):
        # % Input:
        # % lat...Latitudes in degree
        # % ellipsoid...ellipsoidal
        # Parameters
        # % Output:
        # % r...Latitutde - dependent Radius

        B = lat * np.pi / 180  # latitutde in [rad]
        R = self._ellipsoid.SemimajorAxis
        e = self._ellipsoid.Eccentricity

        r = R * np.sqrt(1 - (e ** 2) * (np.sin(B) ** 2))

        return r

    def _getG_EF(self, theta, z):
        '''see the paper JGR of Ehsan, Eq.17, Eq.18'''
        B = (90 - theta) * np.pi / 180

        f = self._ellipsoid.Flattening
        R = self._ellipsoid.SemimajorAxis
        je = self._ellipsoid.je
        m = self._ellipsoid.ca

        f2 = -f + 2.5 * m + 0.5 * np.power(f, 2) - 26e0 / 7e0 \
             * f * m + 15e0 / 4e0 * np.power(m, 2)

        f4 = -0.5 * np.power(f, 2) + 2.5 * f * m

        gB = je * (1 + f2 * np.power(np.cos(B), 2) + f4 * np.power(np.cos(2 * B), 4))

        term = 1 + f + m - 2 * f * np.power(np.cos(B), 2)

        g_theta_z = gB * (1 - 2 / R * term * z + (3 / R ** 2) * np.power(z, 2))

        return g_theta_z

    def _getHeight_EF(self, theta, H):
        '''get the geometric height, see Ehsan's JGR, Eq. 16'''

        f = self._ellipsoid.Flattening
        R = self._ellipsoid.SemimajorAxis
        je = self._ellipsoid.je
        m = self._ellipsoid.ca

        r_e_theta = self._getR(theta)

        B = (90 - theta) * np.pi / 180

        f2 = -f + 2.5 * m + 0.5 * np.power(f, 2) - 26e0 / 7e0 \
             * f * m + 15e0 / 4e0 * np.power(m, 2)

        f4 = -0.5 * np.power(f, 2) + 2.5 * f * m

        g_theta = je * (1 + f2 * np.power(np.cos(B), 2) + f4 * np.power(np.cos(2 * B), 4))

        term1 = g_theta * r_e_theta / Constants.g_wmo

        z = r_e_theta * H / (term1 - H)

        return z

    def getGeoid(self):
        """
        get geoid undulation of N * M points
        :param lat: geocentric latitude in degree, [dimension N*M]
        :param lon: geocentric longitude in degree, [dimension N*M]
        :return: one dimension geoid-undulation
        """
        if self.__Geoid is not None:
            return self.__Geoid.copy()
        lat, lon = self._ld.getLatLon_v2()
        geoid = GeoidUndulation(self._ellipsoid.type).getGeoid(lat, lon).flatten()

        return geoid
        # return 0.

    @staticmethod
    def defaultConfig(isWrite=True):
        config = {'MaxDegree': 180,
                  'Method': IntegralChoice.GFZ06VI.name,
                  'Parallel': -1,
                  'Ellipsoid': EllipsoidType.GRS80_IERS2010.name}
        if isWrite:
            with open('../Settings/integral.setting.json', 'w') as f:
                f.write(json.dumps(config))
        return config
