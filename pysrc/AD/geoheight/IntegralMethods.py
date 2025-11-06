"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/8 10:45
@Description:
"""

import time as ti
from multiprocessing.pool import Pool

import numpy as np

from pysrc.AD.geoheight.InnerIntegral import InnerIntegral, GeopotentialHeight, LoadFields
from pysrc.ancillary.constant.Setting import InterpOption, IntegralChoice, Constants


# from pysrc.Smooth import Smooth


class SphereSP(InnerIntegral):

    def __init__(self, gh: GeopotentialHeight, ld: LoadFields):
        InnerIntegral.__init__(self, gh, ld)
        self.__lev = len(gh.getP())
        pass

    def deltaI(self):
        # AOD RL05
        lev = self.__lev
        pml = self._gh.getP()

        # calculate the surface pressure
        sp = pml[0]
        for i in range(lev - 1):
            sp = sp + pml[i + 1]

        deltaI = [sp / Constants.g_wmo]

        # print(deltaI[0][17719])

        return deltaI


class NonSphereSP(InnerIntegral):

    def __init__(self, gh: GeopotentialHeight, ld: LoadFields):
        InnerIntegral.__init__(self, gh, ld)
        self.__lev = len(gh.getP())
        pass

    def deltaI(self):

        # AOD RL05
        lev = self.__lev
        pml = self._gh.getP()
        zml = self._gh.getZ_half()
        lat, lon = self._ld.getLatLon()

        # calculate the surface pressure
        sp = pml[0]

        for i in range(lev - 1):
            sp = sp + pml[i + 1]

        z = self._getHeight(lat, zml[-1])
        r = self._getR(lat) + self.getGeoid() + z

        ar = r / self._ellipsoid.SemimajorAxis

        deltaI = []

        iniPower = ar
        for i in range(self._maxDeg + 1):
            I_lev = np.zeros(np.size(ar))
            iniPower = iniPower * ar
            # I_lev = np.power(ar, i + 2) * sp / self._getG(lat, z)
            I_lev = iniPower * sp / self._getG(lat, z)
            deltaI.append(I_lev)

            # self.printInfo(i)

        return deltaI


class GFZ06VI(InnerIntegral):

    def __init__(self, gh: GeopotentialHeight, ld: LoadFields):
        InnerIntegral.__init__(self, gh, ld)
        self.__lev = len(gh.getP())
        pass

    def deltaI(self):
        lev = self.__lev
        pml = self._gh.getP()
        zml = self._gh.getZ_half()
        lat, lon = self._ld.getLatLon()

        deltaI = []

        r0 = self._getR(lat) + self.getGeoid()

        # term1 = []
        # term2 = []
        # for j in range(lev):
        #     z = self._getHeight(lat, zml[j])
        #
        #     r = r0 + z
        #
        #     ar = r / self._ellipsoid.SemimajorAxis
        #
        #     term1.append(ar)
        #     term2.append(pml[j] / self._getG(lat, z))
        #
        # term1 = np.array(term1)
        # term2 = np.array(term2)

        z0 = self._getHeight(lat, zml)
        r1 = r0 + z0
        term1 = r1 / self._ellipsoid.SemimajorAxis
        term2 = pml / self._getG(lat, z0)

        if self._processors <= 1:

            '''No parallel'''
            iniPower = term1
            for i in range(self._maxDeg + 1):
                # I_lev = np.zeros(np.size(r0))
                iniPower = iniPower * term1

                # for j in range(lev):
                #     I_lev = I_lev + iniPower[j] * term2[j]

                # I_lev = np.sum(iniPower*term2,0)
                I_lev = np.einsum('ij,ij->j', iniPower, term2, optimize='greedy')

                deltaI.append(I_lev)
                pass

            pass


        else:
            '''Parallel computing'''
            term1list = [term1 for i in range(self._maxDeg + 1)]
            term2list = [term2 for i in range(self._maxDeg + 1)]
            pool = Pool(self._processors)
            ret = pool.map(self.func, list(zip(term1list, term2list, range(self._maxDeg + 1))))
            pool.close()
            pool.join()
            ret.sort(key=lambda x: x[1])
            deltaI = [x[0] for x in ret]

        return deltaI

    def deltaI2(self):
        lev = self.__lev
        pml = self._gh.getP()
        zml = self._gh.getZ_half()
        lat, lon = self._ld.getLatLon()

        deltaI = []

        r0 = self._getR(lat) + self.getGeoid()

        # term1 = []
        # term2 = []
        # for j in range(lev):
        #     z = self._getHeight(lat, zml[j])
        #
        #     r = r0 + z
        #
        #     ar = r / self._ellipsoid.SemimajorAxis
        #
        #     term1.append(ar)
        #     term2.append(pml[j] / self._getG(lat, z))

        # term1 = np.array(term1)
        # term2 = np.array(term2)

        z0 = self._getHeight(lat, zml)
        r1 = r0 + z0
        adr = r1 / self._ellipsoid.SemimajorAxis
        pdg = pml / self._getG(lat, z0)

        return adr, pdg

    @staticmethod
    def func(Val):
        term1 = Val[0]
        term2 = Val[1]
        index = Val[2]

        lev = len(term1)
        I_lev = np.zeros(np.size(term1[0]))

        for j in range(lev):
            I_lev = I_lev + np.power(term1[j], index + 2) * term2[j]

        return I_lev, index


class GFZ04VI(InnerIntegral):
    def __init__(self, gh: GeopotentialHeight, ld: LoadFields):
        InnerIntegral.__init__(self, gh, ld)
        self.__lev = len(gh.getP())
        pass

    def deltaI(self):
        """
        With the assumption of spherical Earth
        :return:
        """

        lev = self.__lev
        pml = self._gh.getP()
        zml = self._gh.getZ_half()
        lat, lon = self._ld.getLatLon()

        deltaI = []

        ar = []
        geoid = self.getGeoid()
        R = self._ellipsoid.SemimajorAxis
        for j in range(lev):
            ar.append(R / (R - zml[j]) + geoid / R)

        ar = np.array(ar)
        iniPower = np.power(ar, 3)

        if self._processors <= 1:
            '''No parallel'''
            for i in range(self._maxDeg + 1):

                I_lev = np.zeros(np.size(lat))
                iniPower = iniPower * ar

                for j in range(lev):
                    I_lev = I_lev + iniPower[j] * pml[j] / Constants.g_wmo

                deltaI.append(I_lev)
        else:
            '''Parallel computing'''
            term1list = [ar for i in range(self._maxDeg + 1)]
            term2list = [pml for i in range(self._maxDeg + 1)]
            pool = Pool(self._processors)
            ret = pool.map(self.func, list(zip(term1list, term2list, range(self._maxDeg + 1))))
            pool.close()
            pool.join()
            ret.sort(key=lambda x: x[1])
            deltaI = [x[0] for x in ret]

        return deltaI

    @staticmethod
    def func(Val):
        term1 = Val[0]
        term2 = Val[1]
        index = Val[2]

        lev = len(term1)
        I_lev = np.zeros(np.size(term1[0]))

        for j in range(lev):
            I_lev = I_lev + np.power(term1[j], index + 4) * term2[j] / Constants.g_wmo

        return I_lev, index


class GFZ05VI(InnerIntegral):
    def __init__(self, gh: GeopotentialHeight, ld: LoadFields):
        InnerIntegral.__init__(self, gh, ld)
        self.__lev = len(gh.getP())
        pass

    def deltaI(self):
        lev = self.__lev
        pml = self._gh.getP()
        zml = self._gh.getZ_half()
        lat, lon = self._ld.getLatLon()

        deltaI = []

        gs = Constants.g_wmo
        '''See Eq. (12) in paper: Comparisons of atmospheric data and reduction methods for the
        analysis of satellite gravimetry observations'''
        gtheta = self._ellipsoid.je + (self._ellipsoid.jb - self._ellipsoid.je) * np.power(np.sin(lat * np.pi / 180), 2)

        ar = []
        PdivideG = []
        geoid = self.getGeoid()
        R = self._ellipsoid.SemimajorAxis

        for j in range(lev):
            ar.append(R / (R - zml[j] * gs / gtheta) + geoid / R)
            # PdivideG.append(self.pml[j] / gtheta)
            PdivideG.append(pml[j] / self._ellipsoid.je)

        ar = np.array(ar)
        iniPower = np.power(ar, 3)

        if self._processors <= 1:
            '''No parallel'''

            for i in range(self._maxDeg + 1):

                I_lev = np.zeros(np.size(lat))
                iniPower = iniPower * ar

                for j in range(lev):
                    I_lev = I_lev + iniPower[j] * PdivideG[j]

                deltaI.append(I_lev)

        else:
            '''Parallel computing'''
            term1list = [ar for i in range(self._maxDeg + 1)]
            term2list = [PdivideG for i in range(self._maxDeg + 1)]
            pool = Pool(self._processors)
            ret = pool.map(self.func, list(zip(term1list, term2list, range(self._maxDeg + 1))))
            pool.close()
            pool.join()
            ret.sort(key=lambda x: x[1])
            deltaI = [x[0] for x in ret]

        return deltaI

    @staticmethod
    def func(Val):
        term1 = Val[0]
        term2 = Val[1]
        index = Val[2]

        lev = len(term1)
        I_lev = np.zeros(np.size(term1[0]))

        for j in range(lev):
            I_lev = I_lev + np.power(term1[j], index + 4) * term2[j]

        return I_lev, index


class EFVI(InnerIntegral):
    """
    method proposed by Ehsan
    """

    def __init__(self, gh: GeopotentialHeight, ld: LoadFields):
        InnerIntegral.__init__(self, gh, ld)
        self.__lev = len(gh.getP())
        pass

    def deltaI(self):
        lev = self.__lev
        pml = self._gh.getP()
        zml = self._gh.getZ_half()
        lat, lon = self._ld.getLatLon()

        deltaI = []

        r0 = self._getR(lat) + self.getGeoid()

        term1 = []
        term2 = []

        for j in range(lev):
            z = self._getHeight_EF(lat, zml[j])

            r = r0 + z

            ar = r / self._ellipsoid.SemimajorAxis

            term1.append(ar)
            term2.append(pml[j] / self._getG_EF(lat, z))

        term1 = np.array(term1)
        term2 = np.array(term2)

        if self._processors <= 1:
            '''No parallel'''

            iniPower = term1

            for i in range(self._maxDeg + 1):
                I_lev = np.zeros(np.size(r0))

                iniPower = iniPower * term1
                # I_lev = np.sum(iniPower * term2, 0)

                for j in range(lev):
                    I_lev = I_lev + iniPower[j] * term2[j]

                deltaI.append(I_lev)

        else:
            '''Parallel computing'''
            term1list = [term1 for i in range(self._maxDeg + 1)]
            term2list = [term2 for i in range(self._maxDeg + 1)]
            pool = Pool(self._processors)
            ret = pool.map(self.func, list(zip(term1list, term2list, range(self._maxDeg + 1))))
            pool.close()
            pool.join()
            ret.sort(key=lambda x: x[1])
            deltaI = [x[0] for x in ret]

        return deltaI

    @staticmethod
    def func(Val):
        term1 = Val[0]
        term2 = Val[1]
        index = Val[2]

        lev = len(term1)
        I_lev = np.zeros(np.size(term1[0]))

        for j in range(lev):
            I_lev = I_lev + np.power(term1[j], index + 2) * term2[j]

        return I_lev, index


def innerIn(config: dict, gh: GeopotentialHeight, ld: LoadFields):
    """
    Main function to call method defined in the Config.
    :param config:
    :param gh:
    :param ld:
    :return:
    """
    name = IntegralChoice[config['Method']]

    cls = None

    if name == IntegralChoice.EFVI:
        cls = EFVI(gh, ld)
    elif name == IntegralChoice.GFZ05VI:
        cls = GFZ05VI(gh, ld)
    elif name == IntegralChoice.GFZ06VI:
        cls = GFZ06VI(gh, ld)
    elif name == IntegralChoice.GFZ04VI:
        cls = GFZ04VI(gh, ld)
    elif name == IntegralChoice.SphereSP:
        cls = SphereSP(gh, ld)
    elif name == IntegralChoice.NonsphereSP:
        cls = NonSphereSP(gh, ld)

    cls.configure(config)

    return cls

# def demo1():
#     lf = LoadFields().configure(LoadFields.defaultConfig(isWrite=False))
#     lf.setTime(date='2001-01-01', time='00:00:00')
#
#     ss = Smooth(lf)
#     ss.setMethod(method=InterpOption.Bilinear).interpAll()
#
#     gh = GeopotentialHeight(lf)
#     gh.produce_z()
#
#     # ii = NonSphereSP(gh, lf)
#     # ii = GFZ06VI(gh, lf)
#     # ii = GFZ04VI(gh, lf)
#     # ii = GFZ05VI(gh, lf)
#     # ii = EFVI(gh, lf)
#
#     begin = ti.time()
#     # deltaI = ii.setMaxDeg(120).setParallel(3).deltaI()
#     deltaI = innerIn(InnerIntegral.defaultConfig(True), gh, lf)
#     print("COST TIME: %s " % ((ti.time() - begin) * 1000))
#
#     pass


# def demo2():
#     lf = LoadFields().configure(LoadFields.defaultConfig(isWrite=False))
#     lf.setTime(date='2001-01-01', time='00:00:00')
#
#     ss = Smooth(lf)
#     ss.setMethod(method=InterpOption.Bilinear).interpAll()
#
#     gh = GeopotentialHeight(lf)
#     gh.produce_z()
#
#     # ii = NonSphereSP(gh, lf)
#     # ii = GFZ06VI(gh, lf)
#     # ii = GFZ04VI(gh, lf)
#     # ii = GFZ05VI(gh, lf)
#     ii = EFVI(gh, lf)
#
#     begin = ti.time()
#     deltaI = ii.setMaxDeg(10).setParallel(-1).deltaI()
#     print("COST TIME: %s " % ((ti.time() - begin) * 1000))
#
#
# if __name__ == '__main__':
#     demo1()
