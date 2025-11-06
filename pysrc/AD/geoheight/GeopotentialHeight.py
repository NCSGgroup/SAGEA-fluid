"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/5 10:52
@Description:
"""
import numpy as np

from pysrc.ancillary.load_file.LoadFields import LoadFields, DataType
from pysrc.ancillary.constant.GeoConstant import ADConstant


class GeopotentialHeight:
    """
    Produce the geopotential (height) on each model (half and full) level
    """

    def __init__(self, field: LoadFields):

        self.__field = field
        self.__sp = field.getField(DataType.PSFC)
        self.__lp = field.getPressureLevel()

        self.__p = None
        self.__Z_half = None
        self.__Z_full = None

        pass

    def __get_ph_levs(self, level):
        """
        Return the pressure at a given level and the one at next level
        :param level:
        :param sp: surface pressure
        :return:
        """
        # a_coef = self.__a
        # b_coef = self.__b
        # sp = self.__sp
        #
        # ph_lev = a_coef[level - 1] + (b_coef[level - 1] * sp)
        # ph_levplusone = a_coef[level] + (b_coef[level] *  nb;l-
        if level == (self.__field.getLevel()-1):
            return self.__lp[level], self.__sp
        else:
            return self.__lp[level], self.__lp[level+1]

    def __compute_z_level(self, lev, t_level, q_level, z_h):

        # compute moist (virtual) temperature. Quote from ECMWF
        # t_level = t_level * (1. + 0.609133 * q_level)
        # another format in accordance with AOD RL06 document
        tv_level = t_level * (1. + 0.608000 * q_level)

        # compute the pressures (on half-levels)
        ph_lev, ph_levplusone = self.__get_ph_levs(lev)


        dlog_p = np.log(ph_levplusone / ph_lev)
        if np.all(dlog_p) == 0:
            alpha = np.zeros((len(ph_lev)))
        else:
            alpha = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

        dlog_p[np.abs(ph_levplusone - ph_lev)<1e-2] = 0
        alpha[np.abs(ph_levplusone - ph_lev) < 1e-2] = 0


        tv_level = tv_level * ADConstant.Rd

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the
        # full level
        z_f = z_h + (tv_level * alpha)

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h = z_h + (tv_level * dlog_p)

        return z_f, z_h, ph_levplusone - ph_lev

    def produce_z(self):
        """
        Compute z at half & full level for the given level, based on t/q/sp.

        We want to integrate up into the atmosphere, starting at the ground so we start at the lowest level (highest
        number) and keep accumulating the height as we go. See the IFS documentation, part III For speed and file
        I/O, we perform the computations with numpy vectors instead of fieldsets.

        :return:
        """

        # z_h = self.__field.getField(DataType.PHISFC)*Constants.g_wmo
        z_h = self.__field.getField(DataType.PHISFC)

        z = []
        p = []
        zf = []

        levels = self.__field.getLevel()
        TEMP = self.__field.getField(DataType.TEMP)
        SHUM = self.__field.getField(DataType.SHUM)
        qLev = self.__field.getQLevel()

        z.append(z_h)
        for lev in list(reversed(list(range(0, levels)))):
            if lev - (levels - qLev) >= 0:
                z_f, z_h, dp = self.__compute_z_level(lev, TEMP[lev], SHUM[lev - (levels - qLev)], z_h)
            else:
                z_f, z_h, dp = self.__compute_z_level(lev, TEMP[lev], np.zeros(len(TEMP[lev])), z_h)

            z.append(z_h)
            p.append(dp)
            zf.append(z_f)

        '''Ehsan removes the z[0] that is the surface geo-potential height (it may be incorrect)'''
        # del z[0]
        '''
        Note: the last z_h is useless because there is no half level at pressure zero at all.
        However, it makes sense when z denotes the model level
        '''
        # del z[-1]

        z.reverse()
        p.reverse()
        zf.reverse()

        '''divided by g to get height'''
        # geopotential height on half model level
        ghml = list(map(lambda x: x / ADConstant.g_wmo, z))

        # geopotential height on full model level
        ghfml = list(map(lambda x: x / ADConstant.g_wmo, zf))

        self.__p = p
        self.__Z_half = ghml
        self.__Z_full = ghfml

        pass

    def getZ_half(self):
        """
        geopoetnetial height at half model level
        :return:
        """
        # return self.__Z_half

        return self.__Z_full

    def getZ_full(self):
        """
        geopoetnetial height at full model level
        :return:
        """
        return self.__Z_full

    def getP(self):
        """
        pressure increment at each level
        :return:
        """
        return self.__p
