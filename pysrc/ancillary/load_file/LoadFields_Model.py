"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2022/3/31
@Description:
"""

"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/3 17:44
@Description:
"""

import json
import os
# import pygrib
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

# from GeoMathKit import GeoMathKit
# from LoadSH_AOD import AOD_GFZ, AODtype
from pysrc.ancillary.constant.Setting import DataType, ForceFields
# import Ngl, Nio
import cfgrib
import xarray

# import netCDF4 as nc

class LoadFields:
    """
    This class deals with loading all necessary inputs for one cycle of AOD computation.
    """
    TimeEpoch = ["00:00:00","03:00:00" ,"06:00:00","09:00:00","12:00:00","15:00:00","18:00:00", "21:00:00"]

    def __init__(self, data_path='I:/ERA5/model level/',GeoHeight='model'):
        self.__epoch_data = {}
        self.__meanfield = None
        self.__modelLevel = None
        self.__path = data_path
        self.__lat, self.__lon = None, None
        self.__nLat, self.__nLon = None, None
        self.__level = None
        self.__q_level = None
        self.a = np.load('I:\Paper3\Auxiliary/ABCoefficients137.npy')[0]
        self.b = np.load('I:\Paper3\Auxiliary/ABCoefficients137.npy')[1]
        self.GeoHeight = GeoHeight

        pass

    def setTime(self, date='2020-01-01', time='00:00:00', OnlyPressure=False):
        """
        set the time epoch of data that will be used in later computation.
        :param OnlyPressure: if True, then only pressure is read.
        :param date:
        :param time:
        :return:
        """

        assert time in self.TimeEpoch
        str1 = date.split('-')
        str2 = time.split(':')

        sstr1 = str1[0] + str1[1] + str1[2]
        sstr2 = sstr1 + str2[0]


        path_dir = os.path.join(self.__path, str1[0], sstr1)

        flag_sp = 'sp-' + sstr2 + '.nc'
        flag_t = 'TEM-' + sstr2 + '.nc'
        flag_q = 'SHM-' + sstr2 + '.nc'
        flag_z = 'GPT-' + str1[0] + '.nc'


        '''name of each data'''
        PHISFC = os.path.join(self.__path, flag_z)
        PSFC = os.path.join(path_dir, flag_sp)
        TEMP = os.path.join(path_dir, flag_t)
        SHUM = os.path.join(path_dir, flag_q)

        if os.path.exists(PHISFC):
            f1 = xarray.open_dataset(PHISFC)
            lat0, lon0 = f1['latitude'].values, f1['longitude'].values
            phisfc = f1['z'].values[0].flatten()
            self.__epoch_data[DataType.PHISFC] = phisfc


        elif os.path.exists(os.path.join(self.__path,'GPT-' + str1[0] + '.grib')):
            flag_z = 'GPT-' + str1[0] + '.grib'
            PHISFC = os.path.join(self.__path, flag_z)
            f1 = cfgrib.open_dataset(PHISFC)
            lat0, lon0 = f1['latitude'].values, f1['longitude'].values
            phisfc = f1['z'].values.flatten()
            self.__epoch_data[DataType.PHISFC] = phisfc

        elif os.path.exists(os.path.join(self.__path, 'GPT.nc')):
            PHISFC = os.path.join(self.__path, 'GPT.nc')
            f1 = xarray.open_dataset(PHISFC)
            lat0, lon0 = f1['latitude'].values, f1['longitude'].values
            phisfc = f1['z'].values[0].flatten()
            self.__epoch_data[DataType.PHISFC] = phisfc
        elif os.path.exists(os.path.join(self.__path, 'GPT.grib')):
            PHISFC = os.path.join(self.__path, 'GPT.grib')
            f1 = cfgrib.open_dataset(PHISFC)
            lat0, lon0 = f1['latitude'].values, f1['longitude'].values
            phisfc = f1['z'].values[0].flatten()
            self.__epoch_data[DataType.PHISFC] = phisfc
        else:
            base_path = self.__path.rsplit('/',2)[0]+'/'
            if '0.25' in self.__path:
                new_path = os.path.join(base_path,'model level 0.25')
            else:
                new_path = os.path.join(base_path,'model level')
            if os.path.exists(os.path.join(new_path,'GPT.nc')):
                PHISFC = os.path.join(new_path, 'GPT.nc')
            else:
                PHISFC = os.path.join(new_path, 'GPT.grib')
            f1 = cfgrib.open_dataset(PHISFC)
            lat0, lon0 = f1['latitude'].values, f1['longitude'].values
            phisfc = f1['z'].values[0].flatten()
            self.__epoch_data[DataType.PHISFC] = phisfc



        if os.path.exists(PSFC):
            f = xarray.open_dataset(PSFC)
            lat1, lon1 = f['latitude'].values, f['longitude'].values
            ps = f['sp'].values[0].flatten()
            self.__epoch_data[DataType.PSFC] = ps

        elif os.path.exists(os.path.join(path_dir,'sp-' + sstr2 + '.grib')):
            flag_sp = 'sp-' + sstr2 + '.grib'
            PSFC = os.path.join(path_dir, flag_sp)
            f = cfgrib.open_dataset(PSFC)
            lat1, lon1 = f['latitude'].values, f['longitude'].values
            ps = f['sp'].values.flatten()
            self.__epoch_data[DataType.PSFC] = ps
        else:
            base_path = self.__path.rsplit('/',2)[0]+'/'
            if '0.25' in self.__path:
                new_path = os.path.join(base_path, 'model level 0.25')
                path_dir = os.path.join(new_path, str1[0], sstr1)
            else:
                new_path = os.path.join(base_path, 'model level')
                path_dir = os.path.join(new_path, str1[0], sstr1)

            if os.path.exists(os.path.join(path_dir, flag_sp)):
                PSFC = os.path.join(new_path, flag_sp)
                f = xarray.open_dataset(PSFC)
                lat1, lon1 = f['latitude'].values, f['longitude'].values
                ps = f['sp'].values[0].flatten()
                self.__epoch_data[DataType.PSFC] = ps
            else:
                flag_sp = 'sp-' + sstr2 + '.grib'
                PSFC = os.path.join(path_dir, flag_sp)
                f = cfgrib.open_dataset(PSFC)
                lat1, lon1 = f['latitude'].values, f['longitude'].values
                ps = f['sp'].values.flatten()
                self.__epoch_data[DataType.PSFC] = ps



        if OnlyPressure:
            assert (lat0 == lat1).all()
            assert (lon0 == lon1).all()
            self.__lat, self.__lon = lat0, lon0
            print(PHISFC)
            print(PSFC)
            return self
        if self.GeoHeight in ['pressure','pressure level','Pressure']:
            if os.path.exists(TEMP):
                ft = xarray.open_dataset(TEMP)
                lat2, lon2 = ft['latitude'].values, ft['longitude'].values
                if 'level' in ft.variables:
                    TEMP_level = ft['level'].values * 100
                elif 'pressure_level' in ft.variables:
                    TEMP_level = ft['pressure_level'].values[::-1]*100
                data_TEMP = []
                for i in range(len(ft['t'][0, :, 0, 0])):
                    data_TEMP.append(ft['t'].values[0, i].flatten())
                data_TEMP = np.array(data_TEMP[::-1])
                self.__epoch_data[DataType.TEMP] = data_TEMP
            else:
                flag_t = 'TEM-' + sstr2 + '.grib'
                TEMP = os.path.join(path_dir, flag_t)
                ft = cfgrib.open_dataset(TEMP)
                lat2, lon2 = ft['latitude'].values, ft['longitude'].values
                # TEMP_level = ft['isobaricInhPa'].values[::-1] * 100
                TEMP_level = ft['isobaricInhPa'].values[::-1] * 100
                data_TEMP = []
                for i in range(len(ft['t'][:, 0, 0])):
                    data_TEMP.append(ft['t'].values[i].flatten())
                data_TEMP = np.array(data_TEMP[::-1])
                self.__epoch_data[DataType.TEMP] = data_TEMP

            if os.path.exists(SHUM):
                fs = xarray.open_dataset(SHUM)
                lat3, lon3 = fs['latitude'].values, fs['longitude'].values
                if 'level' in fs.variables:
                    SHUM_level = fs['level'].values * 100
                elif 'pressure_level' in fs.variables:
                    SHUM_level = fs['pressure_level'].values[::-1]*100
                data_SHUM = []
                for i in range(len(fs['q'][0, :, 0, 0])):
                    data_SHUM.append(fs['q'].values[0, i].flatten())
                data_SHUM = np.array(data_SHUM[::-1])
                self.__epoch_data[DataType.SHUM] = data_SHUM
            else:
                flag_q = 'SHM-' + sstr2 + '.grib'
                SHUM = os.path.join(path_dir, flag_q)
                ft = cfgrib.open_dataset(SHUM)
                lat3, lon3 = ft['latitude'].values, ft['longitude'].values
                SHUM_level = ft['isobaricInhPa'].values[::-1] * 100
                # SHUM_level = ft['isobaricInhPa'].values
                date_SHUM = []
                for i in range(len(ft['q'][:, 0, 0])):
                    date_SHUM.append(ft['q'].values[i].flatten())
                data_SHUM = np.array(date_SHUM[::-1])
                self.__epoch_data[DataType.SHUM] = data_SHUM

            assert (lat0 == lat1).all() and (lat0 == lat2).all() and (lat0 == lat3).all()
            assert (lon0 == lon1).all() and (lon0 == lon2).all() and (lon0 == lon3).all()

            iso_pres = []

            for i in range(len(TEMP_level)):
                pres_level = np.ones(len(ps)) * TEMP_level[i]
                if (TEMP_level[i] - ps < 0).all():
                    iso_pres.append(pres_level)
                    continue
                index = TEMP_level[i] - ps > 0
                pres_level[index] = ps[index]
                iso_pres.append(pres_level)

        elif self.GeoHeight in ['model','model level','Model']:
            if os.path.exists(TEMP):
                ft = xarray.open_dataset(TEMP)
                lat2, lon2 = ft['latitude'].values, ft['longitude'].values
                if 'level' in ft.variables:
                    TEMP_level = ft['level'].values
                elif 'model_level' in ft.variables:
                    TEMP_level = ft['model_level'].values
                data_TEMP = []
                for i in range(len(ft['t'][0,:,0,0])):
                    data_TEMP.append(ft['t'].values[0,i].flatten())
                data_TEMP = np.array(data_TEMP)
                self.__epoch_data[DataType.TEMP] = data_TEMP

            else:
                flag_t = 'TEM-' + sstr2 + '.grib'
                TEMP = os.path.join(path_dir, flag_t)
                ft = cfgrib.open_dataset(TEMP)
                lat2, lon2 = ft['latitude'].values, ft['longitude'].values
                data_TEMP = []
                TEMP_level = ft['hybrid'].values
                for i in range(len(ft['t'][:, 0, 0])):
                    data_TEMP.append(ft['t'].values[i].flatten())
                data_TEMP = np.array(data_TEMP)
                self.__epoch_data[DataType.TEMP] = data_TEMP

            if os.path.exists(SHUM):
                fs = xarray.open_dataset(SHUM)
                lat3, lon3 = fs['latitude'].values, fs['longitude'].values
                if 'level' in fs.variables:
                    SHUM_level = fs['level'].values
                elif 'model_level' in fs.variables:
                    SHUM_level = fs['model_level'].values
                data_SHUM = []
                for i in range(len(fs['q'][0,:,0,0])):
                    data_SHUM.append(fs['q'].values[0,i].flatten())
                data_SHUM = np.array(data_SHUM)
                self.__epoch_data[DataType.SHUM] = data_SHUM

            else:
                flag_q = 'SHM-' + sstr2 + '.grib'
                SHUM = os.path.join(path_dir, flag_q)
                fs = cfgrib.open_dataset(SHUM)
                lat3, lon3 = fs['latitude'].values, fs['longitude'].values
                SHUM_level = fs['hybrid']
                data_SHUM = []
                for i in range(len(fs['q'][:, 0, 0])):
                    data_SHUM.append(fs['q'].values[i].flatten())
                data_SHUM = np.array(data_SHUM)


                self.__epoch_data[DataType.SHUM] = data_SHUM

            assert (lat0 == lat1).all() and (lat0 == lat2).all() and (lat0 == lat3).all()
            assert (lon0 == lon1).all() and (lon0 == lon2).all() and (lon0 == lon3).all()

            iso_pres = []
            for i in range(len(TEMP_level)):
                pres_level = self.a[i]+self.b[i]*ps
                iso_pres.append(pres_level)
            pass

        self.__level = len(TEMP_level)
        self.__q_level = len(SHUM_level)
        self.__epoch_data['PsLevel'] = np.array(iso_pres)
        self.__lat, self.__lon = lat0, lon0
        print(PHISFC)
        print(PSFC)
        print(TEMP)
        print(SHUM)

        # # this is to fix the top and bottom layer
        # top_temp = self.__epoch_data[DataType.TEMP][:1]
        # top_shum = self.__epoch_data[DataType.SHUM][:1]
        # top_ps = self.__epoch_data['PsLevel'][:1]
        # bot_temp = self.__epoch_data[DataType.TEMP][-1:]
        # bot_shum = self.__epoch_data[DataType.SHUM][-1:]
        # bot_ps = self.__epoch_data['PsLevel'][-1:]
        #
        # #
        # middle_tem = self.__epoch_data[DataType.TEMP][1:-1]
        # middle_shum = self.__epoch_data[DataType.SHUM][1:-1]
        # middle_ps = self.__epoch_data['PsLevel'][1:-1]
        # print(f"top:{top_shum.shape}")
        # print(f"bottom:{bot_shum.shape}")
        # print(f"middle:{middle_shum.shape}")
        #
        # self.__epoch_data[DataType.TEMP] = np.vstack((top_temp, bot_temp))
        # self.__epoch_data[DataType.SHUM] = np.vstack((top_shum, bot_shum))
        # self.__epoch_data['PsLevel'] = np.vstack((top_ps, bot_ps))
        # self.__level = len(self.__epoch_data[DataType.TEMP])
        # self.__q_level = len(self.__epoch_data[DataType.SHUM])
        # # #
        # slice = 3
        # # # 'This is to sample to layers 135/slice+2'
        # #
        # self.__epoch_data[DataType.TEMP] = np.vstack((top_temp,middle_tem[::slice],bot_temp))
        # self.__epoch_data[DataType.SHUM] = np.vstack((top_shum,middle_shum[::slice],bot_shum))
        # self.__epoch_data['PsLevel'] = np.vstack((top_ps,middle_ps[::slice],bot_ps))
        # self.__level = len(self.__epoch_data[DataType.TEMP])
        # self.__q_level = len(self.__epoch_data[DataType.SHUM])



        #'This is to sample to layer 137-135/slice'
        # index = np.arange(0,len(middle_tem),slice)
        # reverse_index = np.sort(index)[::-1]
        # mid_tem = np.delete(middle_tem,reverse_index,axis=0)
        # mid_shum = np.delete(middle_shum,reverse_index,axis=0)
        # mid_ps = np.delete(middle_ps,reverse_index,axis=0)
        # # print(index)
        # print(reverse_index)
        # print(np.shape(middle_tem))
        # print(np.shape(mid_tem))

        # self.__epoch_data[DataType.TEMP] = np.vstack((top_temp,mid_tem,bot_temp))
        # self.__epoch_data[DataType.SHUM] = np.vstack((top_shum,mid_shum,bot_shum))
        # self.__epoch_data['PsLevel'] = np.vstack((top_ps,mid_ps,bot_ps))
        # self.__level = len(self.__epoch_data[DataType.TEMP])
        # self.__q_level = len(self.__epoch_data[DataType.SHUM])


        # print(self.__level)
        # print(self.__q_level)





        return self

    def setTime_for_tide(self, date='2010-01-01', time='06:00:00'):
        """
        set the time epoch of data that will be used in later computation.
        :param OnlyPressure: if True, then only pressure is read.
        :param date:
        :param time:
        :return:
        """

        assert time in self.TimeEpoch
        str1 = date.split('-')
        str2 = time.split(':')

        sstr1 = str1[0] + str1[1] + str1[2]
        sstr2 = sstr1 + str2[0]


        path_dir = os.path.join(self.__path, str1[0], sstr1)

        flag_sp = 'sp-'+sstr2+'.nc'
        PSFC = os.path.join(path_dir, flag_sp)

        if os.path.exists(PSFC):

            f = xarray.open_dataset(PSFC)
            self.__lat,self.__lon = f['latitude'].values,f['longitude'].values
            ps = f['sp'].values[0]
            self.__epoch_data[DataType.PSFC] = ps

        else:

            flag_sp = 'sp-'+sstr2 + '.grib'
            '''name of each data'''
            PSFC = os.path.join(path_dir, flag_sp)
            f = cfgrib.open_dataset(PSFC)
            self.__lat,self.__lon = f['latitude'].values,f['longitude'].values
            ps = f['sp'].values
            self.__epoch_data[DataType.PSFC] = ps

        # print(PSFC)

        return self

    def getField(self, kind: DataType = DataType.TEMP):
        return self.__epoch_data[kind]

    def getPressureLevel(self):
        return self.__epoch_data['PsLevel']

    def getLevel(self):
        return self.__level

    def getQLevel(self):
        return self.__q_level

    def getLatLon(self):
        """
        position of N*M points
        :return: geodetic latitude [dimension: N*M] and longitude [dimension: N*M] in degree
        """

        lon, lat = np.meshgrid(self.__lon, self.__lat)

        return lat.flatten(), lon.flatten()

    def getLatLon_v2(self):
        """
        position of N*M points
        :return: geodetic latitude [dimension: N] and longitude [dimension: M] in degree
        """
        return self.__lat, self.__lon

    def getNlatNlon(self):
        """
        position of N*M points
        :return: N, M
        """
        return len(self.__lat), len(self.__lon)


def demo1():
    ld = LoadFields()
    ld.setTime_for_tide()


def demo2():
    ld = LoadFields(data_path='H:/ERA5/pressure level/',GeoHeight='pressure level')
    ld.setTime(date='2022-01-01',time='12:00:00',OnlyPressure=False)

    data = ld.getField(kind=DataType.PHISFC)
    print(data.shape)

def demo3():
    ld = LoadFields(data_path='H:/ERA5/model level/',GeoHeight='model level')
    ld.setTime(date='2010-01-01', time='12:00:00', OnlyPressure=False)
    lat = ld.getLatLon_v2()[0]
    # print((ld.a))




if __name__ == '__main__':
    demo3()