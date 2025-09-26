import os
import numpy as np
import pandas as pd
from pysrc.ancillary.constant.Setting import EAMtype


class StorageEOP:
    def __init__(self):
        self.__fileDir = None
        self.__fileFullPath = None
        self.__EAM = None
        self.portionType = 'AAM'
        self.source = 'surface pressure'
        self.time_year = None
        self.HeadName = None
        self.mean_EOP = {}

    def setSource_Information(self,source='ERA5 surface pressure'):
        self.source = source
        return self

    def setRootDir(self,fileDir):
        self.__fileDir = fileDir

        return self

    def setMeanFiled_Information(self,begin_date,end_date,mean_EOP:dict):
        self.begin_mean = begin_date
        self.end_mean = end_date
        self.mean_EOP = mean_EOP
        return self

    def setEAM_Information(self,EAM:list,type=EAMtype.AAM):
        self.__EAM = EAM
        self.EAM_len = len(EAM)

        self.time_year = EAM[0]["YYYY"]

        if type == EAMtype.AAM:
            self.portionType = 'AAM'
            self.EAM_len = 6*self.EAM_len
            self.HeadName = 'Atmospheric Angular Momentum'
        elif type == EAMtype.OAM:
            self.portionType = 'OAM'
            self.EAM_len = 6 * self.EAM_len
            self.HeadName = 'Oceanic Angular Momentum'
        elif type == EAMtype.HAM:
            self.portionType = 'HAM'
            self.EAM_len = 3 * self.EAM_len
            self.HeadName = 'Hydrological Angular Momentum'
        elif type == EAMtype.SLAM:
            self.portionType = 'SLAM'
            self.EAM_len = 3 * self.EAM_len
            self.HeadName = 'Sea-level Angular Momentum'

        if not os.path.exists(self.__fileDir):
            os.makedirs(self.__fileDir)

        self.__fileFullPath = self.__fileDir+os.sep+self.portionType+'_daily_'+self.time_year+'.asc'
        return self
    def EOPstyle_ByProduct(self):
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.__fileFullPath, 'w') as file:
            file.write(f'Effective {self.HeadName} Functions ({self.portionType})\n\n')

            file.write('%-31s%3s%-31s \n' % ('Producer agency  ', ': ', 'HUST'))
            file.write('%-31s%3s%-31s \n' % ('File units', ': ', 'dimensionless earth rotation excitation functions'))
            file.write('%-31s%3s%-31s \n' % ('Issue date ', ': ', f'{now}'))
            file.write('%-31s%3s%-31s \n\n' % ('Version number', ': ', 'V1.0'))

            file.write('%-31s%3s%-31s \n' % ('Contact', ': ', 'Weihang Zhang: zwh_cge@hust.edu.cn & Fan Yang: yfan@aau.aak'))


            file.write('%-31s%3s%-31s \n' % ('Temporal domain', ': ', f'{self.time_year}-01-01 UTC until {self.time_year}-12-31 UTC'))
            file.write('%-31s%3s%-31s \n' % ('Temporal resolution', ': ', 'daily'))
            file.write('%-31s%3s%-31s \n' % ('Number of data records', ': ', f'{self.EAM_len}'))
            file.write('%-31s%3s%-31s \n\n' % ('Subtracted mean range', ': ', 'None'))

            file.write('%-31s%3s%-31s \n' % ('Data source', ': ', f'{self.source}'))
            file.write('%-31s%3s%-31s \n' % ('Horizontal domain', ': ', 'global'))
            file.write('%-31s%3s%-31s \n' % ('Horizontal resolution', ': ', '0.5 degree (regular grid)'))
            file.write('%-31s%3s%-31s \n\n' % ('Correction', ': ', 'inverted barometer correction over ocean applied, surface topography effect considered'))

            file.write('%-31s%3s%-31s \n' % ('Data assimilated', ': ', 'None'))

            file.write(5*'------------------------------------'+'\n')

            file.write('%-5s %-5s %-5s %-5s %-10s %20s %20s %35s %35s %20s %20s \n' %
                       ('YYYY', 'MM','DD', 'HH','MJD',' ',f'{self.portionType} mass term',' ',' ',f'{self.portionType} motion term',' '))
            file.write('%-10s %-10s %-10s %-10s %-10s %-25s %-25s %-24s %-24s %-20s %-20s \n\n' %
                       (' ', ' ', ' ', ' ', ' ', 'x-component ','y-component','z-component ', 'x-component ','y-component','z-component'))
            self._mainContent(EAM=self.__EAM,file=file)

        print(f'Finished {self.portionType} in {self.time_year} saving!\n'
              f'Save path is:{self.__fileFullPath}\n'
              f'===========================================================================')

    def EOPstyle_DeMean(self):
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.__fileFullPath, 'w') as file:
            file.write(f'Effective {self.HeadName} Functions ({self.portionType})\n\n')

            file.write('%-31s%3s%-31s \n' % ('Producer agency  ', ': ', 'HUST'))
            file.write('%-31s%3s%-31s \n' % ('File units', ': ', 'dimensionless earth rotation excitation functions'))
            file.write('%-31s%3s%-31s \n' % ('Issue date ', ': ', f'{now}'))
            file.write('%-31s%3s%-31s \n\n' % ('Version number', ': ', 'V1.0'))

            file.write('%-31s%3s%-31s \n' % ('Contact', ': ', 'Weihang Zhang: zwh_cge@hust.edu.cn & Fan Yang: yfan@aau.aak'))


            file.write('%-31s%3s%-31s \n' % ('Temporal domain', ': ', f'{self.time_year}-01-01 UTC until {self.time_year}-12-31 UTC'))
            file.write('%-31s%3s%-31s \n' % ('Temporal resolution', ': ', 'daily'))
            file.write('%-31s%3s%-31s \n' % ('Number of data records', ': ', f'{self.EAM_len}'))
            file.write('%-31s%3s%-31s \n' % ('Subtracted mean range', ': ', f'mean from {self.begin_mean} to {self.end_mean}'))
            file.write('%-31s%3s  %-20.10e %-20.10e %-20.10e %-20.10e %-20.10e %-20.10e \n\n' % ('Mean values', ': ',
                                                                                            self.mean_EOP['mass_chi1'],self.mean_EOP['mass_chi2'],self.mean_EOP['mass_chi3'],
                                                                                            self.mean_EOP['motion_chi1'],self.mean_EOP['motion_chi2'],self.mean_EOP['motion_chi3']))

            file.write('%-31s%3s%-31s \n' % ('Data source', ': ', f'{self.source}'))
            file.write('%-31s%3s%-31s \n' % ('Horizontal domain', ': ', 'global'))
            file.write('%-31s%3s%-31s \n' % ('Horizontal resolution', ': ', '0.5 degree (regular grid)'))
            file.write('%-31s%3s%-31s \n\n' % ('Correction', ': ', 'inverted barometer correction over ocean applied, surface topography effect considered'))

            file.write('%-31s%3s%-31s \n' % ('Data assimilated', ': ', 'None'))

            file.write(5*'------------------------------------'+'\n')

            file.write('%-5s %-5s %-5s %-5s %-10s %20s %20s %35s %35s %20s %20s \n' %
                       ('YYYY', 'MM','DD', 'HH','MJD',' ',f'{self.portionType} mass term',' ',' ',f'{self.portionType} motion term',' '))
            file.write('%-10s %-10s %-10s %-10s %-10s %-25s %-25s %-24s %-24s %-20s %-20s \n\n' %
                       (' ', ' ', ' ', ' ', ' ', 'x-component ','y-component','z-component ', 'x-component ','y-component','z-component'))
            self._mainContent(EAM=self.__EAM,file=file)

        print(f'Finished {self.portionType} in {self.time_year} saving!\n'
              f'Save path is:{self.__fileFullPath}\n'
              f'===========================================================================')

    def _mainContent(self,EAM,file):
        for i in np.arange(len(EAM)):
            file.write('%-5s %-5s %-5s %-5s %10s %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e\n' %
                       (EAM[i]['YYYY'],EAM[i]['MM'],EAM[i]['DD'],EAM[i]['HH'],EAM[i]['MJD'],
                        EAM[i]['mass_chi1'],EAM[i]['mass_chi2'],EAM[i]['mass_chi3'],
                        EAM[i]['motion_chi1'],EAM[i]['motion_chi2'],EAM[i]['motion_chi3']))


def demo_without_demean():
    import xarray as xr
    from pysrc.earth_rotation.EarthOrientation_Massive import EOP_Massive
    from SaGEA.auxiliary.aux_tool.MathTool import MathTool
    # from tqdm import tqdm
    begin_date,end_date = '2000-01-01','2005-12-31'
    date_year = pd.date_range(start=begin_date,end=end_date,freq='YE').strftime("%Y").tolist()
    date_month = pd.date_range(start=begin_date,end=end_date,freq='MS').strftime("%Y-%m").tolist()
    date_day = pd.date_range(start=begin_date,end=end_date,freq="D").strftime("%Y-%m-%d").tolist()
    date_hour = pd.date_range(start='00:00:00', end='23:59:59', freq='24h').strftime('%H:%M:%S').tolist()

    for year in date_year:
        EOP_dateset = []
        date_month = pd.date_range(start=f"{year}-01-01",end=f"{year}-12-31",freq='MS').strftime("%Y-%m-%d").tolist()
        date_day = pd.date_range(start=f"{year}-01-01",end=f"{year}-12-31",freq="D").strftime("%Y-%m-%d").tolist()
        for date_str in date_month:
            # print(date_str)
            for epoch_str in date_hour:
                lat, lon = MathTool.get_global_lat_lon_range(resolution=0.5)
                sp_arr = np.random.uniform(-100, 100, (len(lat), len(lon)))
                v_arr = np.random.uniform(-10, 10, (len(lat), len(lon)))
                u_arr = np.random.uniform(-10, 10, (len(lat), len(lon)))
                lev_pressure = np.random.uniform(0, 100000, 37)

                a = EOP_Massive(date=date_str, epoch=epoch_str, type=EAMtype.AAM)
                a.setlatlon(lat=lat, lon=lon)
                a.PM_mass_term(Ps=sp_arr)
                a.PM_motion_term(Us=u_arr, Vs=v_arr, levPres=lev_pressure)
                a.LOD_mass_term(Ps=sp_arr)
                a.LOD_motion_term(Us=u_arr, levPres=lev_pressure)
                eop_data = a.GetCurrentEOP()
                EOP_dateset.append(eop_data)

        b = StorageEOP()
        b.setRootDir(fileDir='I:/HUST_EAM')
        b.setEAM_Information(EAM=EOP_dateset, type=EAMtype.AAM)
        b.setSource_Information(source='Random data')
        b.EOPstyle_ByProduct()

def demo_with_demean():
    import xarray as xr
    from pysrc.earth_rotation.EarthOrientation_Massive import EOP_Massive
    from SaGEA.auxiliary.aux_tool.MathTool import MathTool
    # from tqdm import tqdm
    begin_date,end_date = '2000-01-01','2005-12-31'
    mean_EOP = {'mass_chi1':0,'mass_chi2':0,'mass_chi3':0,
                'motion_chi1':0,'motion_chi2':0,'motion_chi3':0}
    date_year = pd.date_range(start=begin_date,end=end_date,freq='YE').strftime("%Y").tolist()
    date_month = pd.date_range(start=begin_date,end=end_date,freq='MS').strftime("%Y-%m").tolist()
    date_day = pd.date_range(start=begin_date,end=end_date,freq="D").strftime("%Y-%m-%d").tolist()
    date_hour = pd.date_range(start='00:00:00', end='23:59:59', freq='24h').strftime('%H:%M:%S').tolist()


    for year in date_year:
        EOP_dateset = []
        date_month = pd.date_range(start=f"{year}-01-01",end=f"{year}-12-31",freq='MS').strftime("%Y-%m-%d").tolist()
        date_day = pd.date_range(start=f"{year}-01-01",end=f"{year}-12-31",freq="D").strftime("%Y-%m-%d").tolist()
        for date_str in date_month:
            # print(date_str)
            for epoch_str in date_hour:
                lat, lon = MathTool.get_global_lat_lon_range(resolution=0.5)
                sp_arr = np.random.uniform(-100, 100, (len(lat), len(lon)))
                v_arr = np.random.uniform(-10, 10, (len(lat), len(lon)))
                u_arr = np.random.uniform(-10, 10, (len(lat), len(lon)))
                lev_pressure = np.random.uniform(0, 100000, 37)

                a = EOP_Massive(date=date_str, epoch=epoch_str, type=EAMtype.AAM)
                a.setlatlon(lat=lat, lon=lon)
                a.PM_mass_term(Ps=sp_arr)
                a.PM_motion_term(Us=u_arr, Vs=v_arr, levPres=lev_pressure)
                a.LOD_mass_term(Ps=sp_arr)
                a.LOD_motion_term(Us=u_arr, levPres=lev_pressure)
                a.GetMeanFiled_dict(EOP_Mean=mean_EOP)
                eop_data = a.GetCurrentEOP()
                EOP_dateset.append(eop_data)

        b = StorageEOP()
        b.setRootDir(fileDir='I:/HUST_EAM')
        b.setEAM_Information(EAM=EOP_dateset, type=EAMtype.AAM)
        b.setSource_Information(source='random data')
        b.setMeanFiled_Information(begin_date=begin_date,end_date=end_date,mean_EOP=mean_EOP)
        b.EOPstyle_DeMean()

if __name__ == '__main__':
    demo_with_demean()