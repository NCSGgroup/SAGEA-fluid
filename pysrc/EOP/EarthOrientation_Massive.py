import numpy as np
import pandas as pd
from pysrc.ancillary.constant.GeoConstant import EOPConstant
from pysrc.ancillary.constant.Setting import EAMtype
import time
class EOP_Massive:
    def __init__(self,date='2010-01-01',epoch='00:00:00',type=EAMtype.AAM):
        self.lat,self.lon = None,None
        self.date = date
        self.epoch = epoch
        self.mjd = f"{(pd.to_datetime(date + ' ' + epoch).to_julian_date() - 2400000.5):.3f}"
        self.mass_term = {"chi1":0,"chi2":0,"chi3":0}
        self.motion_term = {"chi1":0,"chi2":0,"chi3":0}
        self.EOP = {}
        self.MeanFiled = {"mass_chi1":0,"mass_chi2":0,"mass_chi3":0,
                          "motion_chi1":0,"motion_chi2":0,"motion_chi3":0}
        self.EAMtype = type
        self.start_time = time.time()
    def GetCurrentEOP(self):
        str_date = self.date.split('-')
        str_epoch = self.epoch.split(':')
        self.EOP['YYYY'],self.EOP['MM'],self.EOP['DD'],self.EOP['HH']= \
            str_date[0],str_date[1],str_date[2],str_epoch[0]

        self.EOP['MJD'] = self.mjd

        self.EOP['mass_chi1'],self.EOP['mass_chi2'],self.EOP['mass_chi3'] = \
            self.mass_term['chi1'],self.mass_term['chi2'],self.mass_term['chi3']

        self.EOP['motion_chi1'], self.EOP['motion_chi2'], self.EOP['motion_chi3'] = \
            self.motion_term['chi1'], self.motion_term['chi2'], self.motion_term['chi3']

        end_time = time.time()
        print(f"-------------------------------------------------------\n"
              f"Finished {self.EAMtype.name} at {self.epoch} on {self.date}\n"
              f"Time consumption: {end_time - self.start_time:.4f} s\n"
              f"-------------------------------------------------------")

        return self.EOP

    def GetMeanFiled(self,mass_chi1,mass_chi2,mass_chi3,motion_chi1,motion_chi2,motion_chi3):
        self.MeanFiled["mass_chi1"],self.MeanFiled["mass_chi2"],self.MeanFiled["mass_chi3"]=mass_chi1,mass_chi2,mass_chi3
        self.MeanFiled["motion_chi1"], self.MeanFiled["motion_chi2"], self.MeanFiled["motion_chi3"] = motion_chi1, motion_chi2, motion_chi3
        return self.MeanFiled

    def GetMeanFiled_dict(self, EOP_Mean:dict):
        self.MeanFiled["mass_chi1"], self.MeanFiled["mass_chi2"], self.MeanFiled[
            "mass_chi3"] = EOP_Mean['mass_chi1'], EOP_Mean['mass_chi2'],EOP_Mean['mass_chi3']
        self.MeanFiled["motion_chi1"], self.MeanFiled["motion_chi2"], self.MeanFiled[
            "motion_chi3"] = EOP_Mean['motion_chi1'], EOP_Mean['motion_chi2'], EOP_Mean['motion_chi3']
        return self.MeanFiled


    def setlatlon(self,lat,lon):
        self.lat = lat
        self.lon = lon
        return self

    def PM_mass_term_SH(self,SH):

        coef_numerator = -1.098 * np.sqrt(5) * (EOPConstant.radius ** 2) * EOPConstant.Mass
        coef_denominator = np.sqrt(3) * (1 + EOPConstant.k2_load) * (EOPConstant.Cm - EOPConstant.Am)

        chi1 = SH[7] * (coef_numerator / coef_denominator)
        chi2 = SH[5] * (coef_numerator / coef_denominator)

        chi = {"chi1": chi1, "chi2": chi2}
        self.mass_term['chi1'] = chi1-self.MeanFiled['mass_chi1']
        self.mass_term['chi2'] = chi2-self.MeanFiled['mass_chi2']

        return chi

    def PM_mass_term(self, Ps):
        """
        :param Ps: Ps is fluid expressed as Pa, shape is (lat,lon)
        :return:
        """

        phi, lam = np.deg2rad(self.lat), np.deg2rad(self.lon)
        dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[0])
        phi_2D, lam_2D = np.meshgrid(phi, lam, indexing='ij')

        sin_phi, cos_phi = np.sin(phi_2D), np.cos(phi_2D)
        sin_lam, cos_lam = np.sin(lam_2D), np.cos(lam_2D)

        dA = (EOPConstant.radius ** 2) * cos_phi * dphi * dlam

        dI13 = Ps * sin_phi * cos_phi * cos_lam * dA
        dI23 = Ps * sin_phi * cos_phi * sin_lam * dA

        I13 = np.sum(dI13, axis=(0, 1))
        I23 = np.sum(dI23, axis=(0, 1))

        coef = (-1.098 * (EOPConstant.radius ** 2)) / ((EOPConstant.Cm - EOPConstant.Am) * EOPConstant.grav)

        chi1 = coef * I13
        chi2 = coef * I23
        PM = {"chi1": chi1, "chi2": chi2}
        self.mass_term['chi1'] = chi1-self.MeanFiled['mass_chi1']
        self.mass_term['chi2'] = chi2-self.MeanFiled['mass_chi2']

        return PM

    def PM_motion_term(self,Us,Vs,levPres,Ps=None,Zth=None):
        """

        :param Us: unit is m/s, shape is (lev,lat,lon)
        :param Vs: unit is m/s, shape is (lev,lat,lon)
        :param levPres: unit is Pa for AAM but m for OAM, shape is (lev,)
        :param Ps: unit is Pa for AAM but m for OAM, shape is (lat,lon)
        :param Zth: unit is m^2/s^2 for AAM (OAM doesn't consider)
        :param type: AAM and OAM, also supports for HAM, and HAM is the same with OAM
        :return: dict, using chi1 and chi2 to get the value
        """

        phi, lam = np.deg2rad(self.lat), np.deg2rad(self.lon)
        dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[0])

        phi_2D, lam_2D = np.meshgrid(phi, lam, indexing="ij")

        sin_phi, cos_phi = np.sin(phi_2D), np.cos(phi_2D)
        sin_lam, cos_lam = np.sin(lam_2D), np.cos(lam_2D)

        dp_g = []
        if self.EAMtype == EAMtype.AAM:
            for lev in np.arange(len(levPres)):
                temp_dp_g = self.__dp_AAM(levPres=levPres, lev=lev, surPres=Ps, geoHeight=Zth) / EOPConstant.grav
                temp_dp_g = temp_dp_g.reshape((len(self.lat), len(self.lon)))
                # print(f"{temp_dp_g.flatten()}")
                dp_g.append(temp_dp_g)
        else:
            for lev in np.arange(len(levPres)):
                temp_dp_g = self.__dp_OAM(levDepth=levPres, lev=lev, surSeaHeight=Ps) / EOPConstant.grav
                temp_dp_g = temp_dp_g.reshape((len(self.lat), len(self.lon)))
                dp_g.append(temp_dp_g)
        dp_g = np.array(dp_g)

        dU = Us*dp_g
        dV = Vs*dp_g

        U,V = np.sum(dU,axis=0),np.sum(dV,axis=0)

        dA = (EOPConstant.radius ** 2) * cos_phi * dphi * dlam
        # dA = dlam * dphi

        L1 = U * sin_phi * cos_lam - V * sin_lam
        L2 = U * sin_phi * sin_lam + V * cos_lam
        # print(f"L1 L2:\n{L1}\n{L2}")

        h1 = np.sum(L1*dA,axis=(0,1))
        h2 = np.sum(L2*dA,axis=(0,1))
        # print(f"h1 h2:{h1, h2}")
        coef = -1.5913/(EOPConstant.omega*(EOPConstant.Cm-EOPConstant.Am))

        chi1 = coef*h1
        chi2 = coef*h2

        PM = {'chi1':chi1,'chi2':chi2}
        self.motion_term['chi1'] = chi1-self.MeanFiled['motion_chi1']
        self.motion_term['chi2'] = chi2-self.MeanFiled['motion_chi2']

        return PM

    def LOD_mass_term_SH(self,SH):

        coef_numerator = 0.753 * (EOPConstant.radius ** 2) * EOPConstant.Mass * 2
        coef_denominator = (1 + EOPConstant.k2_load) * EOPConstant.Cm * 3

        C00 = SH[0]
        C20 = SH[6]
        chi3 = (coef_numerator / coef_denominator) * (C00 - np.sqrt(5) * C20)
        LOD = {"chi3": chi3}
        self.mass_term['chi3'] = chi3-self.MeanFiled['mass_chi3']

        return LOD
    def LOD_mass_term(self,Ps):
        """
        :param Ps: Ps is fluid expressed as Pa, shape is (lat,lon)
        :return:
        """

        phi, lam = np.deg2rad(self.lat), np.deg2rad(self.lon)
        dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[2])
        phi_2D, lam_2D = np.meshgrid(phi, lam, indexing="ij")

        sin_phi, cos_phi = np.sin(phi_2D), np.cos(phi_2D)
        sin_lam, cos_lam = np.sin(lam_2D), np.cos(lam_2D)

        dA = (EOPConstant.radius ** 2) * cos_phi * dphi * dlam
        dI33 = Ps * cos_phi * cos_phi * dA

        I33 = np.sum(dI33, axis=(0, 1))

        coef = (0.753 * (EOPConstant.radius ** 2)) / (EOPConstant.Cm * EOPConstant.grav)

        chi3 = coef * I33
        LOD = {'chi3': chi3}
        self.mass_term['chi3'] = chi3-self.MeanFiled['mass_chi3']

        return LOD

    def LOD_motion_term(self,Us,levPres,Ps=None,Zth=None):

        phi, lam = np.deg2rad(self.lat), np.deg2rad(self.lon)
        dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[2])
        phi_2D, lam_2D = np.meshgrid(phi, lam, indexing="ij")

        sin_phi, cos_phi = np.sin(phi_2D), np.cos(phi_2D)
        sin_lam, cos_lam = np.sin(lam_2D), np.cos(lam_2D)

        dp_g = []
        if self.EAMtype == EAMtype.AAM:
            for lev in np.arange(len(levPres)):
                temp_dp_g = self.__dp_AAM(levPres=levPres,lev=lev,surPres=Ps,geoHeight=Zth)/EOPConstant.grav
                temp_dp_g = temp_dp_g.reshape((len(self.lat),len(self.lon)))
                dp_g.append(temp_dp_g)
        else:
            for lev in np.arange(len(levPres)):
                temp_dp_g = self.__dp_OAM(levDepth=levPres,lev=lev,surSeaHeight=Ps)/EOPConstant.grav
                temp_dp_g = temp_dp_g.reshape((len(self.lat),len(self.lon)))
                dp_g.append(temp_dp_g)

        dp_g = np.array(dp_g)
        dU = Us*dp_g
        U = np.sum(dU,axis=0)

        dA = (EOPConstant.radius**2)*cos_phi*dphi*dlam

        h3 = np.sum(U*cos_phi*dA,axis=(0,1))

        coef = 0.998/(EOPConstant.Cm*EOPConstant.omega)

        chi3 = coef*h3
        LOD = {'chi3':chi3}
        self.motion_term['chi3'] = chi3-self.MeanFiled['motion_chi3']

        return LOD

    def __dp_AAM_full(self,levPres,lev,surPres=None,geoHeight=None):
        sampe_arr = np.ones((len(self.lat), len(self.lon))).flatten()
        iso_pres = []
        Radius,grav = EOPConstant.radius,EOPConstant.grav

        if geoHeight is None:
            iso_R = np.ones((len(levPres),len(sampe_arr)))*Radius

        else:
            R_sets = Radius*sampe_arr
            geo_height = geoHeight.reshape(len(geoHeight),-1)/grav
            iso_R = geo_height+R_sets

        if surPres is None:
            for i in np.arange(len(levPres)):
                pres_level = levPres[i]*sampe_arr
                iso_pres.append(pres_level)

            iso_R = np.array(iso_R)
            iso_pres = np.array(iso_pres)
            top_pres = np.zeros(len(sampe_arr))


            if lev == len(levPres)-1:
                return (iso_pres[lev]-top_pres)*iso_R[lev]
            else:
                return (iso_pres[lev]-iso_pres[lev+1])*iso_R[lev]

        else:
            sp_flatten = surPres.flatten()
            for i in np.arange(len(levPres)):

                pres_level = levPres[i]*sampe_arr
                if (levPres[i]-sp_flatten<=0).all():
                    iso_pres.append(pres_level)
                    continue
                index = levPres[i]-sp_flatten>0
                pres_level[index] = sp_flatten[index]
                iso_pres.append(pres_level)
            iso_R = np.array(iso_R)
            iso_pres = np.array(iso_pres)

            if lev == 0:
                return (sp_flatten - iso_pres[lev]) * iso_R[lev]
            else:
                return (iso_pres[lev - 1] - iso_pres[lev]) * iso_R[lev]

    def __dp_OAM_full(self,levDepth,lev,surSeaHeight=None):
        sample_arr = np.ones((len(self.lat),len(self.lon))).flatten()
        iso_pres,iso_R = [],[]
        Radius = EOPConstant.radius

        if surSeaHeight is None:
            for i in np.arange(len(levDepth)):
                depth_level = levDepth[i] * sample_arr
                pres_level = EOPConstant.grav*EOPConstant.rho_water*depth_level
                radius_level = sample_arr * Radius + depth_level
                iso_R.append(radius_level)
                iso_pres.append(pres_level)

            iso_R = np.array(iso_R)
            iso_pres = np.array(iso_pres)
            top_pres = np.zeros(len(sample_arr))


            if lev == 0:
                return (top_pres-iso_pres[lev]) * iso_R[lev]
            else:
                return (iso_pres[lev-1] - iso_pres[lev]) * iso_R[lev]

        else:
            ssh_flatten = surSeaHeight.flatten()
            for i in np.arange(len(levDepth)):
                depth_level = sample_arr*levDepth[i]
                radius_level = sample_arr * Radius + depth_level
                iso_R.append(radius_level)
                if (depth_level[i]-ssh_flatten<=0).all():
                    pres_level = EOPConstant.grav*EOPConstant.rho_water*depth_level
                    iso_pres.append(pres_level)
                    continue
                index = depth_level[i]-ssh_flatten>0
                depth_level[index] = ssh_flatten[index]
                pres_level = EOPConstant.grav*EOPConstant.rho_water*depth_level
                iso_pres.append(pres_level)
            iso_pres = np.array(iso_pres)
            iso_R = np.array(iso_R)

            if lev == 0:
                return (ssh_flatten - iso_pres[lev]) * iso_R[lev]
            else:
                return (iso_pres[lev - 1] - iso_pres[lev]) * iso_R[lev]

    def __dp_AAM(self,levPres,lev,lat,lon,surPres=None,geoHeight=None):
        iso_pres = self.__get_lev(levs=levPres,lat=lat,lon=lon,surface=surPres)
        # print(f"iso pres:{iso_pres.shape}")
        # print(f"levPres:{levPres.shape}")
        sample_arr = np.ones((len(lat), len(lon))).flatten()
        Radius, grav = EOPConstant.radius, EOPConstant.grav

        if geoHeight is None:
            iso_R = np.ones((len(levPres), len(sample_arr))) * Radius

        else:
            R_sets = Radius * sample_arr
            geo_height = geoHeight.reshape(len(geoHeight), -1) / grav
            iso_R = geo_height + R_sets

        return (iso_pres[lev]-iso_pres[lev+1])*iso_R[lev]

    def __dp_OAM(self,levDepth,lev,lat,lon,surSeaHeight=None,geoHeight=None):
        sample_arr = np.ones((len(lat), len(lon))).flatten()
        Radius, grav = EOPConstant.radius, EOPConstant.grav
        iso_pres = self.__get_lev(levs=levDepth,lat=lat,lon=lon,surface=surSeaHeight)*EOPConstant.grav * EOPConstant.rho_water

        if geoHeight is None:
            iso_R = np.ones((len(levDepth), len(sample_arr))) * Radius
        else:
            iso_R = []
            for i in np.arange(len(levDepth)):
                depth_level = sample_arr * levDepth[i]
                radius_level = sample_arr * Radius + depth_level
                iso_R.append(radius_level)

        return (iso_pres[lev]-iso_pres[lev+1])*iso_R[lev]



    def __get_lev(self,levs,lat,lon,surface=None):
        sample_arr = np.ones((len(lat),len(lon))).flatten()
        iso_levs = np.ones((len(levs)+1,len(sample_arr)))

        for i in np.arange(1,len(levs)):
            iso_levs[i] = 0.5*(levs[i]+levs[i-1])*sample_arr

        iso_levs[0] = 2*levs[0]*sample_arr-iso_levs[1]
        iso_levs[-1] = 2*levs[-1]*sample_arr-iso_levs[-2]
        if surface is not None:
            sur_arr = surface.flatten()
            for i in np.arange(len(iso_levs)):
                if (iso_levs[i]-sur_arr<=0).all():
                    continue
                index = iso_levs[i]-sur_arr>0
                iso_levs[i][index] = sur_arr[index]
        return iso_levs



def demo_LoadForm():
    from lib.SaGEA.auxiliary.aux_tool.MathTool import MathTool

    from pysrc.ancillary.storage_file.StorageEOP import StorageEOP

    lat,lon = MathTool.get_global_lat_lon_range(resolution=0.5)
    sp_arr = np.random.uniform(-100,100,(len(lat),len(lon)))
    v_arr = np.random.uniform(-10,10,(len(lat),len(lon)))
    u_arr = np.random.uniform(-10,10,(len(lat),len(lon)))
    lev_pressure = np.random.uniform(0,100000,37)
    date_str,time_str = '2020-01-01',"06:00:00"

    a = EOP_Massive(date=date_str,epoch=time_str,type=EAMtype.AAM)
    a.setlatlon(lat=lat,lon=lon)
    a.PM_mass_term(Ps=sp_arr)
    a.PM_motion_term(Us=u_arr,Vs=v_arr,levPres=lev_pressure)
    a.LOD_mass_term(Ps=sp_arr)
    a.LOD_motion_term(Us=u_arr,levPres=lev_pressure)
    eop_data = a.GetCurrentEOP()
    print(eop_data.keys())

    EOP_list = [eop_data]
    b = StorageEOP()
    b.setRootDir(fileDir='I:/')
    b.setEAM_Information(EAM=EOP_list,type=EAMtype.AAM)
    b.setSource_Information(source='Random data')
    b.EOPstyle_ByProduct()


def demo_LoadTrue():
    import xarray as xr

    sp_file = xr.open_dataset("I:\ERA5\MAD_SL/2005/sp-200501.nc")
    u_file = xr.open_dataset("I:\ERA5\MAD_PL/2005/u_wind-200501.nc")
    v_file =xr.open_dataset("I:\ERA5\MAD_PL/2005/v_wind-200501.nc")

    sp = sp_file['sp'].values[0]
    u_wind = u_file['u'].values[0]
    v_wind = v_file['v'].values[0]
    lev_pressure = v_file['pressure_level'].values
    lat,lon = v_file['latitude'].values,v_file['longitude'].values
    time_epoch = v_file['valid_time'].values[0]
    date,full_time = str(time_epoch).split('T')
    epoch = full_time.split('.')[0]
    print(sp.shape,u_wind.shape,v_wind.shape)
    print(date,epoch)

    a = EOP_Massive(date=date,epoch=epoch,type=EAMtype.AAM)
    a.setlatlon(lat=lat,lon=lon)
    a.PM_mass_term(Ps=sp)
    a.PM_motion_term(Us=u_wind,Vs=v_wind,levPres=lev_pressure)
    a.LOD_mass_term(Ps=sp)
    a.LOD_motion_term(Us=u_wind,levPres=lev_pressure)
    eop_data = a.GetCurrentEOP()
    print(eop_data)


def demo_SaveFile():
    from lib.SaGEA.auxiliary.aux_tool.MathTool import MathTool
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
        print(len(EOP_dateset))

if __name__ =="__main__":
    demo_LoadForm()
