import numpy as np
import pandas as pd


from pysrc.ancillary.constant.GeoConstant import EOPConstant
import time
import SaGEA.auxiliary.preference.EnumClasses as Enums
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from pysrc.ancillary.load_file.DataClass import SHC,GRID
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.sealevel_equation.SeaLevelEquation import PseudoSpectralSLE
from pysrc.ancillary.geotools.LLN import LoveNumber
from pysrc.ancillary.constant.Setting import EAMtype,EOPtype




class EOP_Massive:
    def __init__(self):
        # self.temp_path = FileTool.get_project_dir("result/EOP/temp/")
        self.lat,self.lon = None,None

    def setlatlon(self,lat,lon):
        self.lat = lat
        self.lon = lon
        return self

    def Get_PM(self,type:EOPtype):
        if type == EOPtype.Mass:
            pass
        pass

    def Get_LOD(self,type:EOPtype):
        pass

    def PM_mass_term_SH(self,SH):
        coef_numerator = -1.098 * np.sqrt(5) * (EOPConstant.radius ** 2) * EOPConstant.Mass
        coef_denominator = np.sqrt(3) * (1 + EOPConstant.k2_load) * (EOPConstant.Cm - EOPConstant.Am)

        chi1 = SH[7] * (coef_numerator / coef_denominator)
        chi2 = SH[5] * (coef_numerator / coef_denominator)

        chi = {"chi1": chi1, "chi2": chi2}
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
        return PM

    def PM_motion_term(self,Us,Vs,levPres,Ps=None,Zth=None,type=EAMtype.AAM):
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
        print(f"dphi:{dphi}")
        print(f"dlam:{dlam}")
        phi_2D, lam_2D = np.meshgrid(phi, lam, indexing="ij")

        sin_phi, cos_phi = np.sin(phi_2D), np.cos(phi_2D)
        sin_lam, cos_lam = np.sin(lam_2D), np.cos(lam_2D)

        dp_g = []
        if type == EAMtype.AAM:
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
        print(f"dp_g shape is:{dp_g.shape}")
        print(f"Us Vs:{Us.shape,Vs.shape}")
        dU = Us*dp_g
        dV = Vs*dp_g

        print(f"dU dV:{dU.shape}{dV.shape}")

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
        return PM

    def LOD_mass_term_SH(self,SH):
        coef_numerator = 0.753 * (EOPConstant.radius ** 2) * EOPConstant.Mass * 2
        coef_denominator = (1 + EOPConstant.k2_load) * EOPConstant.Cm * 3

        C00 = SH[0]
        C20 = SH[6]
        chi3 = (coef_numerator / coef_denominator) * (C00 - np.sqrt(5) * C20)
        LOD = {"chi3": chi3}
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
        return LOD

    def LOD_motion_term(self,Us,levPres,Ps=None,Zth=None,type=EAMtype.AAM):

        phi, lam = np.deg2rad(self.lat), np.deg2rad(self.lon)
        dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[2])
        phi_2D, lam_2D = np.meshgrid(phi, lam, indexing="ij")

        sin_phi, cos_phi = np.sin(phi_2D), np.cos(phi_2D)
        sin_lam, cos_lam = np.sin(lam_2D), np.cos(lam_2D)

        dp_g = []
        if type == EAMtype.AAM:
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
        return LOD

    def __dp_AAM(self,levPres,lev,surPres=None,geoHeight=None):
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
            for i in np.arange(len(levPres)):
                sp_flatten = surPres.flatten()
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

    def __dp_OAM(self,levDepth,lev,surSeaHeight=None):
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


            if lev == len(levDepth) - 1:
                return (iso_pres[lev] - top_pres) * iso_R[lev]
            else:
                return (iso_pres[lev] - iso_pres[lev + 1]) * iso_R[lev]

        else:
            for i in np.arange(len(levDepth)):
                ssh_flatten = surSeaHeight.flatten()
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

    def mean_field(self,begin='2005-01-01',end='2006-01-31',freq='D'):
        date_range = pd.date_range(start=begin,end=end,freq=freq).strftime("%Y%m%d").tolist()
        date_year = pd.date_range(start=begin,end=end,freq='YE').strftime("%Y").tolist()
        print(self.temp_path)
        print(date_range)
        print(date_year)



def demo1():
    import xarray as xr
    from pysrc.aliasing_model.specify.IBcorrection import IBcorrection
    sp_file = xr.open_dataset("I:/ERA5/MAD_SL/2000/sp-200001.nc")
    u_file = xr.open_dataset("I:/ERA5/MAD_PL/2000/u_wind-200001.nc")
    v_file = xr.open_dataset("I:/ERA5/MAD_PL/2000/v_wind-200001.nc")
    geo_file = xr.open_dataset("I:/ERA5/MAD_PL/2000/geop-200001.nc")

    sp,u,v,levPres = sp_file['sp'].values,u_file['u'].values,v_file['v'].values,u_file['pressure_level'].values*100
    print(f"sp/u/v/levPres shapes:{sp.shape,u.shape,v.shape,levPres.shape}")
    lat,lon = sp_file['latitude'].values,sp_file['longitude'].values
    a = EOP_Massive()
    a.setlatlon(lat=lat,lon=lon)

    PM_motion = a.PM_motion_term(Us=u[0],Vs=v[0],levPres=levPres,Ps=sp[0])
    print(f"PM motion: {PM_motion}")

    ib = IBcorrection(lat=lat, lon=lon)
    asp_ib_f = ib.correct(grids=sp.flatten())
    sp_ib = asp_ib_f.reshape((len(lat),len(lon)))
    PM_mass = a.PM_mass_term_grid(Ps=sp_ib)
    print(f"PM mass: {PM_mass}")

    LOD_motion = a.LOD_motion_term(Us=u[0],levPres=levPres,Ps=sp)
    print(f"LOD motion:{LOD_motion}")

    LOD_mass = a.LOD_mass_term_grid(Ps=sp[0])
    print(f"LOD mass:{LOD_mass}")


if __name__ =="__main__":
    demo1()
