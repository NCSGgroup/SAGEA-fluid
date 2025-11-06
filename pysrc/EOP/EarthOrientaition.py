import numpy as np

from pysrc.ancillary.constant.GeoConstant import EOPConstant
import time
import lib.SaGEA.auxiliary.preference.EnumClasses as Enums
from lib.SaGEA.auxiliary.aux_tool.FileTool import FileTool
from lib.SaGEA.data_class.DataClass import SHC,GRID
from lib.SaGEA.auxiliary.aux_tool.MathTool import MathTool
from lib.SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.SAL.SeaLevelEquation import PseudoSpectralSLE
from pysrc.ancillary.geotools.LLN import LoveNumber
from pysrc.ancillary.constant.Setting import EAMtype
from tqdm import tqdm
class EOP:
    def __init__(self):
        self.rad_to_mas = EOPConstant.rad_to_mas
        self.rad_to_ms = EOPConstant.rad_to_ms

    def PM_mass_term(self,Ps,lat,lon,isMas=False):
        phi, lam = np.deg2rad(lat), np.deg2rad(lon)
        dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[0])
        phi_2D, lam_2D = np.meshgrid(phi, lam, indexing='ij')

        sin_phi, cos_phi = np.sin(phi_2D), np.cos(phi_2D)
        sin_lam, cos_lam = np.sin(lam_2D), np.cos(lam_2D)

        dA = (EOPConstant.radius ** 2) * cos_phi * dphi * dlam

        dI13 = Ps * sin_phi * cos_phi * cos_lam * dA
        dI23 = Ps * sin_phi * cos_phi * sin_lam * dA

        I13 = np.sum(dI13, axis=(1, 2))
        I23 = np.sum(dI23, axis=(1, 2))

        coef = (-1.098 * (EOPConstant.radius ** 2)) / ((EOPConstant.Cm - EOPConstant.Am) * EOPConstant.grav)

        chi1 = coef * I13
        chi2 = coef * I23
        if isMas:
            chi1 = chi1 * self.rad_to_mas
            chi2 = chi2 * self.rad_to_mas

        PM = {"chi1": chi1, "chi2": chi2}
        return PM

    def PM_mass_term_SH(self,SH,isMas=False):
        coef_numerator = -1.098 * np.sqrt(5) * (EOPConstant.radius ** 2) * EOPConstant.Mass
        coef_denominator = np.sqrt(3) * (1 + EOPConstant.k2_load) * (EOPConstant.Cm - EOPConstant.Am)

        '''Dobslaw 2010 coefs'''
        # ks,K2 = EOPConstant.ks,EOPConstant.k2
        # Cm,Am = EOPConstant.Cm,EOPConstant.Am
        # R, M = EOPConstant.radius,EOPConstant.Mass
        # coef_numerator = -ks*np.sqrt(5)*(R**2)*M
        # coef_denominator = (ks-K2)*(Cm-Am)*np.sqrt(3)


        chi1 = SH[:, 7] * (coef_numerator / coef_denominator)
        chi2 = SH[:, 5] * (coef_numerator / coef_denominator)
        if isMas:
            chi1 = chi1 * self.rad_to_mas
            chi2 = chi2 * self.rad_to_mas
        chi = {"chi1": chi1, "chi2": chi2}
        return chi

    def PM_mass_term_EWHSH(self,EWH_SH,isMas=False):
        coef = (4 * np.pi * (EOPConstant.radius ** 4) * EOPConstant.rho_water) / (np.sqrt(15))

        I13 = coef * EWH_SH[:, 7]
        I23 = coef * EWH_SH[:, 5]

        chi1 = (-1.098 * I13) / (EOPConstant.Cm - EOPConstant.Am)
        chi2 = (-1.098 * I23) / (EOPConstant.Cm - EOPConstant.Am)
        if isMas:
            chi1 = chi1 * self.rad_to_mas
            chi2 = chi2 * self.rad_to_mas
        chi = {"chi1": chi1, "chi2": chi2}
        return chi

    def PM_motion_term(self, Us, Vs, lat, lon, levPres, Ps, Zth=None, type=EAMtype.AAM, isMas=True):
        """
        :param u_speed: eastward along latitude (also known as zonal wind)
        :param v_speed: northward along longitude (also known as meridional wind)
        Notes: both u and v shape is (time,layer,lat,lon)
        :param lat: the range is from 90--90
        :param lon: the range is from 0-360
        :param layer: multi-layer pressure data, make sure the unit is Pa (not hPa)
        :param surf:
        Notes: surf shape is (time,lat,lon)
        :param z: used to caculate the vertical height in vertical integration
        Notes: z shape is (time, layer, lat, lon)
        :param type: Including AAM,OAM,HAM,SLAM, but only AAM and OAM are supported.
        :param isMas:  False means results are rad, True means results are mas
        :return:
        """
        chi1_series, chi2_series = [], []
        for i in tqdm(np.arange(len(Us))):
            dp_g = []
            if type == EAMtype.AAM:
                for lev in np.arange(len(levPres)):
                    if Zth is None:
                        if Ps is None:
                            temp_dp_g = self.__dp_AAM(levPres=levPres, lev=lev, lat=lat, lon=lon) / EOPConstant.grav
                        else:
                            temp_dp_g = self.__dp_AAM(levPres=levPres, lev=lev, lat=lat, lon=lon,
                                                      surPres=Ps[i]) / EOPConstant.grav
                    else:
                        if Ps is None:
                            temp_dp_g = self.__dp_AAM(levPres=levPres, lev=lev, lat=lat, lon=lon,
                                                      geoHeight=Zth[i]) / EOPConstant.grav
                        else:
                            temp_dp_g = self.__dp_AAM(levPres=levPres, lev=lev, lat=lat, lon=lon, geoHeight=Zth[i],
                                                      surPres=Ps[i]) / EOPConstant.grav
                    temp_dp_g = temp_dp_g.reshape((len(lat), len(lon)))
                    dp_g.append(temp_dp_g)
            else:
                for lev in np.arange(len(levPres)):
                    if Ps is None:
                        temp_dp_g = self.__dp_OAM(levDepth=levPres, lev=lev, lat=lat, lon=lon) / EOPConstant.grav
                    else:
                        temp_dp_g = self.__dp_OAM(levDepth=levPres, lev=lev, lat=lat, lon=lon,
                                                  surSeaHeight=Ps[i]) / EOPConstant.grav
                    temp_dp_g = temp_dp_g.reshape((len(lat), len(lon)))
                    dp_g.append(temp_dp_g)
            dp_g = np.array(dp_g)

            dU = Us[i] * dp_g
            dV = Vs[i] * dp_g

            U, V = np.sum(dU, axis=0), np.sum(dV, axis=0)

            phi, lam = np.deg2rad(lat), np.deg2rad(lon)
            dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[0])

            phi_grid, lam_grid = np.meshgrid(phi, lam, indexing="ij")
            cos_phi, sin_phi = np.cos(phi_grid), np.sin(phi_grid)
            cos_lam, sin_lam = np.cos(lam_grid), np.sin(lam_grid)

            dA = dlam * dphi

            L1 = (U * sin_phi * cos_lam * cos_phi - V * sin_lam * cos_phi)
            L2 = (U * sin_phi * sin_lam * cos_phi + V * cos_lam * cos_phi)

            h1 = np.sum(L1 * dA, axis=(0, 1))
            h2 = np.sum(L2 * dA, axis=(0, 1))

            # '''Dobslaw 2010 coefficients'''
            # coef_numerator = -EOPConstant.ks*(EOPConstant.radius**2)
            # coef_denominator = (EOPConstant.ks-EOPConstant.k2)*(EOPConstant.Cm-EOPConstant.Am)*EOPConstant.omega

            coef_numerator = -1.5913 * (EOPConstant.radius ** 2)
            coef_denominator = EOPConstant.omega * (EOPConstant.Cm - EOPConstant.Am)

            chi1 = (coef_numerator / coef_denominator) * h1
            chi2 = (coef_numerator / coef_denominator) * h2

            if isMas:
                chi1 = chi1 * self.rad_to_mas
                chi2 = chi2 * self.rad_to_mas

            chi1_series.append(chi1)
            chi2_series.append(chi2)
        chi1_series, chi2_series = np.array(chi1_series), np.array(chi2_series)

        chi = {"chi1": chi1_series, "chi2": chi2_series}
        return chi

    def LOD_mass_term_SH(self, SH, isMs=False):
        """
        :param SH: Different with PM EWH term, the type of LOD is Stokes coefficients.
        :param isMas: the same follow before.
        :return: chi3 with unit mas or rad, the unit of (Delta)LOD is seconds (s).
        """
        coef_numerator = 0.753*(EOPConstant.radius**2)*EOPConstant.Mass*2
        coef_denominator = (1+EOPConstant.k2_load)*EOPConstant.Cm*3

        C00 = SH[:, 0]
        C20 = SH[:, 6]
        chi3 = (coef_numerator/coef_denominator) * (C00 - np.sqrt(5) * C20)
        if isMs:
            chi3 = chi3*self.rad_to_ms
        LOD = {"chi3": chi3}
        return LOD

    def LOD_mass_term(self, Ps, lat, lon, isMs=True):

        coef_numerator = 0.998*(EOPConstant.radius**3)
        coef_denominator = EOPConstant.Cm*EOPConstant.omega*EOPConstant.grav

        phi, lam = np.deg2rad(lat), np.deg2rad(lon)
        dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[2])
        phi_grid, lam_grid = np.meshgrid(phi, lam, indexing="ij")

        cos_phi = np.cos(phi_grid)
        dA = cos_phi * dphi * dlam

        kernal = Ps * cos_phi * cos_phi * dA
        chi3 = (coef_numerator/coef_denominator)*np.sum(kernal, axis=(1, 2))
        if isMs:
            chi3 = chi3*self.rad_to_ms
        LOD = {"chi3": chi3}
        return LOD

    def LOD_motion_term(self, Us, lat, lon, levPres, Ps=None, Zth=None,type=EAMtype.AAM, isMs=True):

        coef_numerator = 0.998*(EOPConstant.radius**2)
        coef_denominator = EOPConstant.Cm*EOPConstant.omega

        chi3_series = []
        for i in tqdm(np.arange(len(Us))):
            dp_g = []
            if type == EAMtype.AAM:
                for lev in np.arange(len(levPres)):
                    if Zth is None:
                        if Ps is None:
                            temp_dp_g = self.__dp_AAM(levPres=levPres, lev=lev, lat=lat, lon=lon) / EOPConstant.grav
                        else:
                            temp_dp_g = self.__dp_AAM(levPres=levPres, lev=lev, lat=lat, lon=lon,
                                                      surPres=Ps[i]) / EOPConstant.grav
                    else:
                        if Ps is None:
                            temp_dp_g = self.__dp_AAM(levPres=levPres, lev=lev, lat=lat, lon=lon,
                                                      geoHeight=Zth[i]) / EOPConstant.grav
                        else:
                            temp_dp_g = self.__dp_AAM(levPres=levPres, lev=lev, lat=lat, lon=lon, geoHeight=Zth[i],
                                                      surPres=Ps[i]) / EOPConstant.grav
                    temp_dp_g = temp_dp_g.reshape((len(lat), len(lon)))
                    dp_g.append(temp_dp_g)
            else:
                for lev in np.arange(len(levPres)):
                    if Ps is None:
                        temp_dp_g = self.__dp_OAM(levDepth=levPres, lev=lev, lat=lat, lon=lon) / EOPConstant.grav
                    else:
                        temp_dp_g = self.__dp_OAM(levDepth=levPres, lev=lev, lat=lat, lon=lon,
                                                  surSeaHeight=Ps[i]) / EOPConstant.grav
                    temp_dp_g = temp_dp_g.reshape((len(lat), len(lon)))
                    dp_g.append(temp_dp_g)
                dp_g = np.array(dp_g)

            dU = Us[i] * dp_g
            U = np.sum(dU, axis=0)

            phi, lam = np.deg2rad(lat), np.deg2rad(lon)
            dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[0])

            phi_grid, lam_grid = np.meshgrid(phi, lam, indexing='ij')
            cos_phi = np.cos(phi_grid)
            dA = cos_phi * dphi * dlam

            h3 = np.sum(U * dA*cos_phi, axis=(0, 1))
            chi3 = (coef_numerator/coef_denominator)* h3
            if isMs:
                chi3 = chi3*self.rad_to_ms
            chi3_series.append(chi3)

        chi3_series = np.array(chi3_series)
        LOD = {"chi3": chi3_series}
        return LOD
    def __dp_AAM_full(self, levPres, lev, lat, lon, surPres=None, geoHeight=None):
        sampe_arr = np.ones((len(lat), len(lon))).flatten()
        iso_pres = []
        Radius, grav = EOPConstant.radius, EOPConstant.grav

        if geoHeight is None:
            iso_R = np.ones((len(levPres), len(sampe_arr))) * Radius

        else:
            R_sets = Radius * sampe_arr
            geo_height = geoHeight.reshape(len(geoHeight), -1) / grav
            iso_R = geo_height + R_sets

        if surPres is None:
            for i in np.arange(len(levPres)):
                pres_level = levPres[i] * sampe_arr
                iso_pres.append(pres_level)

            iso_R = np.array(iso_R)
            iso_pres = np.array(iso_pres)
            top_pres = np.zeros(len(sampe_arr))

            if lev == len(levPres) - 1:
                return (iso_pres[lev] - top_pres) * iso_R[lev]
            else:
                return (iso_pres[lev] - iso_pres[lev + 1]) * iso_R[lev]

        else:
            sp_flatten = surPres.flatten()
            for i in np.arange(len(levPres)):
                pres_level = levPres[i] * sampe_arr
                if (levPres[i] - sp_flatten <= 0).all():
                    iso_pres.append(pres_level)
                    continue
                index = levPres[i] - sp_flatten > 0
                pres_level[index] = sp_flatten[index]
                iso_pres.append(pres_level)
            iso_R = np.array(iso_R)
            iso_pres = np.array(iso_pres)

            if lev == 0:
                return (sp_flatten - iso_pres[lev]) * iso_R[lev]
            else:
                return (iso_pres[lev - 1] - iso_pres[lev]) * iso_R[lev]

    def __dp_OAM_full(self, levDepth, lev, lat, lon, surSeaHeight=None):
        sample_arr = np.ones((len(lat), len(lon))).flatten()
        iso_pres, iso_R = [], []
        Radius = EOPConstant.radius

        if surSeaHeight is None:
            for i in np.arange(len(levDepth)):
                depth_level = levDepth[i] * sample_arr
                pres_level = EOPConstant.grav * EOPConstant.rho_water * depth_level
                radius_level = sample_arr * Radius + depth_level
                iso_R.append(radius_level)
                iso_pres.append(pres_level)

            iso_R = np.array(iso_R)
            iso_pres = np.array(iso_pres)
            top_pres = np.zeros(len(sample_arr))

            if lev == 0:
                return (top_pres - iso_pres[lev]) * iso_R[lev]
            else:
                return (iso_pres[lev - 1] - iso_pres[lev]) * iso_R[lev]

        else:
            ssh_flatten = surSeaHeight.flatten()
            for i in np.arange(len(levDepth)):
                depth_level = sample_arr * levDepth[i]
                radius_level = sample_arr * Radius + depth_level
                iso_R.append(radius_level)
                if (depth_level[i] - ssh_flatten <= 0).all():
                    pres_level = EOPConstant.grav * EOPConstant.rho_water * depth_level
                    iso_pres.append(pres_level)
                    continue
                index = depth_level[i] - ssh_flatten > 0
                depth_level[index] = ssh_flatten[index]
                pres_level = EOPConstant.grav * EOPConstant.rho_water * depth_level
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



class GRACE_Exciatation:
    def __init__(self, GRACE, OceanSH, GAD, lmax):
        """
        :param GRACE: GRACE here is the SH of mass coefficients;
        :param OceanSH: OceanSH here is the SH of mass coefficients;
        :param GAD: GAD is the non-tidal ocean SH derived from oceanic circulation, and convert it to mass coefficients;
        :param lmax: Max degree
        """
        self.GRACE = SHC(GRACE)
        if OceanSH is not None:
            self.OceanSH = SHC(OceanSH)
        else:
            self.OceanSH = SHC(GAD)
        self.GAD = SHC(GAD)
        self.lmax = lmax
        self.res = 1
        self.lat, self.lon = MathTool.get_global_lat_lon_range(self.res)

        self.LLN_method = Enums.LLN_Data.PREM
        self.frame = Enums.Frame.CF

    def setResolution(self, resolution):
        self.res = resolution
        self.lat, self.lon = MathTool.get_global_lat_lon_range(resolution)
        # print(f"-----------------\n"
        #       f"Setting the processing data resolution is: {resolution} degree\n"
        #       f"The lat is from {self.lat[0]} to {self.lat[-1]},the lon is from {self.lon[0]} to {self.lon[-1]}\n"
        #       f"----------------")
        return self

    def setLatLon(self, lat, lon):
        self.lat, self.lon = lat, lon
        self.res = np.abs(self.lat[1] - self.lat[0])
        # print(f"-----------------\n"
        #       f"Setting the processing data resolution is: {self.res} degree\n"
        #       f"The lat is from {self.lat[0]} to {self.lat[-1]},the lon is from {self.lon[0]} to {self.lon[-1]}."
        #       f"----------------")
        return self

    def setOcean(self, ocean_mask=None):
        if ocean_mask is not None:
            mask_grid = ocean_mask
        else:
            oceanmask_path = FileTool.get_project_dir("data/basin_mask/SH/Ocean_maskSH.dat")
            oceanmask_sh = load_SHC(oceanmask_path, key='', lmax=self.lmax)
            grid_basin = oceanmask_sh.to_grid(grid_space=self.res)
            grid_basin.limiter(threshold=0.5)
            mask_grid = grid_basin.value[0]
        return mask_grid

    def setLoveNumber(self, method: Enums.LLN_Data.PREM, frame: Enums.Frame.CF):
        self.LLN_method = method
        self.frame = frame
        return self

    def I_Matrix_Term(self, mask=None):
        N = len(self.GRACE.value)
        I = np.zeros((N, 6, 6))
        ocean_mask = self.setOcean(ocean_mask=mask)
        theta, phi = MathTool.get_colat_lon_rad(lat=self.lat, lon=self.lon)
        Pilm = MathTool.get_Legendre(lat=theta, lmax=self.lmax, option=0)

        cosC10 = np.cos(0 * phi)
        cosC11 = np.cos(1 * phi)
        sinS11 = np.sin(1 * phi)
        sinS21 = np.cos(1 * phi)
        cosC20 = np.cos(0 * phi)
        cosC21 = np.cos(1 * phi)

        CoreI10C = Pilm[:, 1, 0][:, None] * ocean_mask * cosC10[None, :]
        CoreI11C = Pilm[:, 1, 1][:, None] * ocean_mask * cosC11[None, :]
        CoreI11S = Pilm[:, 1, 1][:, None] * ocean_mask * sinS11[None, :]
        CoreI21S = Pilm[:, 2, 1][:, None] * ocean_mask * sinS21[None, :]
        CoreI20C = Pilm[:, 2, 0][:, None] * ocean_mask * cosC20[None, :]
        CoreI21C = Pilm[:, 2, 1][:, None] * ocean_mask * cosC21[None, :]

        # CoreI30C = Pilm[:, 3, 0][:, None] * ocean_mask * cosC30[None, :]

        I_10C = GRID(grid=CoreI10C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_11C = GRID(grid=CoreI11C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_11S = GRID(grid=CoreI11S, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_21S = GRID(grid=CoreI21S, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_20C = GRID(grid=CoreI20C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_21C = GRID(grid=CoreI21C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)

        I[:, 0, 0], I[:, 0, 1], I[:, 0, 2], I[:, 0, 3], I[:, 0, 4], I[:, 0, 5] = \
            I_10C.value[:, 2], I_11C.value[:, 2], I_11S.value[:, 2], I_20C.value[:, 2], I_21C.value[:, 2], I_21S.value[
                                                                                                           :, 2]

        I[:, 1, 0], I[:, 1, 1], I[:, 1, 2], I[:, 1, 3], I[:, 1, 4], I[:, 1, 5] = \
            I_10C.value[:, 3], I_11C.value[:, 3], I_11S.value[:, 3], I_20C.value[:, 3], I_21C.value[:, 3], I_21S.value[
                                                                                                           :, 3]

        I[:, 2, 0], I[:, 2, 1], I[:, 2, 2], I[:, 2, 3], I[:, 2, 4], I[:, 2, 5] = \
            I_10C.value[:, 1], I_11C.value[:, 1], I_11S.value[:, 1], I_20C.value[:, 1], I_21C.value[:, 1], I_21S.value[
                                                                                                           :, 1]

        I[:, 3, 0], I[:, 3, 1], I[:, 3, 2], I[:, 3, 3], I[:, 3, 4], I[:, 3, 5] = \
            I_10C.value[:, 6], I_11C.value[:, 6], I_11S.value[:, 6], I_20C.value[:, 6], I_21C.value[:, 6], I_21S.value[
                                                                                                           :, 6]

        I[:, 4, 0], I[:, 4, 1], I[:, 4, 2], I[:, 4, 3], I[:, 4, 4], I[:, 4, 5] = \
            I_10C.value[:, 7], I_11C.value[:, 7], I_11S.value[:, 7], I_20C.value[:, 7], I_21C.value[:, 7], I_21S.value[
                                                                                                           :, 7]
        I[:, 5, 0], I[:, 5, 1], I[:, 5, 2], I[:, 5, 3], I[:, 5, 4], I[:, 5, 5] = \
            I_10C.value[:, 5], I_11C.value[:, 5], I_11S.value[:, 5], I_20C.value[:, 5], I_21C.value[:, 7], I_21S.value[
                                                                                                           :, 5]
        I = I
        # print("-------------Finished I Matrix computation-------------")
        return I

    def G_Matrix_Term(self, mask=None):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:, 0:4] = 0
        GRACE_SH[:, 5:8] = 0
        # GRACE_SH[:,12] = 0
        kernal = (SHC(c=GRACE_SH).to_grid(self.res).value) * (self.setOcean(ocean_mask=mask))
        G_SH = GRID(grid=kernal, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        G = np.zeros((len(GRACE_SH), 6))
        G[:, 0] = G_SH[:, 2]
        G[:, 1] = G_SH[:, 3]
        G[:, 2] = G_SH[:, 1]
        G[:, 3] = G_SH[:, 6]
        G[:, 4] = G_SH[:, 7]
        G[:, 5] = G_SH[:, 5]
        G = G
        # print(f"G V2 is: {G[0]}")
        # print("-------------Finished G Matrix computation-------------")
        return G

    def Ocean_Model_Term(self, C10, C11, S11, C20, C21, S21):
        GAD_Correct = self.GAD.value
        OM_SH = self.OceanSH.value
        OM = np.zeros((len(OM_SH), 6))
        # print(OM.shape)
        OM[:, 0] = OM_SH[:, 2] - GAD_Correct[:, 2] + C10
        OM[:, 1] = OM_SH[:, 3] - GAD_Correct[:, 3] + C11
        OM[:, 2] = OM_SH[:, 1] - GAD_Correct[:, 1] + S11
        OM[:, 3] = OM_SH[:, 6] - GAD_Correct[:, 6] + C20
        OM[:, 4] = OM_SH[:, 7] - GAD_Correct[:, 7] + C21
        OM[:, 5] = OM_SH[:, 5] - GAD_Correct[:, 5] + S21
        # print(f"OceanModel Term: {OM[0]}")
        return OM

    def GRD_Term(self, C10=None, C11=None, S11=None, C20=None, C21=None, S21=None, mask=None, GRD=False, rotation=True):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:, 1] = S11
        GRACE_SH[:, 2] = C10
        GRACE_SH[:, 3] = C11
        GRACE_SH[:, 6] = C20
        GRACE_SH[:, 7] = C21
        GRACE_SH[:, 5] = S21
        # GRACE_SH[:,]
        GRACE_SH = SHC(c=GRACE_SH).convert_type(from_type=Enums.PhysicalDimensions.Density,
                                                to_type=Enums.PhysicalDimensions.EWH)
        GRACE_GRID = GRACE_SH.to_grid(self.res)
        if GRD:
            SLE = PseudoSpectralSLE(SH=GRACE_SH.value, lmax=self.lmax)
            SLE.setLatLon(lat=self.lat, lon=self.lon)
            SLE.setLoveNumber(lmax=self.lmax, method=self.LLN_method, frame=self.frame)
            UpdateTerm = SLE.SLE(mask=mask, rotation=rotation)['RSL_SH']
        else:
            ocean_mask = self.setOcean(ocean_mask=mask)
            land_mask = 1 - ocean_mask
            OceanArea = MathTool.get_acreage(basin=ocean_mask)
            uniform_value = GRACE_GRID.integral(mask=land_mask, average=False) / OceanArea
            uniform_mask = uniform_value[:, None, None] * ocean_mask
            UpdateTerm = GRID(grid=uniform_mask, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        UpdateTerm = SHC(c=UpdateTerm).convert_type(from_type=Enums.PhysicalDimensions.EWH,
                                                    to_type=Enums.PhysicalDimensions.Density).value
        return UpdateTerm[:, 2], UpdateTerm[:, 3], UpdateTerm[:, 1], UpdateTerm[:, 6], UpdateTerm[:, 7], UpdateTerm[:,5]

    def Low_Degree_Term(self, mask=None, GRD=False, rotation=True):
        """
        the series of Stokes coefficients follow: C10, C11, S11, C20, C21, S21
        that means, index 0->C10, 1->C11, 2->S11, 3->C20, 4->C21, 5->S21
        """
        print(f"=========Begin GRACE Degree Terms computing==========")
        start_time = time.time()
        GRACE_SH = self.GRACE.value
        I_C10, I_C11, I_S11, I_C20, I_C21, I_S21 = [np.zeros(len(GRACE_SH))] * 6
        OM = self.Ocean_Model_Term(C10=I_C10, C11=I_C11, S11=I_S11, C20=I_C20, C21=I_C21, S21=I_S21)
        I = self.I_Matrix_Term(mask=mask)
        G = self.G_Matrix_Term(mask=mask)

        I_inv = np.linalg.inv(I)
        # print(f"I and I_inv:\n{I[0]}\n\n{I_inv[0]}")
        # print(f"verfiy: {I[0]@I_inv[0]}")
        C = np.einsum('nij,nj->ni', I_inv, OM - G)

        GRD_Ocean_Term = self.GRD_Term(C10=C[:, 0], C11=C[:, 1], S11=C[:, 2], C20=C[:, 3], C21=C[:, 4], S21=C[:, 5],
                                       mask=mask, GRD=GRD, rotation=rotation)
        for iter in np.arange(100):
            OM_new = self.Ocean_Model_Term(C10=GRD_Ocean_Term[0], C11=GRD_Ocean_Term[1], S11=GRD_Ocean_Term[2],
                                           C20=GRD_Ocean_Term[3], C21=GRD_Ocean_Term[4], S21=GRD_Ocean_Term[5])
            C_new = np.einsum('nij,nj->ni', I_inv, OM_new - G)
            delta = np.abs(C_new - C).flatten()
            if np.max(delta) < 10e-4:
                print(f"Iterative number is: {iter + 1}")
                break
            C = C_new

        lln = LoveNumber().config(lmax=self.lmax, method=self.LLN_method).get_Love_number()
        lln.convert(target=self.frame)
        k = lln.LLN[Enums.LLN_variable.k]

        factor = 1.021 / (EOPConstant.rho_earth * EOPConstant.radius)
        factor2 = (3 + 3 * k[2]) / (5 * EOPConstant.rho_earth * EOPConstant.radius)
        # factor3 = (3+3*k[3])/(7*EarthConstant.rhoear*EarthConstant.radiusm)

        print(f"Love numbers degree-1:{k[1]},degre-2:{k[2]},degree-3:{k[3]}")
        Mass_Coef = {"C10": C[:, 0], "C11": C[:, 1], "S11": C[:, 2], "C20": C[:, 3], "C21": C[:, 4], "S21": C[:, 5]}
        Stokes_Coef = {"C10": C[:, 0] * factor, "C11": C[:, 1] * factor, "S11": C[:, 2] * factor,
                       "C20": C[:, 3] * factor2, "C21": C[:, 4] * factor2, "S21": C[:, 5] * factor2}
        EWH_Coef = {"C10": C[:, 0] / EOPConstant.rho_water, "C11": C[:, 1] / EOPConstant.rho_water,
                    "S11": C[:, 2] / EOPConstant.rho_water,
                    "C20": C[:, 3] / EOPConstant.rho_water, "C21": C[:, 4] / EOPConstant.rho_water,
                    "S21": C[:, 5] / EOPConstant.rho_water}

        SH = {"Mass": Mass_Coef, "Stokes": Stokes_Coef, "EWH": EWH_Coef}
        end_time = time.time()
        print('---------------------------------------------------\n'
              'GRACE-OBP: GCM, C20, C21, and S21 estimation\n'
              '---------------------------------------------------')
        print('%-20s%-20s ' % ('Maxdegree:', f'{self.lmax}'))
        print('%-20s%-20s ' % ('Resolution:', f'{self.res}°'))
        print('%-20s%-20s ' % ('LoveNumber:', f'{self.LLN_method}'))
        print('%-20s%-20s ' % ('Frame:', f'{self.frame}'))
        print("%-20s%-20s " % ('SAL:', f'{GRD} (if False, omit rotation)'))
        print("%-20s%-20s " % ('Rotation feedback:', f'{rotation}'))
        print('%-20s%-20s ' % ('Iteration:', f'{iter + 1}'))
        print('%-20s%-20s ' % ('Convergence:', f'{np.max(delta)}'))
        print('%-20s%-20s ' % ('Time-consuming:', f'{end_time - start_time:.4f} s'))
        print(f"---------------------------------------------------")
        return SH

    def Convert_Mass_to_Coordinates(self,C10,C11,S11):
        k1 = 0.021
        rho_earth = EOPConstant.rho_earth
        X = np.sqrt(3) * (1 + k1) * C11 / rho_earth
        Y = np.sqrt(3) * (1 + k1) * S11 / rho_earth
        Z = np.sqrt(3) * (1 + k1) * C10 / rho_earth
        Coordinate = {"X": X, "Y": Y, "Z": Z}
        return Coordinate

    def __PM(self,C21, S21, isMas=False):
        """
        :param C21: EWH Stoke coefficient
        :param S21: EWH Stoke coefficient
        :return: unit is mas (1 mas = 3 cm)
        """
        # unit_scale = 1
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180 * 3600 * 1000 / np.pi
        C = EOPConstant.Cm
        A = EOPConstant.Am
        factor = -4 * np.pi * (EOPConstant.radius ** 4) * EOPConstant.rho_water / (np.sqrt(15))
        I13 = factor * C21
        I23 = factor * S21
        chi1 = rad_to_mas * I13 / (C - A)
        chi2 = rad_to_mas * I23 / (C - A)
        chi = {"chi1": chi1, "chi2": chi2}
        return chi

    def GSM_Like(self, mask=None, GRD=False, rotation=True):
        SH = self.Low_Degree_Term(mask=mask, GRD=GRD, rotation=rotation)
        C = SH['Mass']
        Coordinate = self.Convert_Mass_to_Coordinates(C10=C["C10"], C11=C["C11"], S11=C["S11"])
        print("-------------Finished GSM-like computation-------------\n"
              "==========================================================")
        return Coordinate

    def PM_mass_term(self, mask=None, GRD=False, rotation=True,isMas=False):
        SH = self.Low_Degree_Term(mask=mask, GRD=GRD, rotation=rotation)
        C = SH['EWH']
        excitation = self.__PM(C21=C['C21'], S21=C['S21'],isMas=isMas)
        print("-------------Finished PM computation-------------\n"
              "==========================================================")
        return excitation


