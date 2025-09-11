import numpy as np
# from SaGEA.auxiliary.preference.Constants import PMConstant
from pysrc.ancillary.constant.GeoConstant import PMConstant
import time
import SaGEA.auxiliary.preference.EnumClasses as Enums
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from pysrc.ancillary.load_file.DataClass import SHC,GRID
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.sealevel_equation.SeaLevelEquation import PseudoSpectralSLE
from pysrc.ancillary.geotools.LLN import LoveNumber
from pysrc.ancillary.constant.Setting import EAMType

class EOP_fast:
    def __init__(self):
        self.factor_PM_mass = -4*np.pi*(PMConstant.radius**4)*PMConstant.rho_water/(np.sqrt(15))
        self.factor_PM_motion = (PMConstant.Cm-PMConstant.Am)*PMConstant.omega*(1-PMConstant.k2/PMConstant.ks)
        self.factor_LOD_mass = 2*0.756*(PMConstant.radius**2)*PMConstant.Mass/(3*(1+PMConstant.k2_load)*PMConstant.Cm)
        self.factor_LOD_mass_grid = 0.756*(PMConstant.radius**4)/(PMConstant.Cm*PMConstant.grav)
        self.factor_LOD_motion = 0.998/(PMConstant.Cm*PMConstant.omega)
        pass

    def PM_mass_term(self,SH,isMas=False):
        """
        :param SH: SH here means EWH harmonic coefficients
        :param isMas: if True, the units of chi1 and chi2 are mas, otherwise is rad
        :return: chi1 and chi2
        """
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180*3600*1000/np.pi
        I13 = self.factor_PM_mass*SH[:,7]
        I23 = self.factor_PM_mass*SH[:,5]

        chi1 = rad_to_mas*I13/(PMConstant.Cm-PMConstant.Am)
        chi2 = rad_to_mas*I23/(PMConstant.Cm-PMConstant.Am)
        chi = {"chi1":chi1,"chi2":chi2}
        return chi
    def __compute_dp_oam(self,z_depth,ssh,lev):


        ssh_flatten = ssh.flatten()
        iso_pres = []

        for i in np.arange(len(z_depth)):
            pres_level = np.ones(len(ssh_flatten))*z_depth[i]
            if (ssh_flatten-pres_level >= 0).all():
                iso_pres.append(pres_level)
                continue
            index = ssh_flatten-pres_level<0
            pres_level[index] = ssh_flatten[index]
            iso_pres.append(pres_level)
        iso_pres = np.array(iso_pres)
        if lev == 0:
            return PMConstant.grav*PMConstant.rho_water*(ssh_flatten-iso_pres[lev])
        else:
            return PMConstant.grav*PMConstant.rho_water*(iso_pres[lev-1]-iso_pres[lev])

    def __compute_dp_aam(self,lp,sp,lev):
        sp_flatten = sp.flatten()
        iso_pres = []
        for j in np.arange(len(lp)):
            pres_level = np.ones(len(sp_flatten))*lp[j]
            if (lp[j]-sp_flatten<0).all():
                iso_pres.append(pres_level)
                continue
            index = lp[j]-sp_flatten >0
            pres_level[index] = sp_flatten[index]
            iso_pres.append(pres_level)

        iso_pres = np.array(iso_pres)
        # print(f"new iso_pres is:{iso_pres.shape}")
        level_number = len(iso_pres)

        if lev == (level_number-1):
            return sp_flatten-iso_pres[lev]
        else:
            return iso_pres[lev+1]-iso_pres[lev]

    def PM_motion_term(self,u_speed,v_speed,lat,lon,layer,surf,type=EAMType.AAM,isMas=True):
        """
        :param u_speed: eastward along latitude (also known as zonal wind)
        :param v_speed: northward along longitude (also known as meridional wind)
        Notes: both u and v shape is (time,layer,lat,lon)
        :param lat: the range is from 90--90
        :param lon: the range is from 0-360
        :param pressure: multi-layer pressure data, make sure the unit is Pa (not hPa)
        :param isMas: False means results are rad, True means results are mas
        :return:
        """
        rad_to_mas = 1
        if isMas:
            rad_to_mas = (180 / np.pi) * 3600 * 1000

        dp_g = []
        R = PMConstant.radius

        if type == EAMType.AAM:
            for lev in np.arange(len(layer)):
                temp_dp_g = self.__compute_dp_aam(lev=lev,lp=layer,sp=surf)/PMConstant.grav
                dp_g.append(temp_dp_g)
            dp_g = np.array(dp_g)
            dp_g = dp_g.reshape(len(surf),len(dp_g),len(lat),len(lon))
            # print(f"dp_g shape is{dp_g.shape}")
        else:
            # dp_g = self.__compute_dp_oam(z_depth=layer)/PMConstant.grav
            # dp_g = dp_g[None, :, None, None]
            for lev in np.arange(len(layer)):
                temp_dp_g = self.__compute_dp_oam(lev=lev, z_depth=layer, ssh=surf) / PMConstant.grav
                dp_g.append(temp_dp_g)
            dp_g = np.array(dp_g)
            dp_g = dp_g.reshape(len(surf), len(dp_g), len(lat), len(lon))
            # dp_g = dp_g[None, :, :, :]

        dU = u_speed*dp_g
        dV = v_speed*dp_g

        U,V = np.sum(dU,axis=1),np.sum(dV,axis=1)

        phi = np.deg2rad(lat)
        lam = np.deg2rad(lon)
        dphi = np.abs(phi[1] - phi[0])
        dlam = np.abs(lam[1] - lam[0])

        phi_grid, lam_grid = np.meshgrid(phi, lam, indexing="ij")
        cos_phi,sin_phi = np.cos(phi_grid)[None,:,:],np.sin(phi_grid)[None,:,:]
        cos_lam, sin_lam = np.cos(lam_grid)[None,:,:],np.sin(lam_grid)[None,:,:]

        dA = cos_phi * dlam * dphi

        L1 = (-U*sin_phi*cos_lam+V*sin_lam)
        L2 = (-U*sin_phi*sin_lam-V*cos_lam)

        H1 = R*(PMConstant.radius ** 2) * np.sum(L1 * dA, axis=(1, 2))
        H2 = R*(PMConstant.radius ** 2) * np.sum(L2 * dA, axis=(1, 2))

        chi1 = H1 / self.factor_PM_motion
        chi2 = H2 / self.factor_PM_motion

        chi1 = chi1 * rad_to_mas
        chi2 = chi2 * rad_to_mas
        chi = {"chi1": chi1, "chi2": chi2}
        return chi

    def LOD_mass_term(self,SH,isMas=False):
        """
        :param SH: Different with PM EWH term, the type of LOD is Stokes coefficients.
        :param isMas: the same follow before.
        :return: chi3 with unit mas or rad, the unit of (Delta)LOD is seconds (s).
        """
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180 * 3600 * 1000 / np.pi

        C00 = SH[:,0]
        C20 = SH[:,6]
        chi3 = self.factor_LOD_mass*(C00-np.sqrt(5)*C20)
        delta_LOD = chi3*PMConstant.LOD
        chi3 = rad_to_mas*chi3
        LOD = {"chi3":chi3,"LOD":delta_LOD}
        return LOD
    def LOD_mass_term_grid(self,Pressure,lat,lon,isMas=True):
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180 * 3600 * 1000 / np.pi

        phi,lam = np.deg2rad(lat),np.deg2rad(lon)
        dphi,dlam = np.abs(phi[1]-phi[0]),np.abs(lam[1]-lam[2])
        phi_grid,lam_grid = np.meshgrid(phi,lam,indexing="ij")

        cos_phi = np.cos(phi_grid)
        dA = cos_phi*dphi*dlam

        kernal = Pressure*cos_phi*cos_phi*dA
        chi3 = np.sum(self.factor_LOD_mass_grid*kernal,axis=(1,2))
        delta_LOD = chi3*PMConstant.LOD
        chi3 = chi3*rad_to_mas
        LOD = {"chi3":chi3,"LOD":delta_LOD}
        return LOD
    def LOD_motion_term(self,u_speed,lat,lon,layer,surf,type=EAMType.AAM,isMas=True):
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180 * 3600 * 1000 / np.pi

        dp_g = []
        if type == EAMType.AAM:
            for lev in np.arange(len(layer)):
                temp_dp_g = self.__compute_dp_aam(lev=lev, lp=layer, sp=surf) / PMConstant.grav
                dp_g.append(temp_dp_g)
            dp_g = np.array(dp_g)
            dp_g = dp_g.reshape(len(surf),len(dp_g), len(lat), len(lon))
            # dp_g = dp_g[None, :, :, :]
        else:
            for lev in np.arange(len(layer)):
                temp_dp_g = self.__compute_dp_oam(lev=lev, z_depth=layer, ssh=surf) / PMConstant.grav
                dp_g.append(temp_dp_g)
            dp_g = np.array(dp_g)
            dp_g = dp_g.reshape(len(surf),len(dp_g), len(lat), len(lon))
            # dp_g = dp_g[None, :, :, :]

        dU = u_speed*dp_g
        U = np.sum(dU,axis=1)

        phi,lam = np.deg2rad(lat),np.deg2rad(lon)
        dphi,dlam = np.abs(phi[1]-phi[0]),np.abs(lam[1]-lam[0])

        phi_grid,lam_grid = np.meshgrid(phi,lam,indexing='ij')
        cos_phi = np.cos(phi_grid)[None,:,:]
        dA = cos_phi*dphi*dlam

        h3 = (PMConstant.radius**3)*np.sum(U*dA,axis=(1,2))
        chi3 = self.factor_LOD_motion*h3
        delta_LOD = chi3*PMConstant.LOD
        chi3 = rad_to_mas*chi3
        LOD = {"chi3":chi3,"LOD":delta_LOD}
        return LOD

class EOP:
    def __init__(self):
        self.factor_PM_mass = -4 * np.pi * (PMConstant.radius ** 4) * PMConstant.rho_water / (np.sqrt(15))
        self.factor_PM_motion = (PMConstant.Cm - PMConstant.Am) * PMConstant.omega * (1 - PMConstant.k2 / PMConstant.ks)
        self.factor_LOD_mass = 2 * 0.756 * (PMConstant.radius ** 2) * PMConstant.Mass / (
                    3 * (1 + PMConstant.k2_load) * PMConstant.Cm)
        self.factor_LOD_mass_grid = 0.756 * (PMConstant.radius ** 4) / (PMConstant.Cm * PMConstant.grav)
        self.factor_LOD_motion = 0.998 / (PMConstant.Cm * PMConstant.omega)
        pass

    def PM_mass_term(self, SH, isMas=False):
        """
        :param SH: SH here means EWH harmonic coefficients
        :param isMas: if True, the units of chi1 and chi2 are mas, otherwise is rad
        :return: chi1 and chi2
        """
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180 * 3600 * 1000 / np.pi
        I13 = self.factor_PM_mass * SH[:, 7]
        I23 = self.factor_PM_mass * SH[:, 5]

        chi1 = rad_to_mas * I13 / (PMConstant.Cm - PMConstant.Am)
        chi2 = rad_to_mas * I23 / (PMConstant.Cm - PMConstant.Am)
        chi = {"chi1": chi1, "chi2": chi2}
        return chi
    def __compute_dp_oam(self, z_depth, ssh, lev):
        '''
        :param z_depth: z_depth here follows the order from large to small
        :param ssh:
        :param lev:
        :return: dz multies water density and gravity constant equals pressure difference
        '''
        ssh_flatten = ssh.flatten()
        iso_pres = []
        for i in np.arange(len(z_depth)):
            pres_level = np.ones(len(ssh_flatten)) * z_depth[i]
            if (z_depth[i]-ssh_flatten<=0).all():
                iso_pres.append(pres_level)
                continue
            index = z_depth[i]-ssh_flatten>0
            pres_level[index] = ssh_flatten[index]
            iso_pres.append(pres_level)
        iso_pres = np.array(iso_pres)
        # print(f"new iso_pres is:{iso_pres.shape}")
        # level_number = len(iso_pres)

        if lev == 0:
            return PMConstant.grav * PMConstant.rho_water *(ssh_flatten - iso_pres[lev])
        else:
            return PMConstant.grav * PMConstant.rho_water *(iso_pres[lev-1] - iso_pres[lev])

        # multi_pres = []
        # ssh_flatten = ssh.reshape(len(ssh), -1)
        # for j in np.arange(len(ssh)):
        #     ssh_single = ssh[j].flatten()
        #     iso_pres = []
        #
        #     for i in np.arange(len(z_depth)):
        #         pres_level = np.ones(len(ssh_single)) * z_depth[i]
        #         if (ssh_single - pres_level >= 0).all():
        #             iso_pres.append(pres_level)
        #             continue
        #         index = ssh_single - pres_level < 0
        #         pres_level[index] = ssh_single[index]
        #         iso_pres.append(pres_level)
        #     iso_pres = np.array(iso_pres)
        #     multi_pres.append(iso_pres)
        #
        # multi_pres = np.array(multi_pres)
        # if lev == 0:
        #     return PMConstant.grav * PMConstant.rho_water * (ssh_flatten[:, :] - multi_pres[:, lev, :])
        # else:
        #     return PMConstant.grav * PMConstant.rho_water * (multi_pres[:, lev - 1, :] - multi_pres[:, lev, :])
    def __compute_dp_aam(self, lp, sp, lev):
        sp_flatten = sp.flatten()
        iso_pres = []
        for j in np.arange(len(lp)):
            pres_level = np.ones(len(sp_flatten)) * lp[j]
            if (lp[j] - sp_flatten <= 0).all():
                iso_pres.append(pres_level)
                continue
            index = lp[j] - sp_flatten > 0
            pres_level[index] = sp_flatten[index]
            iso_pres.append(pres_level)

        iso_pres = np.array(iso_pres)
        # level_number = len(iso_pres)

        if lev == 0:
            return sp_flatten - iso_pres[lev]
        else:
            return iso_pres[lev-1] - iso_pres[lev]

        # if lev == (level_number - 1):
        #     return sp_flatten - iso_pres[lev]
        # else:
        #     return iso_pres[lev + 1] - iso_pres[lev]

    def PM_motion_term(self, u_speed, v_speed, lat, lon, layer, surf, type=EAMType.AAM, isMas=True):
        """
        :param u_speed: eastward along latitude (also known as zonal wind)
        :param v_speed: northward along longitude (also known as meridional wind)
        Notes: both u and v shape is (time,layer,lat,lon)
        :param lat: the range is from 90--90
        :param lon: the range is from 0-360
        :param pressure: multi-layer pressure data, make sure the unit is Pa (not hPa)
        :param isMas: False means results are rad, True means results are mas
        :return:
        """
        rad_to_mas = 1
        if isMas:
            rad_to_mas = (180 / np.pi) * 3600 * 1000
        R = PMConstant.radius
        chi1_series,chi2_series = [],[]
        for i in np.arange(len(u_speed)):
            dp_g = []
            if type == EAMType.AAM:
                for lev in np.arange(len(layer)):
                    temp_dp_g = self.__compute_dp_aam(lev=lev, lp=layer, sp=surf[i]) / PMConstant.grav
                    dp_g.append(temp_dp_g)
                dp_g = np.array(dp_g)
                dp_g = dp_g.reshape(len(dp_g), len(lat), len(lon))
            else:
                for lev in np.arange(len(layer)):
                    temp_dp_g = self.__compute_dp_oam(lev=lev, z_depth=layer, ssh=surf[i]) / PMConstant.grav
                    dp_g.append(temp_dp_g)
                dp_g = np.array(dp_g)
                dp_g = dp_g.reshape(len(dp_g), len(lat), len(lon))

            dU = u_speed[i] * dp_g
            dV = v_speed[i] * dp_g

            U, V = np.sum(dU, axis=0), np.sum(dV, axis=0)

            phi = np.deg2rad(lat)
            lam = np.deg2rad(lon)
            dphi = np.abs(phi[1] - phi[0])
            dlam = np.abs(lam[1] - lam[0])

            phi_grid, lam_grid = np.meshgrid(phi, lam, indexing="ij")
            cos_phi, sin_phi = np.cos(phi_grid), np.sin(phi_grid)
            cos_lam, sin_lam = np.cos(lam_grid), np.sin(lam_grid)

            dA = cos_phi * dlam * dphi

            L1 = (-U * sin_phi * cos_lam + V * sin_lam)
            L2 = (-U * sin_phi * sin_lam - V * cos_lam)

            H1 = R * (PMConstant.radius ** 2) * np.sum(L1 * dA, axis=(0, 1))
            H2 = R * (PMConstant.radius ** 2) * np.sum(L2 * dA, axis=(0, 1))

            chi1 = H1 / self.factor_PM_motion
            chi2 = H2 / self.factor_PM_motion

            chi1 = chi1 * rad_to_mas
            chi2 = chi2 * rad_to_mas
            chi1_series.append(chi1)
            chi2_series.append(chi2)

        chi1_series,chi2_series = np.array(chi1_series),np.array(chi2_series)
        chi = {"chi1": chi1_series, "chi2": chi2_series}
        return chi

    def LOD_mass_term(self, SH, isMas=False):
        """
        :param SH: Different with PM EWH term, the type of LOD is Stokes coefficients.
        :param isMas: the same follow before.
        :return: chi3 with unit mas or rad, the unit of (Delta)LOD is seconds (s).
        """
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180 * 3600 * 1000 / np.pi

        C00 = SH[:, 0]
        C20 = SH[:, 6]
        chi3 = self.factor_LOD_mass * (C00 - np.sqrt(5) * C20)
        delta_LOD = chi3 * PMConstant.LOD
        chi3 = rad_to_mas * chi3
        LOD = {"chi3": chi3, "LOD": delta_LOD}
        return LOD

    def LOD_mass_term_grid(self, Pressure, lat, lon, isMas=True):
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180 * 3600 * 1000 / np.pi

        phi, lam = np.deg2rad(lat), np.deg2rad(lon)
        dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[2])
        phi_grid, lam_grid = np.meshgrid(phi, lam, indexing="ij")

        cos_phi = np.cos(phi_grid)
        dA = cos_phi * dphi * dlam

        kernal = Pressure * cos_phi * cos_phi * dA
        chi3 = np.sum(self.factor_LOD_mass_grid * kernal, axis=(1, 2))
        delta_LOD = chi3 * PMConstant.LOD
        chi3 = chi3 * rad_to_mas
        LOD = {"chi3": chi3, "LOD": delta_LOD}
        return LOD

    def LOD_motion_term(self, u_speed, lat, lon, layer, surf, type=EAMType.AAM, isMas=True):
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180 * 3600 * 1000 / np.pi

        R = PMConstant.radius
        chi3_series,LOD_series = [],[]
        for i in np.arange(len(u_speed)):
            dp_g = []
            if type == EAMType.AAM:
                for lev in np.arange(len(layer)):
                    temp_dp_g = self.__compute_dp_aam(lev=lev, lp=layer, sp=surf[i]) / PMConstant.grav
                    dp_g.append(temp_dp_g)
                dp_g = np.array(dp_g)
                dp_g = dp_g.reshape(len(dp_g), len(lat), len(lon))
                # dp_g = dp_g[None, :, :, :]
            else:
                for lev in np.arange(len(layer)):
                    temp_dp_g = self.__compute_dp_oam(lev=lev, z_depth=layer, ssh=surf[i]) / PMConstant.grav
                    dp_g.append(temp_dp_g)
                dp_g = np.array(dp_g)
                dp_g = dp_g.reshape(len(dp_g), len(lat), len(lon))
                # dp_g = dp_g[None, :, :, :]

            dU = u_speed[i] * dp_g
            U = np.sum(dU, axis=0)

            phi, lam = np.deg2rad(lat), np.deg2rad(lon)
            dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[0])

            phi_grid, lam_grid = np.meshgrid(phi, lam, indexing='ij')
            cos_phi = np.cos(phi_grid)
            dA = cos_phi * dphi * dlam

            h3 = (PMConstant.radius ** 3) * np.sum(U * dA, axis=(0, 1))
            chi3 = self.factor_LOD_motion * h3
            delta_LOD = chi3 * PMConstant.LOD
            chi3 = rad_to_mas * chi3
            chi3_series.append(chi3)
            LOD_series.append(delta_LOD)
        chi3_series,LOD_series = np.array(chi3_series),np.array(LOD_series)
        LOD = {"chi3": chi3_series, "LOD": LOD_series}
        return LOD



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

        factor = 1.021 / (PMConstant.rho_earth * PMConstant.radius)
        factor2 = (3 + 3 * k[2]) / (5 * PMConstant.rho_earth * PMConstant.radius)
        # factor3 = (3+3*k[3])/(7*EarthConstant.rhoear*EarthConstant.radiusm)

        print(f"Love numbers degree-1:{k[1]},degre-2:{k[2]},degree-3:{k[3]}")
        Mass_Coef = {"C10": C[:, 0], "C11": C[:, 1], "S11": C[:, 2], "C20": C[:, 3], "C21": C[:, 4], "S21": C[:, 5]}
        Stokes_Coef = {"C10": C[:, 0] * factor, "C11": C[:, 1] * factor, "S11": C[:, 2] * factor,
                       "C20": C[:, 3] * factor2, "C21": C[:, 4] * factor2, "S21": C[:, 5] * factor2}
        EWH_Coef = {"C10": C[:, 0] / PMConstant.rho_water, "C11": C[:, 1] / PMConstant.rho_water,
                    "S11": C[:, 2] / PMConstant.rho_water,
                    "C20": C[:, 3] / PMConstant.rho_water, "C21": C[:, 4] / PMConstant.rho_water,
                    "S21": C[:, 5] / PMConstant.rho_water}

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
        rho_earth = PMConstant.rho_earth
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
        C = PMConstant.Cm
        A = PMConstant.Am
        factor = -4 * np.pi * (PMConstant.radius ** 4) * PMConstant.rho_water / (np.sqrt(15))
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


def demo1():
    from datetime import date
    from SaGEA.auxiliary.aux_tool.FileTool import FileTool
    from SaGEA.auxiliary.aux_tool.MathTool import MathTool
    from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
    import SaGEA.auxiliary.preference.EnumClasses as Enums
    lmax = 60
    begin_date, end_date = date(2010, 1, 1), date(2010, 12, 31)
    gad_dir, gad_key = FileTool.get_project_dir("I:\GFZ\GAB\GFZ_GFZ-Release-06_GAX_products_GAB/"), "gfc"
    shc_gad = load_SHC(gad_dir, key=gad_key, lmax=lmax, begin_date=begin_date, end_date=end_date)
    shc_gad.de_background()
    SH = shc_gad.value
    SH[:, 0] = 0
    A = EOP()
    LOD = A.LOD_mass_term(SH=SH, isMas=False)
    print(LOD['LOD'])
    print(LOD['chi3'])
    print(f"-----------------------------")

    res = 0.5
    shc_gad.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,to_type=Enums.PhysicalDimensions.Pressure)
    grid_gad = shc_gad.to_grid(grid_space=res)
    lat,lon = MathTool.get_global_lat_lon_range(res)

    LOD2 = A.LOD_mass_term_grid(Pressure=grid_gad.value,lat=lat,lon=lon,isMas=False)
    print(LOD2['LOD'])
    print(LOD2['chi3'])

def demo2():
    import xarray as xr
    import pandas as pd
    from tqdm import tqdm
    u_set, v_set ,sp_set = [], [],[]
    date_range = pd.date_range(start='2009-01-01', end='2009-12-31', freq="MS").strftime("%Y%m").tolist()
    for i in tqdm(date_range):
        sp_temp = xr.open_dataset(f"I:\ERA5\MAD/2009/sp-{i}.nc")
        u_temp = xr.open_dataset(f"I:\ERA5\MAD/2009/u_wind-{i}.nc")
        v_temp = xr.open_dataset(f"I:\ERA5\MAD/2009/v_wind-{i}.nc")
        u_set.append(u_temp['u'].values[0])
        v_set.append(v_temp['v'].values[0])
        sp_set.append(sp_temp['sp'].values[0])
    # u_wind = u_set['u'].values
    # v_wind = v_set['v'].values
    pressure = u_temp['pressure_level'].values[::-1] * 100
    lats = u_temp['latitude'].values
    lons = u_temp['longitude'].values

    sp_set = np.array(sp_set)
    u_set = np.array(u_set)[:,::-1,:,:]
    v_set = np.array(v_set)[:,::-1,:,:]
    u_mean = np.mean(u_set, axis=0)
    v_mean = np.mean(v_set, axis=0)

    u_set = u_set - u_mean[None, :, :, :]
    v_set = v_set - v_mean[None, :, :, :]

    chi = EOP().PM_motion_term(u_speed=u_set, v_speed=v_set, lat=lats, lon=lons,
                               multi_press=pressure,surf_press=sp_set, isMas=True)
    print(chi['chi1'])

def demo3():
    import xarray as xr
    import pandas as pd
    from tqdm import tqdm
    u_set,  sp_set = [], []
    date_range = pd.date_range(start='2009-01-01', end='2009-12-31', freq="MS").strftime("%Y%m").tolist()
    for i in tqdm(date_range):
        sp_temp = xr.open_dataset(f"I:\ERA5\MAD/2009/sp-{i}.nc")
        u_temp = xr.open_dataset(f"I:\ERA5\MAD/2009/u_wind-{i}.nc")
        u_set.append(u_temp['u'].values[0])
        sp_set.append(sp_temp['sp'].values[0])

    pressure = u_temp['pressure_level'].values[::-1] * 100

    lats = u_temp['latitude'].values
    lons = u_temp['longitude'].values

    sp_set = np.array(sp_set)
    u_set = np.array(u_set)[:, ::-1, :, :]

    u_set = np.array(u_set)

    u_mean = np.mean(u_set, axis=0)


    u_set = u_set - u_mean[None, :, :, :]


    chi = EOP().LOD_motion_term(u_speed=u_set,lat=lats, lon=lons,
                                multi_press=pressure, surf_press=sp_set, isMas=True)
    print(chi['chi3'])

def demo4():
    a = [4,6,7,8]
    a = EOP().compute_dp_oam(multi=a)


if __name__ =="__main__":
    demo4()
