import numpy as np
import time
from pysrc.Auxiliary.Constants import PMConstant

class EOP:
    def __init__(self):
        self.factor_PM_mass = -4*np.pi*(PMConstant.radius**4)*PMConstant.rho_water/(np.sqrt(15))
        self.factor_PM_motion = (PMConstant.Cm-PMConstant.Am)*PMConstant.omega*(1-PMConstant.k2/PMConstant.ks)
        self.factor_LOD_mass = None
        self.factor_LOD_motion = None
        pass

    def PM_mass_term(self,SH,isMas=True):
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
    def PM_motion_term(self,u_speed,v_speed,lat,lon,pressure,isMas=True):
        """
        :param u_speed: latitude velocity (also known as zonal velocity)
        :param v_speed: longitude velocity. Notes: both u and v shape is (time,layer,lat,lon)
        :param lat: the range is from 90--90
        :param lon: the range is from 0-360
        :param pressure: multi-layer pressure data
        :param isMas: False means results are rad, True means results are mas
        :return:
        """
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180*3600*1000/np.pi

        dp = np.zeros_like(pressure)
        dp[0] = pressure[0]-(pressure[0]+pressure[1])/2
        for k in np.arange(1,len(pressure)-1):
            dp[k] = (pressure[k-1]+pressure[k+1])/2
        dp[-1] = (pressure[-2]+pressure[-1])/2-pressure[-1]

        phi = np.deg2rad(lat)
        lam = np.deg2rad(lon)
        dphi = np.abs(phi[1]-phi[0])
        dlam = np.abs(lam[1]-lam[0])

        phi_grid,lam_grid = np.meshgrid(phi,lam,indexing='ij')
        cos_phi = np.cos(phi_grid)
        cos_phi_4d = cos_phi[None,None,:,:]
        sin_lam = np.sin(lam)
        cos_lam = np.cos(lam)
        dA = cos_phi*dlam*dphi

        u_sin_lam = u_speed*sin_lam
        u_cos_lam = u_speed*cos_lam
        v_sin_lam = v_speed*sin_lam
        v_cos_lam = v_speed*cos_lam

        dg_g = (dp/PMConstant.grav)[np.newaxis,:,np.newaxis,np.newaxis]
        dL1 = (u_sin_lam+v_cos_lam)*cos_phi_4d*dg_g
        dL2 = (u_cos_lam-v_sin_lam)*cos_phi_4d*dg_g
        L1_vert = np.sum(dL1,axis=1)
        L2_vert = np.sum(dL2,axis=2)

        h1 = (PMConstant.radius**3)*np.sum(L1_vert*dA[None,:,:],axis=(1,2))
        h2 = (PMConstant.radius**3)*np.sum(L2_vert*dA[None,:,:],axis=(1,2))

        chi1 = h1/self.factor_PM_motion
        chi2 = h2/self.factor_PM_motion

        chi1 = rad_to_mas*chi1
        chi2 = rad_to_mas*chi2

        chi = {"chi1":chi1,"chi2":chi2}
        return chi

    def LOD_mass_term(self,SH,isMas=True):
        pass
    def LOD_motion_term(self,u_speed,lat,lon,pressure,isMas=True):
        pass


