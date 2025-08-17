import numpy as np
from pysrc.Auxiliary.Constants import PMConstant

class EOP:
    def __init__(self):
        self.factor_PM_mass = -4*np.pi*(PMConstant.radius**4)*PMConstant.rho_water/(np.sqrt(15))
        self.factor_PM_motion = (PMConstant.Cm-PMConstant.Am)*PMConstant.omega*(1-PMConstant.k2/PMConstant.ks)
        self.factor_LOD_mass = 2*0.756*(PMConstant.radius**2)*PMConstant.Mass/(3*(1+PMConstant.k2_load)*PMConstant.Cm)
        self.factor_LOD_mass_grid = 0.756*(PMConstant.radius**4)/(PMConstant.Cm*PMConstant.grav)
        self.factor_LOD_motion = 0.998/(PMConstant.Cm*PMConstant.omega)
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
        :param pressure: multi-layer pressure data, make sure the unit is Pa (not hPa)
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
        """
        :param SH: Different with PM mass term, the type of LOD is Stokes coefficients.
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
    def LOD_motion_term(self,u_speed,lat,lon,pressure,isMas=True):
        rad_to_mas = 1
        if isMas:
            rad_to_mas = 180 * 3600 * 1000 / np.pi

        dp = np.zeros_like(pressure)
        dp[0] = pressure[0]-(pressure[0]+pressure[1])/2
        for k in np.arange(1,len(pressure)-1):
            dp[k] = (pressure[k-1]-pressure[k+2])/2
        dp[-1] = (pressure[-2]+pressure[-1])/2-pressure[-1]

        phi,lam = np.deg2rad(lat),np.deg2rad(lon)
        dphi,dlam = np.abs(phi[1]-phi[0]),np.abs(lam[1]-lam[0])

        phi_grid,lam_grid = np.meshgrid(phi,lam,indexing='ij')
        cos_phi = np.cos(phi_grid)
        dA = cos_phi*dphi*dlam

        dp_g = (dp/PMConstant.grav)[None,:,None,None]
        cos_phi_4d = cos_phi[None,None,:,:]

        dL3 = u_speed*cos_phi_4d*dp_g
        L3_vert = np.sum(dL3,axis=1)

        h3 = (PMConstant.radius**3)*np.sum(L3_vert*dA[None,:,:],axis=(1,2))
        chi3 = self.factor_LOD_motion*h3
        delta_LOD = chi3*PMConstant.LOD
        chi3 = rad_to_mas*chi3
        LOD = {"chi3":chi3,"LOD":delta_LOD}
        return LOD


def demo1():
    from datetime import date
    from pysrc.Auxiliary.FileTool import FileTool
    from pysrc.Auxiliary.MathTool import MathTool
    from pysrc.LoadFile.LoadL2SH import load_SHC
    import pysrc.Auxiliary.EnumClasses as Enums
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




if __name__ =="__main__":
    demo1()
