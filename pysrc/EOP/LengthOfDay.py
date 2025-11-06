import numpy as np
from pysrc.ancillary.constant.GeoConstant import EOPConstant
from lib.SaGEA.auxiliary.aux_tool.FileTool import FileTool
from lib.SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.ancillary.constant.Setting import EAMtype


def LOD_mass_term(C00,C20,isMas=False):
    Cm = EOPConstant.Cm
    radius = EOPConstant.radius
    M = EOPConstant.Mass
    k2 = EOPConstant.k2_load
    LOD = EOPConstant.LOD
    rad_to_mas = 1
    if isMas:
        rad_to_mas = 180 * 3600 * 1000 / np.pi

    C00 = C00
    C20 = C20

    factor = 2*0.756*(radius**2)*M/(3*(1+k2)*Cm)
    chi3 = factor*(C00-np.sqrt(5)*C20)
    delta_LOD = chi3*LOD
    chi3 = rad_to_mas*chi3

    LOD = {"chi3":chi3,"LOD":delta_LOD}
    return LOD

def LOD_mass_term_normal(P,lat,lon,isMas=False):
    """
    :param P: the grid dateset unit with pressure
    :param lat: latitude from 90--90, not co-latitude due to cos phi is used here
    :param lon: longitude from 0-360
    :param isMas: unit with mas is True, False is the unit with rad
    :return: unit of delta LOD is s
    """
    grav = EOPConstant.grav
    radius = EOPConstant.radius
    LOD = EOPConstant.LOD
    Cm = EOPConstant.Cm

    rad_to_mas = 1
    if isMas:
        rad_to_mas = 180 * 3600 * 1000 / np.pi

    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)

    dphi = np.abs(phi[1]-phi[0])
    dlam = np.abs(lam[1]-lam[0])

    phi_grid,lam_grid = np.meshgrid(phi,lam,indexing='ij')

    cos_phi = np.cos(phi_grid)
    dA = cos_phi*dphi*dlam
    kernal = P*cos_phi*cos_phi*dA
    factor = 0.756*(radius**4)/(Cm*grav)
    chi3 = np.sum(factor*kernal,axis=(1,2))
    delta_LOD = chi3 * LOD
    chi3 = chi3*rad_to_mas


    LOD = {"chi3":chi3,"LOD":delta_LOD}
    return LOD

def LOD_motion_term(u,pl,lat,lon,isMas=False):
    Cm = EOPConstant.Cm
    grav = EOPConstant.grav
    radius = EOPConstant.radius
    Omega = EOPConstant.omega
    LOD = EOPConstant.LOD
    rad_to_mas = 1
    if isMas:
        rad_to_mas = 180 * 3600 * 1000 / np.pi

    dp = np.zeros_like(pl)
    dp[0] = pl[0]-(pl[0]+pl[1])/2
    for k in np.arange(1,len(pl)-1):
        dp[k] = (pl[k-1]-pl[k+1])/2
    dp[-1] = (pl[-2]+pl[-1])/2- pl[-1]

    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    dphi = np.abs(phi[1] - phi[0])
    dlam = np.abs(lam[1] - lam[0])

    phi_grid, lam_grid = np.meshgrid(phi, lam, indexing="ij")
    cos_phi = np.cos(phi_grid)
    dA = cos_phi*dlam*dphi

    dp_g = (dp/grav)[None,:,None,None]
    cos_phi_4d = cos_phi[None,None,:,:]

    dL3 = u*cos_phi_4d*dp_g
    L3_vert = np.sum(dL3,axis=1)

    h3 = (radius**3)*np.sum(L3_vert*dA[None,:,:],axis=(1,2))

    factor = 0.998/(Cm*Omega)
    chi3 = factor*h3
    delta_LOD = chi3*LOD
    chi3 = rad_to_mas*chi3

    LOD = {"chi3":chi3,"LOD":delta_LOD}
    return LOD

class LOD:
    def __init__(self):
        self.rad_to_ms = EOPConstant.rad_to_ms
        pass

    def Mass_term_SH(self,SH,isMs=True):
        coef_numerator = 0.753 * (EOPConstant.radius ** 2) * EOPConstant.Mass * 2
        coef_denominator = (1 + EOPConstant.k2_load) * EOPConstant.Cm * 3

        C00 = SH[:, 0]
        C20 = SH[:, 6]
        chi3 = (coef_numerator / coef_denominator) * (C00 - np.sqrt(5) * C20)
        if isMs:
            chi3 = chi3 * self.rad_to_ms
        LOD = {"chi3": chi3}
        return LOD

    def Mass_term(self,Ps,lat,lon,isMs=False):
        coef_numerator = 0.998 * (EOPConstant.radius ** 3)
        coef_denominator = EOPConstant.Cm * EOPConstant.omega * EOPConstant.grav

        phi, lam = np.deg2rad(lat), np.deg2rad(lon)
        dphi, dlam = np.abs(phi[1] - phi[0]), np.abs(lam[1] - lam[2])
        phi_grid, lam_grid = np.meshgrid(phi, lam, indexing="ij")

        cos_phi = np.cos(phi_grid)
        dA = cos_phi * dphi * dlam

        kernal = Ps * cos_phi * cos_phi * dA
        chi3 = (coef_numerator / coef_denominator) * np.sum(kernal, axis=(1, 2))
        if isMs:
            chi3 = chi3 * self.rad_to_ms
        LOD = {"chi3": chi3}
        return LOD
    def Motion_term(self,Us,lat,lon,levPres,Ps=None,Zth=None,type=EAMtype.AAM,isMs=False):
        coef_numerator = 0.998 * (EOPConstant.radius ** 2)
        coef_denominator = EOPConstant.Cm * EOPConstant.omega

        chi3_series = []
        for i in np.arange(len(Us)):
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

            h3 = (EOPConstant.radius ** 2) * np.sum(U * dA * cos_phi, axis=(0, 1))
            chi3 = (coef_numerator / coef_denominator) * h3
            if isMs:
                chi3 = chi3 * self.rad_to_ms
            chi3_series.append(chi3)

        chi3_series = np.array(chi3_series)
        LOD = {"chi3": chi3_series}
        return LOD
    def __dp_AAM(self, levPres, lev,lat,lon, surPres=None, geoHeight=None):
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

    def __dp_OAM(self,levDepth,lev,lat,lon,surSeaHeight=None):
        sample_arr = np.ones((len(lat),len(lon))).flatten()
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
                return (top_pres - iso_pres[lev]) * iso_R[lev]
            else:
                return (iso_pres[lev - 1] - iso_pres[lev]) * iso_R[lev]

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


def demo2():
    from datetime import date
    lmax = 60
    begin_date, end_date = date(2010, 1, 1), date(2010, 12, 31)
    gad_dir, gad_key = FileTool.get_project_dir("I:\GFZ\GAB\GFZ_GFZ-Release-06_GAX_products_GAB/"), "gfc"
    shc_gad = load_SHC(gad_dir, key=gad_key, lmax=lmax, begin_date=begin_date, end_date=end_date)
    shc_gad.de_background()


    SH = shc_gad.value
    SH[:,0] = 0
    a = LOD().Mass_term_SH(SH=SH,isMs=True)



def demo3():
    pass

if __name__ == '__main__':
    demo2()