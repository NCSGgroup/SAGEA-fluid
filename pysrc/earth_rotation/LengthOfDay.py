import numpy as np
# from SaGEA.auxiliary.preference.Constants import PMConstant
from pysrc.aux_fuction.constant.GeoConstant import PMConstant
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC



def LOD_mass_term(C00,C20,isMas=False):
    Cm = PMConstant.Cm
    radius = PMConstant.radius
    M = PMConstant.Mass
    k2 = PMConstant.k2_load
    LOD = PMConstant.LOD
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
    grav = PMConstant.grav
    radius = PMConstant.radius
    LOD = PMConstant.LOD
    Cm = PMConstant.Cm

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
    Cm = PMConstant.Cm
    grav = PMConstant.grav
    radius = PMConstant.radius
    Omega = PMConstant.omega
    LOD = PMConstant.LOD
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
        pass


def demo2():
    from datetime import date
    lmax = 60
    begin_date, end_date = date(2010, 1, 1), date(2010, 12, 31)
    gad_dir, gad_key = FileTool.get_project_dir("I:\GFZ\GAB\GFZ_GFZ-Release-06_GAX_products_GAB/"), "gfc"
    shc_gad = load_SHC(gad_dir, key=gad_key, lmax=lmax, begin_date=begin_date, end_date=end_date)
    shc_gad.de_background()

    # shc_gad.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,to_type=Enums.PhysicalDimensions.Pressure)
    # res = 1
    # grid_gad = shc_gad.to_grid(grid_space=res)
    # lat,lon = MathTool.get_global_lat_lon_range(res)
    #
    # LOD = LOD_mass_term_normal(P=grid_gad.value,lat=lat,lon=lon,isMas=False)
    # print(LOD['LOD'])
    # print(LOD['chi3'])


    # print(shc_gad.value.shape)
    # print(shc_gad.value[4,:7])
    # shc_gad = shc_gad.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,to_type=Enums.PhysicalDimensions.Density)
    SH = shc_gad.value
    SH[:,0] = 0
    LOD = LOD_mass_term(C00=SH[:,0],C20=SH[:,6],isMas=False)
    print(LOD['LOD'])
    print(LOD['chi3'])

    # S21 = shc_gad.value[:,5]
    # C21 = shc_gad.value[:,7]
    # print(S21.shape,C21.shape)
    # chi = mass_term(C21=C21,S21=S21)
    # print(chi['chi2'])

def demo3():
    pass

if __name__ == '__main__':
    demo2()