import numpy as np
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from pysrc.ancillary.load_file.DataClass import SHC,GRID
import SaGEA.auxiliary.preference.EnumClasses as Enums
from pysrc.sealevel_equation.SeaLevelEquation import PseudoSpectralSLE
# from SaGEA.auxiliary.preference.Constants import PMConstant
from pysrc.ancillary.constant.GeoConstant import PMConstant
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.ancillary.geotools.LLN import LoveNumber
import time

def mass_term(C21,S21,isMas=False):
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
    factor = -4*np.pi*(PMConstant.radius**4)*PMConstant.rho_water/(np.sqrt(15))
    I13 = factor*C21
    I23 = factor*S21
    chi1 = rad_to_mas*I13/(C-A)
    chi2 = rad_to_mas*I23/(C-A)
    chi = {"chi1":chi1,"chi2":chi2}
    return chi
def motion_term(u,v,lat,lon,dp,isMas=False):
    radius = PMConstant.radius
    grav = PMConstant.grav
    Omega = PMConstant.omega
    k2,ks = PMConstant.k2,PMConstant.ks
    Cm,Am = PMConstant.Cm,PMConstant.Am

    rad_to_mas = 1
    if isMas:
        rad_to_mas = (180 / np.pi) * 3600 * 1000
    # the colat is based on range of lat is from 90 to -90
    


    pass
def AAM_motion_term(u,v,lat,lon,pl,isMas=False):
    """
    :param u: latitude velocity (also known as zonal velocity)
    :param v: longitude velocity. Notes: both u and v shape is (time,layer,lat,lon)
    :param lat: the range is from 90--90
    :param lon: the range is from 0-360
    :param pl: pl means the pressure of every layer
    :param isMas: False means results are rad, True means results are mas
    :return:
    """
    radius = PMConstant.radius
    grav = PMConstant.grav
    Omega = PMConstant.omega
    k2,ks = PMConstant.k2,PMConstant.ks
    Cm,Am = PMConstant.Cm,PMConstant.Am
    rad_to_mas = 1
    if isMas:
        rad_to_mas = (180 / np.pi) * 3600 * 1000

    dp = np.zeros_like(pl)
    dp[0] = pl[0]-(pl[0]+pl[1])/2
    for k in np.arange(1,len(pl)-1):
        dp[k] = (pl[k-1]-pl[k+1])/2
    dp[-1] = (pl[-2]+pl[-1])/2- pl[-1]

    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    dphi = np.abs(phi[1]-phi[0])
    dlam = np.abs(lam[1]-lam[0])

    phi_grid, lam_grid = np.meshgrid(phi, lam, indexing="ij")

    cos_phi = np.cos(phi_grid)

    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    dA = cos_phi*dlam*dphi


    u_sin_lam = u*sin_lam
    u_cos_lam = u*cos_lam

    v_sin_lam = v*sin_lam
    v_cos_lam = v*cos_lam

    dp_g = (dp/grav)[np.newaxis,:,np.newaxis,np.newaxis]
    cos_phi_4d = cos_phi[None,None,:,:]

    dL1 = (u_sin_lam+v_cos_lam)*cos_phi_4d*dp_g
    dL2 = (u_cos_lam-v_sin_lam)*cos_phi_4d*dp_g

    L1_vert = np.sum(dL1,axis=1)
    L2_vert = np.sum(dL2,axis=1)

    H1 = (radius**3)*np.sum(L1_vert*dA[None,:,:],axis=(1,2))
    H2 = (radius**3)*np.sum(L2_vert*dA[None,:,:],axis=(1,2))

    factors = (Cm-Am)*Omega*(1-k2/ks)
    chi1 = H1/factors
    chi2 = H2/factors

    chi1 = chi1*rad_to_mas
    chi2 = chi2*rad_to_mas
    chi = {"chi1":chi1,"chi2":chi2}
    return chi
def Convert_Mass_to_Coordinates(C10, C11, S11):
    k1 = 0.021
    rho_earth = PMConstant.rho_earth
    X = np.sqrt(3) * (1 + k1) * C11 / rho_earth
    Y = np.sqrt(3) * (1 + k1) * S11 / rho_earth
    Z = np.sqrt(3) * (1 + k1) * C10 / rho_earth
    Coordinate = {"X": X, "Y": Y, "Z": Z}
    return Coordinate

def Convert_Stokes_to_Coordinates(C10, C11, S11):
    X = np.sqrt(3) * PMConstant.radius * C11
    Y = np.sqrt(3) * PMConstant.radius * S11
    Z = np.sqrt(3) * PMConstant.radius * C10
    Coordinate = {"X": X, "Y": Y, "Z": Z}
    return Coordinate

class PolarMotion:
    def __init__(self,GRACE,OceanSH,GAD,lmax):
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
        self.lat,self.lon = MathTool.get_global_lat_lon_range(self.res)

        self.LLN_method = Enums.LLN_Data.PREM
        self.frame = Enums.Frame.CF

    def setResolution(self,resolution):
        self.res = resolution
        self.lat,self.lon = MathTool.get_global_lat_lon_range(resolution)
        print(f"-----------------\n"
              f"Setting the processing data resolution is: {resolution} degree\n"
              f"The lat is from {self.lat[0]} to {self.lat[-1]},the lon is from {self.lon[0]} to {self.lon[-1]}\n"
              f"----------------")
        return self
    def setLatLon(self,lat,lon):
        self.lat,self.lon = lat,lon
        self.res = np.abs(self.lat[1]-self.lat[0])
        print(f"-----------------\n"
              f"Setting the processing data resolution is: {self.res} degree\n"
              f"The lat is from {self.lat[0]} to {self.lat[-1]},the lon is from {self.lon[0]} to {self.lon[-1]}."
              f"----------------")
        return self
    def setOcean(self,ocean_mask=None):
        if ocean_mask is not None:
            mask_grid = ocean_mask
        else:
            oceanmask_path = FileTool.get_project_dir("data/basin_mask/SH/Ocean_maskSH.dat")
            oceanmask_sh = load_SHC(oceanmask_path,key='',lmax=self.lmax)
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


        I[:, 0, 0], I[:, 0, 1], I[:, 0, 2], I[:, 0, 3], I[:, 0, 4],I[:, 0, 5] = \
            I_10C.value[:, 2], I_11C.value[:, 2], I_11S.value[:, 2], I_20C.value[:, 2], I_21C.value[:, 2], I_21S.value[:,2]

        I[:, 1, 0], I[:, 1, 1], I[:, 1, 2], I[:, 1, 3], I[:, 1, 4], I[:, 1, 5] = \
            I_10C.value[:, 3], I_11C.value[:, 3], I_11S.value[:, 3], I_20C.value[:, 3], I_21C.value[:, 3], I_21S.value[:,3]

        I[:, 2, 0], I[:, 2, 1], I[:, 2, 2], I[:, 2, 3], I[:, 2, 4], I[:, 2, 5] = \
            I_10C.value[:, 1], I_11C.value[:, 1], I_11S.value[:, 1], I_20C.value[:, 1], I_21C.value[:, 1], I_21S.value[:,1]

        I[:, 3, 0], I[:, 3, 1], I[:, 3, 2], I[:, 3, 3], I[:, 3, 4], I[:, 3, 5] = \
            I_10C.value[:, 6], I_11C.value[:, 6], I_11S.value[:, 6], I_20C.value[:, 6], I_21C.value[:, 6], I_21S.value[:,6]

        I[:, 4, 0], I[:, 4, 1], I[:, 4, 2], I[:, 4, 3], I[:, 4, 4], I[:, 4, 5] = \
            I_10C.value[:, 7], I_11C.value[:, 7], I_11S.value[:, 7], I_20C.value[:, 7], I_21C.value[:, 7], I_21S.value[:,7]
        I[:, 5, 0], I[:, 5, 1], I[:, 5, 2], I[:, 5, 3], I[:, 5, 4], I[:, 5, 5] = \
            I_10C.value[:, 5], I_11C.value[:, 5], I_11S.value[:, 5], I_20C.value[:, 5], I_21C.value[:, 7], I_21S.value[:,5]
        I = I
        print("-------------Finished I Matrix computation-------------")
        return I
    def G_Matrix_Term(self,mask=None,SLE=False):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:,0:4] = 0
        GRACE_SH[:,5:8] = 0
        # GRACE_SH[:,12] = 0
        if SLE:
            GRACE_SH = SHC(c=GRACE_SH).convert_type(from_type=Enums.PhysicalDimensions.Density,to_type=Enums.PhysicalDimensions.EWH)
            SLE = PseudoSpectralSLE(SH=GRACE_SH.value,lmax=self.lmax)
            SLE.setLoveNumber(lmax=self.lmax,method=self.LLN_method,frame=self.frame)
            SLE.setLatLon(lat=self.lat,lon=self.lon)
            kernal_SH = SLE.SLE(mask=self.setOcean(ocean_mask=mask),rotation=True)['RSL_SH']
            kernal = SHC(c=kernal_SH).convert_type(from_type=Enums.PhysicalDimensions.EWH,to_type=Enums.PhysicalDimensions.Density)
            kernal = (kernal.to_grid(grid_space=self.res).value)*(self.setOcean(ocean_mask=mask))
        else:
            kernal = (SHC(c=GRACE_SH).to_grid(self.res).value)*(self.setOcean(ocean_mask=mask))
        G_SH = GRID(grid=kernal,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        G = np.zeros((len(GRACE_SH),6))
        G[:,0] = G_SH[:,2]
        G[:,1] = G_SH[:,3]
        G[:,2] = G_SH[:,1]
        G[:,3] = G_SH[:,6]
        G[:,4] = G_SH[:,7]
        G[:,5] = G_SH[:,5]
        G = G
        # print(f"G V2 is: {G[0]}")
        print("-------------Finished G Matrix computation-------------")
        return G
    def Ocean_Model_Term(self,C10,C11,S11,C20,C21,S21):
        GAD_Correct = self.GAD.value
        OM_SH = self.OceanSH.value
        OM = np.zeros((len(OM_SH),6))
        # print(OM.shape)
        OM[:,0] = OM_SH[:,2]-GAD_Correct[:,2]+C10
        OM[:,1] = OM_SH[:,3]-GAD_Correct[:,3]+C11
        OM[:,2] = OM_SH[:,1]-GAD_Correct[:,1]+S11
        OM[:,3] = OM_SH[:,6]-GAD_Correct[:,6]+C20
        OM[:,4] = OM_SH[:,7]-GAD_Correct[:,7]+C21
        OM[:,5] = OM_SH[:,5]-GAD_Correct[:,5]+S21
        # print(f"OceanModel Term: {OM[0]}")
        return OM
    def GRD_Term(self,C10=None,C11=None,S11=None,C20=None,C21=None,S21=None,mask=None,GRD=False,rotation=True):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:,1]=S11
        GRACE_SH[:,2]=C10
        GRACE_SH[:,3]=C11
        GRACE_SH[:,6]=C20
        GRACE_SH[:,7]=C21
        GRACE_SH[:,5]=S21
        # GRACE_SH[:,]
        GRACE_SH = SHC(c=GRACE_SH).convert_type(from_type=Enums.PhysicalDimensions.Density,to_type=Enums.PhysicalDimensions.EWH)
        GRACE_GRID = GRACE_SH.to_grid(self.res)
        if GRD:
            SLE = PseudoSpectralSLE(SH=GRACE_SH.value,lmax=self.lmax)
            SLE.setLatLon(lat=self.lat,lon=self.lon)
            SLE.setLoveNumber(lmax=self.lmax,method=self.LLN_method,frame=self.frame)
            UpdateTerm = SLE.SLE(mask=mask,rotation=rotation)['RSL_SH']
        else:
            ocean_mask = self.setOcean(ocean_mask=mask)
            land_mask = 1-ocean_mask
            OceanArea = MathTool.get_acreage(basin=ocean_mask)
            uniform_value = GRACE_GRID.integral(mask=land_mask,average=False)/OceanArea
            uniform_mask = uniform_value[:,None,None]*ocean_mask
            UpdateTerm = GRID(grid=uniform_mask,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        UpdateTerm = SHC(c=UpdateTerm).convert_type(from_type=Enums.PhysicalDimensions.EWH,to_type=Enums.PhysicalDimensions.Density).value
        return UpdateTerm[:,2],UpdateTerm[:,3],UpdateTerm[:,1],UpdateTerm[:,6],UpdateTerm[:,7], UpdateTerm[:,5]
    def Low_Degree_Term(self,mask=None,GRD=False,rotation=True,SLE=False):
        """
        the series of Stokes coefficients follow: C10, C11, S11, C20, C21, S21
        that means, index 0->C10, 1->C11, 2->S11, 3->C20, 4->C21, 5->S21
        """
        print(f"=========Begin GRACE Degree Terms computing==========")
        start_time = time.time()
        GRACE_SH = self.GRACE.value
        I_C10,I_C11,I_S11,I_C20,I_C21,I_S21 = [np.zeros(len(GRACE_SH))]*6
        OM = self.Ocean_Model_Term(C10=I_C10,C11=I_C11,S11=I_S11,C20=I_C20,C21=I_C21,S21=I_S21)
        I = self.I_Matrix_Term(mask=mask)
        G = self.G_Matrix_Term(mask=mask,SLE=SLE)

        I_inv = np.linalg.inv(I)
        # print(f"I and I_inv:\n{I[0]}\n\n{I_inv[0]}")
        # print(f"verfiy: {I[0]@I_inv[0]}")
        C = np.einsum('nij,nj->ni',I_inv,OM-G)

        GRD_Ocean_Term = self.GRD_Term(C10=C[:,0],C11=C[:,1],S11=C[:,2],C20=C[:,3],C21=C[:,4],S21=C[:,5],
                                       mask=mask,GRD=GRD,rotation=rotation)
        for iter in np.arange(100):
            OM_new = self.Ocean_Model_Term(C10=GRD_Ocean_Term[0],C11=GRD_Ocean_Term[1],S11=GRD_Ocean_Term[2],
                                           C20=GRD_Ocean_Term[3],C21=GRD_Ocean_Term[4],S21=GRD_Ocean_Term[5])
            C_new = np.einsum('nij,nj->ni', I_inv, OM_new - G)
            delta = np.abs(C_new-C).flatten()
            if np.max(delta) < 10e-4:
                print(f"Iterative number is: {iter + 1}")
                break
            C = C_new

        lln = LoveNumber().config(lmax=self.lmax, method=self.LLN_method).get_Love_number()
        lln.convert(target=self.frame)
        k = lln.LLN[Enums.LLN_variable.k]

        factor = 1.021/(PMConstant.rho_earth*PMConstant.radius)
        factor2 = (3+3*k[2])/(5*PMConstant.rho_earth*PMConstant.radius)
        # factor3 = (3+3*k[3])/(7*EarthConstant.rhoear*EarthConstant.radiusm)

        print(f"Love numbers degree-1:{k[1]},degre-2:{k[2]},degree-3:{k[3]}")
        Mass_Coef = {"C10":C[:,0],"C11":C[:,1],"S11":C[:,2],"C20":C[:,3],"C21":C[:,4],"S21":C[:,5]}
        Stokes_Coef = {"C10":C[:,0]*factor,"C11":C[:,1]*factor,"S11":C[:,2]*factor,
                       "C20":C[:,3]*factor2,"C21":C[:,4]*factor2,"S21":C[:,5]*factor2}
        EWH_Coef = {"C10":C[:,0]/PMConstant.rho_water,"C11":C[:,1]/PMConstant.rho_water,"S11":C[:,2]/PMConstant.rho_water,
                    "C20":C[:,3]/PMConstant.rho_water,"C21":C[:,4]/PMConstant.rho_water,"S21":C[:,5]/PMConstant.rho_water}

        SH = {"Mass":Mass_Coef,"Stokes":Stokes_Coef,"EWH":EWH_Coef}
        end_time = time.time()
        print(f"----------------------------------------------\n"
              f"time-consuming: {end_time - start_time:.4f} s\n"
              f"==============================================\n")
        return SH
    def GSM_Like(self,mask=None,GRD=False,rotation=True,SLE=False):
        SH = self.Low_Degree_Term(mask=mask,GRD=GRD,rotation=rotation,SLE=SLE)
        C = SH['Mass']
        Coordinate = Convert_Mass_to_Coordinates(C10=C["C10"],C11=C["C11"],S11=C["S11"])
        print("-------------Finished GSM-like computation-------------\n"
              "==========================================================")
        return Coordinate

    def PM(self,mask=None,GRD=False,rotation=True,SLE=False):
        SH = self.Low_Degree_Term(mask=mask,GRD=GRD,rotation=rotation,SLE=SLE)
        C = SH['EWH']
        excitation = mass_term(C21=C['C21'],S21=C['S21'])
        print("-------------Finished PM computation-------------\n"
              "==========================================================")
        return excitation



def demo_AOM_motion():
    import xarray as xr
    import pandas as pd
    from tqdm import tqdm
    # u_set = xr.open_dataset("I:\ERA5\MAD/2010/u_wind-201001.nc")
    # v_set = xr.open_dataset("I:\ERA5\MAD/2010/v_wind-201001.nc")
    u_set,v_set=[],[]
    date_range = pd.date_range(start='2009-01-01',end='2009-12-31',freq="MS").strftime("%Y%m").tolist()
    for i in tqdm(date_range):
        u_temp = xr.open_dataset(f"I:\ERA5\MAD/2009/u_wind-{i}.nc")
        v_temp = xr.open_dataset(f"I:\ERA5\MAD/2009/v_wind-{i}.nc")
        u_set.append(u_temp['u'].values[0])
        v_set.append(v_temp['v'].values[0])
    # u_wind = u_set['u'].values
    # v_wind = v_set['v'].values
    pressure = u_temp['pressure_level'].values*100
    lats = u_temp['latitude'].values
    lons = u_temp['longitude'].values

    u_set = np.array(u_set)
    v_set = np.array(v_set)
    u_mean = np.mean(u_set,axis=0)
    v_mean = np.mean(v_set,axis=0)

    u_set = u_set-u_mean[None,:,:,:]
    v_set = v_set-v_mean[None,:,:,:]

    # print(v_set['v'].values.max())
    # print(u_set['u'].values.max())


    print(pressure[0])
    print(pressure[-1])
    print(pressure[-2])
    chi = AAM_motion_term(u=u_set,v=v_set,lat=lats,lon=lons,pl=pressure,isMas=True)
    print(chi['chi1'])
    # print(chi['chi2'])




if __name__ == "__main__":
    # C21 = np.zeros((10,1))
    # S21 = np.ones((10,1))
    # chi1,chi2 = Excitation_PM(C21,S21)
    # print(chi1)
    demo_AOM_motion()