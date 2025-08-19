import numpy as np
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from pysrc.aux_fuction.load_file.DataClass import SHC,GRID
import SaGEA.auxiliary.preference.EnumClasses as Enums
from pysrc.sealevel_equation.SeaLevelEquation import PseudoSpectralSLE
from SaGEA.auxiliary.preference.Constants import PMConstant
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.aux_fuction.LLN import LoveNumber
import time


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

class J2:
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
        I = np.zeros((N, 5, 5))
        ocean_mask = self.setOcean(ocean_mask=mask)
        theta, phi = MathTool.get_colat_lon_rad(lat=self.lat, lon=self.lon)
        Pilm = MathTool.get_Legendre(lat=theta, lmax=self.lmax, option=0)

        cosC10 = np.cos(0 * phi)
        cosC11 = np.cos(1 * phi)
        sinS11 = np.sin(1 * phi)
        cosC20 = np.cos(0 * phi)
        cosC30 = np.cos(0 * phi)

        CoreI10C = Pilm[:, 1, 0][:, None] * ocean_mask * cosC10[None, :]
        CoreI11C = Pilm[:, 1, 1][:, None] * ocean_mask * cosC11[None, :]
        CoreI11S = Pilm[:, 1, 1][:, None] * ocean_mask * sinS11[None, :]
        CoreI20C = Pilm[:, 2, 0][:, None] * ocean_mask * cosC20[None, :]
        CoreI30C = Pilm[:, 3, 0][:, None] * ocean_mask * cosC30[None, :]

        I_10C = GRID(grid=CoreI10C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_11C = GRID(grid=CoreI11C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_11S = GRID(grid=CoreI11S, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_20C = GRID(grid=CoreI20C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_30C = GRID(grid=CoreI30C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)

        I[:, 0, 0], I[:, 0, 1], I[:, 0, 2], I[:, 0, 3], I[:, 0, 4] = \
            I_10C.value[:, 2], I_11C.value[:, 2], I_11S.value[:, 2], I_20C.value[:, 2], I_30C.value[:, 2]

        I[:, 1, 0], I[:, 1, 1], I[:, 1, 2], I[:, 1, 3], I[:, 1, 4] = \
            I_10C.value[:, 3], I_11C.value[:, 3], I_11S.value[:, 3], I_20C.value[:, 3], I_30C.value[:, 3]

        I[:, 2, 0], I[:, 2, 1], I[:, 2, 2], I[:, 2, 3], I[:, 2, 4] = \
            I_10C.value[:, 1], I_11C.value[:, 1], I_11S.value[:, 1], I_20C.value[:, 1], I_30C.value[:, 1]

        I[:, 3, 0], I[:, 3, 1], I[:, 3, 2], I[:, 3, 3], I[:, 3, 4] = \
            I_10C.value[:, 6], I_11C.value[:, 6], I_11S.value[:, 6], I_20C.value[:, 6], I_30C.value[:, 6]

        I[:, 4, 0], I[:, 4, 1], I[:, 4, 2], I[:, 4, 3], I[:, 4, 4] = \
            I_10C.value[:, 12], I_11C.value[:, 12], I_11S.value[:, 12], I_20C.value[:, 12], I_30C.value[:, 12]
        I = I
        print("-------------Finished I Matrix computation-------------")
        return I
    def G_Matrix_Term(self,mask=None,SLE=False):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:, 0:4] = 0
        GRACE_SH[:, 6] = 0
        GRACE_SH[:, 12] = 0
        if SLE:
            GRACE_SH = SHC(c=GRACE_SH).convert_type(from_type=Enums.PhysicalDimensions.Density,
                                                    to_type=Enums.PhysicalDimensions.EWH)
            SLE = PseudoSpectralSLE(SH=GRACE_SH.value, lmax=self.lmax)
            SLE.setLoveNumber(lmax=self.lmax, method=self.LLN_method, frame=self.frame)
            SLE.setLatLon(lat=self.lat, lon=self.lon)
            kernal_SH = SLE.SLE(mask=self.setOcean(ocean_mask=mask), rotation=True)['RSL_SH']
            kernal = SHC(c=kernal_SH).convert_type(from_type=Enums.PhysicalDimensions.EWH,
                                                   to_type=Enums.PhysicalDimensions.Density)
            kernal = (kernal.to_grid(grid_space=self.res).value) * (self.setOcean(ocean_mask=mask))
        else:
            kernal = (SHC(c=GRACE_SH).to_grid(self.res).value) * (self.setOcean(ocean_mask=mask))
        G_SH = GRID(grid=kernal, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        G = np.zeros((len(GRACE_SH), 5))
        G[:, 0] = G_SH[:, 2]
        G[:, 1] = G_SH[:, 3]
        G[:, 2] = G_SH[:, 1]
        G[:, 3] = G_SH[:, 6]
        G[:, 4] = G_SH[:, 12]
        G = G
        # print(f"G V2 is: {G[0]}")
        print("-------------Finished G Matrix computation-------------")
        return G
    def Ocean_Model_Term(self,C10,C11,S11,C20,C30):
        GAD_Correct = self.GAD.value
        OM_SH = self.OceanSH.value
        OM = np.zeros((len(OM_SH),3))
        # print(OM.shape)
        OM[:,0] = OM_SH[:,2]-GAD_Correct[:,2]+C10
        OM[:,1] = OM_SH[:,3]-GAD_Correct[:,3]+C11
        OM[:,2] = OM_SH[:,1]-GAD_Correct[:,1]+S11
        OM[:,3] = OM_SH[:,6]-GAD_Correct[:,6]+C20
        OM[:,4] = OM_SH[:,12]-GAD_Correct[:,12]+C30
        return OM
    def GRD_Term(self,C10=None,C11=None,S11=None,C20=None,C30=None,mask=None,GRD=False,rotation=True):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:,1] = S11
        GRACE_SH[:,2] = C10
        GRACE_SH[:,3] = C11
        GRACE_SH[:, 6] = C20
        GRACE_SH[:, 12] = C30

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
        return UpdateTerm[:,2],UpdateTerm[:,3],UpdateTerm[:,1],UpdateTerm[:,6],UpdateTerm[:, 12]
    def Low_Degree_Term(self,mask=None,GRD=False,rotation=True,SLE=False):
        """
        the series of Stokes coefficients follow: C10, C11, S11, C20, C21, S21
        that means, index 0->C10, 1->C11, 2->S11, 3->C20, 4->C21, 5->S21
        """
        print(f"=========Begin GRACE Degree Terms computing==========")
        start_time = time.time()
        GRACE_SH = self.GRACE.value
        I_C10,I_C11,I_S11,I_C20,I_C30 = [np.zeros(len(GRACE_SH))]*5
        OM = self.Ocean_Model_Term(C10=I_C10,C11=I_C11,S11=I_S11,C20=I_C20,C30=I_C30)
        I = self.I_Matrix_Term(mask=mask)
        G = self.G_Matrix_Term(mask=mask,SLE=SLE)

        I_inv = np.linalg.inv(I)
        # print(f"I and I_inv:\n{I[0]}\n\n{I_inv[0]}")
        # print(f"verfiy: {I[0]@I_inv[0]}")
        C = np.einsum('nij,nj->ni',I_inv,OM-G)

        GRD_Ocean_Term = self.GRD_Term(C10=C[:,0],C11=C[:,1],S11=C[:,2],C20=C[:,3],C30=C[:,4],
                                       mask=mask,GRD=GRD,rotation=rotation)
        for iter in np.arange(100):
            OM_new = self.Ocean_Model_Term(C10=GRD_Ocean_Term[0],C11=GRD_Ocean_Term[1],S11=GRD_Ocean_Term[2],
                                           C20=GRD_Ocean_Term[3],C30=GRD_Ocean_Term[4])
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
        factor3 = (3+3*k[3])/(7*PMConstant.rho_earth*PMConstant.radius)

        print(f"Love numbers degree-1:{k[1]},degre-2:{k[2]},degree-3:{k[3]}")
        Mass_Coef = {"C10":C[:,0],"C11":C[:,1],"S11":C[:,2],"C20":C[:,3],"C30":C[:,4]}
        Stokes_Coef = {"C10":C[:,0]*factor,"C11":C[:,1]*factor,"S11":C[:,2]*factor,
                       "C20":C[:,3]*factor2,"C30":C[:,4]*factor3}

        SH = {"Mass":Mass_Coef,"Stokes":Stokes_Coef}
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
    def Full_Geocenter(self,GAC=None,mask=None,GRD=False,rotation=True,SLE=False):
        GAC = SHC(c=GAC)
        GAC_Coordinate = Convert_Stokes_to_Coordinates(C10=GAC.value[:,2],C11=GAC.value[:,3],S11=GAC.value[:,1])
        SH = self.Low_Degree_Term(mask=mask,GRD=GRD,rotation=rotation,SLE=SLE)
        C = SH['Mass']
        GSM_Coordinate = Convert_Mass_to_Coordinates(C10=C["C10"], C11=C["C11"], S11=C["S11"])
        X = GAC_Coordinate['X']+GSM_Coordinate['X']
        Y = GAC_Coordinate['Y']+GSM_Coordinate['Y']
        Z = GAC_Coordinate['Z']+GSM_Coordinate['Z']
        full_geocenter = {"X":X,"Y":Y,"Z":Z}
        print("-------------Finished Full-Geocenter computation-------------\n"
              "=============================================================")
        return full_geocenter

