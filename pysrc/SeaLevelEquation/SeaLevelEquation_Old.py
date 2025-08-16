from pysrc.Auxiliary.FileTool import FileTool
from pysrc.Auxiliary.MathTool import MathTool
from pysrc.LoadFile.LoadL2SH import load_SHC
from pysrc.Auxiliary.Constants import GeoConstants,EarthConstant
from pysrc.Auxiliary.LLN import LoveNumber,LLN_Data,LLN_variable
from pysrc.Auxiliary.EnumClasses import Displacement,GreenFunction
from pysrc.BasisFunction import PointLoad,DiskLoad
import numpy as np
from pysrc.LoadFile.DataClass import SHC,GRID
import xarray as xr
from tqdm import tqdm
import time
from pysrc.BasisFunction.Harmonic import Harmonic


class SpatialSLE:
    '''
    This method is referring the research article of Adhikari (2019);
    https://doi.org/10.5194/essd-11-629-2019
    '''
    def __init__(self,grid,lat,lon):
        self.lat,self.lon = lat,lon
        self.Load = GRID(grid=grid,lat=lat,lon=lon)
        self.lmax = 60
        self.lln_lmax = 60
        self.res = np.abs(lat[1]-lat[0])
        self.Green = GreenFunction.PointLoad

    def setmaxDegree(self, lmax,lln_lmax):
        self.lmax = lmax
        self.lln_lmax = lln_lmax
        print(f"The configuration information:\n"
              f"lmax:{self.lmax},lln_lmax: {self.lln_lmax}, resolution:{self.res},lat:{self.lat.shape},lon:{self.lon.shape}")
        return self

    def setGreenFunctionType(self,kind:GreenFunction.PointLoad):
        self.Green = kind
        return self

    def Ocean_function(self, loadfile="data/basin_mask/SH/Ocean_maskSH.dat"):
        OceanFuction_SH = FileTool.get_project_dir(loadfile)
        shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)  # load basin mask (in SHC)
        grid_basin = shc_OceanFunction.to_grid(grid_space=self.res)
        grid_basin.limiter(threshold=0.5)
        ocean_function = grid_basin.value[0]
        return ocean_function

    def BaryTerm(self,mask):
        ocean_mask = mask
        land_mask = 1-ocean_mask
        OceanArea = MathTool.get_acreage(basin=ocean_mask)
        E = -(self.Load.integral(mask=land_mask,average=False))/OceanArea
        # print(f"Barystatic term is {type(E)},{E.shape}")
        E = E[:,np.newaxis,np.newaxis]*mask
        return E

    def GRDTerm(self,WL,mask,rotation=False):
        ocean_mask = mask
        assert WL.ndim == 3, "The dimension should be 3"
        Load = WL.reshape((len(WL), -1)).T
        print(f"Load shape is:{Load.shape}")
        lln = LoveNumber().config(lmax=self.lln_lmax, method=LLN_Data.PREM).get_Love_number()
        lon_2D, lat_2D = np.meshgrid(self.lon, self.lat)
        point = {
            'lat': lat_2D.flatten(),
            'lon': lon_2D.flatten(),
        }
        if self.Green is GreenFunction.PointLoad:
            grids = PointLoad.Grids_generation.Equal_angular_distance(resolution=self.res)
            grids['EWH'] = Load * EarthConstant.rhow
            gfa = PointLoad.GFA_regular_grid(lln=lln)
            gfa.configure(grids=grids)
            Bg = gfa.evaluation(points=point, variable=Displacement.Vertical, resolution=self.res).T
            EPg = gfa.evaluation(points=point, variable=Displacement.Geoheight, resolution=self.res).T

        elif self.Green is GreenFunction.DiskLoad:
            rr = DiskLoad.grid2radius(lat_center=lat_2D, grid_size=self.res)
            grids = {
                'lat': lat_2D.flatten(),
                'lon': lon_2D.flatten(),
                'radius': rr[0].flatten(),
                'EWH': Load
            }
            gfa = DiskLoad.GFA_regular_grid(lln=lln)
            gfa.configure(grids=grids,cf=1000)
            Bg = gfa.evaluation(points=point, variable=Displacement.Vertical).T
            EPg = gfa.evaluation(points=point, variable=Displacement.Geoheight).T
        else:
            Bg, EPg = None,None
            assert self.Green in [GreenFunction.DiskLoad,GreenFunction.PointLoad],\
                "Please input right Green-Function! Using like setGreenFunction(kind=GreenFunction.PointLoad)!"

        Bg = Bg.reshape(len(WL), len(self.lat), len(self.lon))
        EPg = EPg.reshape(len(WL), len(self.lat), len(self.lon))

        if rotation:
            Rota = self.RotationTerm(WL=WL)
            EPr = Rota["EP"]
            Br = Rota["B"]
            EP = EPr+EPg
            B = Br+Bg
        else:
            B = Bg
            EP = EPg

        Mean_GRD = GRID(grid=EP - B, lat=self.lat, lon=self.lon).integral(mask=ocean_mask, average=True)
        Mean_GRD = Mean_GRD[:,np.newaxis,np.newaxis]*ocean_mask

        GRD = EP-B-Mean_GRD
        print(f"The GRD is: {GRD.shape},\n"
              f"The Mean GRD: {Mean_GRD.shape},\n"
              f"The GHC is: {(EP+self.BaryTerm(mask=mask)-Mean_GRD).shape},\n"
              f"The VLM is: {B.shape}")

        GRDterm = {"GRD":GRD,
                   "GRD_Mean":Mean_GRD,
                   "GHC":EP+self.BaryTerm(mask=mask)-Mean_GRD,
                   "VLM":B}
        return GRDterm

    def RotationTerm(self, WL):
        k2, h2 = EarthConstant.k2, EarthConstant.h2
        grav = EarthConstant.grav
        self.lat, self.lon = np.array(self.lat), np.array(self.lon)
        SHF = Harmonic(lat=self.lat, lon=self.lon, lmax=self.lmax, option=1).get_spherical_harmonic_function()[
            "Upsilon"]
        print(f"SHF shape is:{SHF.shape}")
        WL_SH = GRID(grid=WL, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        T_00 = WL_SH[:, 0]
        T_20 = WL_SH[:, 6]
        T_21 = WL_SH[:, 7]
        T_2m1 = WL_SH[:, 5]

        T_vec = np.array([T_00, T_20, T_21, T_2m1])
        MatrixS = self.Rotation_matrix()
        delta_J = MatrixS['Psi'] @ T_vec
        m = MatrixS['Gamma'] @ delta_J
        delta_Lambda = MatrixS['Phi'] @ m
        print(f"Delta_Lambda shape is :{delta_Lambda.shape}")

        SH = np.zeros_like(WL_SH)
        SH[:,0] = delta_Lambda[0,:]
        grid1 = np.einsum("wjl,gl->gwj", SHF, SH)

        SH[:,0] = 0
        SH[:,6] = delta_Lambda[1,:]
        SH[:,7] = delta_Lambda[2,:]
        SH[:,5] = delta_Lambda[3,:]

        grid2 = np.einsum("wjl,gl->gwj", SHF, SH)

        EPR = (grid1 + (1 + k2)* grid2)/grav
        BR = (h2 / grav) * grid2
        print(f"The EP and B are: {EPR.shape} and {BR.shape}")
        rotation = {
            "EP": EPR,
            "B": BR
        }
        return rotation

        # S_rot = MatrixS['T'] @ delta_Lambda
        #
        # # print(f"new shape are: {delta_J.shape, m.shape, delta_Lambda.shape, S_rot.shape}")
        # S_rot = S_rot.T
        # SH_rotation = np.zeros_like(SH)
        # SH_rotation[:, 6] = S_rot[:, 0]
        # SH_rotation[:, 7] = S_rot[:, 1]
        # SH_rotation[:, 5] = S_rot[:, 2]
        # print(f"rotation is:{SH_rotation.shape}")
        # return SH_rotation

    def Rotation_matrix(self):
        a = EarthConstant.radiusm
        rho_w = EarthConstant.rhow
        Omega_E = EarthConstant.Omega
        A, C = EarthConstant.A, EarthConstant.C
        sigma_0 = EarthConstant.Chandler
        k2, h2 = EarthConstant.k2, EarthConstant.h2
        g = EarthConstant.grav
        lln = LoveNumber().config(lmax=self.lmax, method=LLN_Data.PREM).get_Love_number()
        kl2, hl2 = lln.LLN[LLN_variable.k][2], lln.LLN[LLN_variable.h][2]

        Psi_JT = np.array([
            [0, 0, -4 / np.sqrt(15), 0],
            [0, 0, 0, -4 / np.sqrt(15)],
            [8 / 3, -8 / (3 * np.sqrt(5)), 0, 0]
        ]) * np.pi * rho_w * a ** 4

        Gamma_mJ = np.diag([
            Omega_E * (1 + kl2) / (A * sigma_0),
            Omega_E * (1 + kl2) / (A * sigma_0),
            -(1 + kl2) / C
        ])

        Phi_lm = (a * Omega_E) ** 2 * np.array([
            [0, 0, 2 / 3],
            [0, 0, -2 / (3 * np.sqrt(5))],
            [-1 / np.sqrt(15), 0, 0],
            [0, -1 / np.sqrt(15), 0]
        ])

        T_SL = (1 + k2 - h2) / g * np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        Maxtrixs = {
            "Psi": Psi_JT,
            "Gamma": Gamma_mJ,
            "Phi": Phi_lm,
            "T": T_SL
        }
        return Maxtrixs

    def SLE(self,mask=None,rotation=False):
        print(f"=========Begin Green Convolution SLF computing==========")
        start_time = time.time()

        if mask is not None:
            ocean_mask = mask
        else:
            ocean_mask = self.Ocean_function()
        assert self.Load.value.ndim == 3, "The dimension should be 3"
        print(f"ocean_mask is:{ocean_mask.shape}")

        Baryterm = self.BaryTerm(mask=ocean_mask)
        S = np.zeros_like(self.Load.value) + Baryterm
        GHC,VLM = None,None
        for iteration in np.arange(100):
            WL = self.Load.value * (1 - ocean_mask) + S*ocean_mask
            GRDterm = self.GRDTerm(WL=WL, mask=ocean_mask, rotation=rotation)
            GHC,VLM = GRDterm['GHC'],GRDterm['VLM']
            GRD = GRDterm['GRD']
            S_new = Baryterm + GRD
            delta = np.abs(np.linalg.norm(S_new*ocean_mask, axis=(1, 2)) - np.linalg.norm(S*ocean_mask, axis=(1, 2)))
            print(f"The iteration is: {iteration + 1},\n"
                  f"The delta is: {np.max(delta)}")
            if np.all(delta < 1e-6):
                break
            S = S_new


        S = GRID(grid=S, lat=self.lat, lon=self.lon)
        S_SH = S.to_SHC(lmax=self.lmax)

        SLE = {"Input":self.Load.value,
               "RSL_SH": S_SH.value,
               "RSL": S.value,
               "GHC": GHC,
               "VLM": VLM,
               "BaryRSL": Baryterm,
               "mask":ocean_mask}

        print(f"===Baryterm is: {Baryterm}\n")
        end_time = time.time()
        print(f"----------------------------------------------\n"
              f"-----time-consuming: {end_time - start_time:.4f} s-------\n"
              f"==============================================\n")

        return SLE
        pass

class SpectralSLE_Quasi:
    '''
    This method is referring the research article of Tamisiea (2010);
    https://doi.org/10.1029/2009JC005687
    '''
    def __init__(self, SH, lmax=60):
        self.lmax = lmax
        self.shc = SHC(c=SH)
        self.res = 1
        self.tolerance = 1e-6
        self.lat, self.lon = MathTool.get_global_lat_lon_range(resolution=self.res)

    def setLatLon(self, lat, lon):
        # self.res = resolution
        self.lat, self.lon = lat, lon
        self.res = np.abs(lat[1] - lat[0])
        print(f"The configuration information:\n"
              f"lmax:{self.lmax},resolution:{self.res},lat:{self.lat.shape},lon:{self.lon.shape}")
        return self

    def Ocean_function(self, mask=None,loadfile="data/basin_mask/SH/Ocean_maskSH.dat"):

        if mask is not None:
            # grid_OceanFunction = xr.open_dataset('../../data/ref_sealevel/ocean_mask.nc')["ocean_mask"].values
            grid_OceanFunction = mask
            shc_OceanFunction = GRID(grid=grid_OceanFunction,lat=self.lat,lon=self.lon).to_SHC(lmax=360)
            N = int((self.lmax + 2) * (self.lmax + 1) / 2 + (self.lmax) * (self.lmax + 1) / 2)
            shc_OceanFunction = shc_OceanFunction.value[:,:N]

        else:
            OceanFuction_SH = FileTool.get_project_dir(loadfile)
            # load basin mask (in SHC)
            shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)
            grid_basin = shc_OceanFunction.to_grid(grid_space=self.res)
            grid_basin.limiter(threshold=0.5)
            grid_OceanFunction = grid_basin.value[0]
            shc_OceanFunction = shc_OceanFunction.value
        ocean_function = {
                "SH":shc_OceanFunction,
                "Grid":grid_OceanFunction
            }

        return ocean_function

    def GRDparameter(self, option=1):
        lln = LoveNumber().config(lmax=self.lmax, method=LLN_Data.PREM).get_Love_number()
        kl = lln.LLN[LLN_variable.k]
        hl = lln.LLN[LLN_variable.h]

        rho_water = GeoConstants.density_water
        rho_earth = GeoConstants.density_earth

        if option == 1:
            Green_N = []
            Green_U = []
            for i in np.arange(self.lmax + 1):
                for j in np.arange(-i, i + 1):
                    Green_N.append(3 * rho_water * (1 + kl[i]) / (rho_earth * (2 * i + 1)))
                    Green_U.append(3 * rho_water * (- hl[i]) / (rho_earth * (2 * i + 1)))
            Green_N = np.array(Green_N)
            Green_U = np.array(Green_U)
        else:
            Green_N = np.zeros((self.lmax + 1, self.lmax + 1))
            Green_U = np.zeros((self.lmax + 1, self.lmax + 1))
            for i in np.arange(self.lmax + 1):
                Green_N[i, 0:i + 1] = 3 * rho_water * (1 + kl[i]) / (rho_earth * (2 * i + 1))
                Green_U[i, 0:i + 1] = 3 * rho_water * (- hl[i]) / (rho_earth * (2 * i + 1))

        Green = {
            "N":Green_N,
            "U":Green_U,
        }
        return Green

    def GRDTerm(self, AL_SH, RSL_SH, rotation=False,option=1):
        Green_coef = self.GRDparameter(option=option)
        GreenN = Green_coef["N"]
        GreenU = Green_coef["U"]
        if option == 1:
            X_SH = SHC(c=GreenN * (AL_SH + RSL_SH))
            P_SH = SHC(c=GreenU * (AL_SH + RSL_SH))
        else:
            AL_C, AL_S = SHC(c=AL_SH).get_cs2d()
            RS_C, RS_S = SHC(c=RSL_SH).get_cs2d()
            X_C = GreenN * (AL_C + RS_C)
            X_S = GreenN * (AL_S + RS_S)
            P_C = GreenU * (AL_C + RS_C)
            P_S = GreenU * (AL_S + RS_S)
            X_SH = SHC(c=X_C, s=X_S)
            P_SH = SHC(c=P_C, s=P_S)

        if rotation:
            Rota_SH = self.RotationTerm(SH=AL_SH+RSL_SH)
            X_SH = X_SH.value+Rota_SH['N']
            X_SH = SHC(c=X_SH)
            P_SH = P_SH.value+Rota_SH['U']
            P_SH = SHC(c=P_SH)

        GRD_SH_value = X_SH.value+P_SH.value
        GRD_SH = SHC(c=GRD_SH_value)
        GRD_GRID = GRD_SH.to_grid(self.res)

        GRD = {
            "SH":GRD_SH.value,
            "N":X_SH.value,
            "U":P_SH.value,
            "Grid":GRD_GRID.value,
        }

        return GRD

    def BaryTerm(self, AL_SH, GRD_SH, mask=None):
        ocean_function = self.Ocean_function(mask=mask)
        ocean_mask = ocean_function["Grid"]
        Mask_SH = ocean_function["SH"]
        Mask00 = Mask_SH[0,0]

        GRD = SHC(c=GRD_SH).to_grid(self.res)
        AL_00 = AL_SH[:, 0]
        # PureBary = (-AL_00) / Mask00
        RO_SH = GRID(grid=GRD.value * ocean_mask, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        Baryterm = -(AL_00 + RO_SH[:, 0]) / Mask00
        Bary_SH = Baryterm[:, None] @ Mask_SH
        Bary_Grid = SHC(c=Bary_SH).to_grid(self.res).value*ocean_mask

        Baryterm = {
            "SH":Bary_SH,
            "Grid":Bary_Grid,
        }

        return Baryterm

    def RSLTerm(self, GRD, Bary):
        """This is quasi sea level"""
        Bary_Grid = SHC(c=Bary['SH']).to_grid(self.res).value
        QuasiRSL = GRD["Grid"]+Bary_Grid
        RSL_SH = GRID(grid=QuasiRSL,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        RSL = {
            "SH":RSL_SH,
            "Grid":QuasiRSL,
        }
        return RSL

    def SLE(self, mask=None, rotation=False):
        print(f"=========Begin Quasi-Spectral SLF computing==========")
        start_time = time.time()
        ocean_function = self.Ocean_function(mask=mask)
        ocean_mask = ocean_function["Grid"]
        Mask_SH = ocean_function['SH']

        AL = self.shc.to_grid(grid_space=self.res).value * (1 - ocean_mask)
        AL_SH = GRID(grid=AL, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value

        GRD = {"Grid": np.zeros_like(AL),
               "SH": np.zeros_like(AL_SH)}
        GHC,VLM = np.zeros_like(AL_SH),np.zeros_like(AL_SH)

        Bary = self.BaryTerm(AL_SH=AL_SH, GRD_SH=GRD['SH'], mask=mask)
        RSL_SH = self.RSLTerm(GRD=GRD, Bary=Bary)['SH']
        for iter in np.arange(100):
            GRD = self.GRDTerm(AL_SH, RSL_SH, rotation=rotation)
            Bary = self.BaryTerm(AL_SH=AL_SH, GRD_SH=GRD['SH'], mask=mask)
            new_RSL_SH = self.RSLTerm(GRD=GRD, Bary=Bary)['SH']
            GHC = GRID(grid=SHC(c=GRD["N"]).to_grid(self.res).value + Bary['Grid'],lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
            VLM = GRD["U"]

            delta = np.max(np.abs(new_RSL_SH - RSL_SH))
            print(f"The iteration is:{iter + 1},\n"
                  f"The delta is: {delta}")
            # print("delta:", delta, "Comparison:", delta < 1e-6, "All:", np.all(delta < 1e-6))
            if np.all(delta < 1e-6):
                break
            RSL_SH = new_RSL_SH
        RSL_SH = SHC(c=RSL_SH)
        RSL = RSL_SH.to_grid(self.res)
        SLE = {"Input": self.shc.value,
               "RSL_SH": RSL_SH.value,
               "RSL": RSL.value,
               "GHC": GHC,
               "VLM": VLM,
               "BaryRSL": Bary['SH'],
               "mask": Mask_SH}
        # print(f"===Baryterm is: {Bary}\n")
        end_time = time.time()
        print(f"----------------------------------------------\n"
              f"-----time-consuming: {end_time - start_time:.4f} s-------\n"
              f"==============================================\n")
        return SLE

    def RotationTerm(self,SH):
        SH_00 = SH[:,0]
        SH_20 = SH[:,6]
        SH_21 = SH[:,7]
        SH_2m1 = SH[:,5]

        SH_vec = np.array([SH_00,SH_20,SH_21,SH_2m1])
        MatrixS = self.Rotation_matrix()
        # print(f"Matrix shape:{MatrixS['Psi'].shape,MatrixS['Gamma'].shape,MatrixS['Phi'].shape,MatrixS['T'].shape}")
        # print(f"SH shape is:{SH_vec.shape}")
        delta_J = MatrixS['Psi']@SH_vec

        m = MatrixS['Gamma']@delta_J

        delta_Lambda = MatrixS['Phi']@m

        Y_rot = MatrixS['N']@delta_Lambda

        P_rot = MatrixS['U']@delta_Lambda

        # print(f"new shape are: {delta_J.shape, m.shape, delta_Lambda.shape, S_rot.shape}")
        Y_rot = Y_rot.T
        SH_rotation_Y = np.zeros_like(SH)
        SH_rotation_Y[:,6] = Y_rot[:,0]
        SH_rotation_Y[:,7] = Y_rot[:,1]
        SH_rotation_Y[:,5] = Y_rot[:,2]
        # print(f"rotation is:{SH_rotation.shape}")
        P_rot = P_rot.T
        SH_rotation_P = np.zeros_like(SH)
        SH_rotation_P[:,6] = P_rot[:,0]
        SH_rotation_P[:,7] = P_rot[:,1]
        SH_rotation_P[:,5] = P_rot[:,2]

        SH_rotation = {
            "N":SH_rotation_Y,
            "U":SH_rotation_P,
        }
        return SH_rotation
    def Rotation_matrix(self):
        a = EarthConstant.radiusm
        rho_w= EarthConstant.rhow
        Omega_E = EarthConstant.Omega
        A,C = EarthConstant.A, EarthConstant.C
        sigma_0 = EarthConstant.Chandler
        k2,h2 = EarthConstant.k2, EarthConstant.h2
        g = EarthConstant.grav
        lln = LoveNumber().config(lmax=self.lmax, method=LLN_Data.PREM).get_Love_number()
        kl2, hl2 = lln.LLN[LLN_variable.k][2], lln.LLN[LLN_variable.h][2]

        Psi_JT = np.array([
            [0, 0, -4/np.sqrt(15), 0],
            [0, 0, 0, -4/np.sqrt(15)],
            [8/3, -8/(3*np.sqrt(5)), 0, 0]
        ])*np.pi*rho_w*a**4

        Gamma_mJ = np.diag([
            Omega_E*(1+kl2)/(A*sigma_0),
            Omega_E*(1+kl2)/(A*sigma_0),
            -(1+kl2)/C
        ])

        Phi_lm = (a*Omega_E)**2*np.array([
            [0, 0, 2/3],
            [0, 0, -2/(3*np.sqrt(5))],
            [-1/np.sqrt(15), 0, 0],
            [0, -1/np.sqrt(15), 0]
        ])

        T_N = (1+k2)/g*np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        T_U = (-h2)/g*np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        Maxtrixs = {
            "Psi":Psi_JT,
            "Gamma":Gamma_mJ,
            "Phi":Phi_lm,
            "N":T_N,
            "U":T_U,
        }
        return Maxtrixs


class SpectralSLE:
    '''
    This method is referring the research article of Tamisiea (2010);
    https://doi.org/10.1029/2009JC005687
    '''
    def __init__(self, SH, lmax=60):
        self.lmax = lmax
        self.shc = SHC(c=SH)
        self.res = 1
        self.tolerance = 1e-6
        self.lat, self.lon = MathTool.get_global_lat_lon_range(resolution=self.res)

    def setLatLon(self, lat, lon):
        # self.res = resolution
        self.lat, self.lon = lat, lon
        self.res = np.abs(lat[1] - lat[0])
        print(f"The configuration information:\n"
              f"lmax:{self.lmax},resolution:{self.res},lat:{self.lat.shape},lon:{self.lon.shape}")
        return self

    def Ocean_function(self, mask=None,loadfile="data/basin_mask/SH/Ocean_maskSH.dat"):

        if mask is not None:
            # grid_OceanFunction = xr.open_dataset('../../data/ref_sealevel/ocean_mask.nc')["ocean_mask"].values
            grid_OceanFunction = mask
            shc_OceanFunction = GRID(grid=grid_OceanFunction,lat=self.lat,lon=self.lon).to_SHC(lmax=360)
            N = int((self.lmax + 2) * (self.lmax + 1) / 2 + (self.lmax) * (self.lmax + 1) / 2)
            shc_OceanFunction = shc_OceanFunction.value[:,:N]

        else:
            OceanFuction_SH = FileTool.get_project_dir(loadfile)
            # load basin mask (in SHC)
            shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)
            grid_basin = shc_OceanFunction.to_grid(grid_space=self.res)
            grid_basin.limiter(threshold=0.5)
            grid_OceanFunction = grid_basin.value[0]
            shc_OceanFunction = shc_OceanFunction.value
        ocean_function = {
                "SH":shc_OceanFunction,
                "Grid":grid_OceanFunction
            }

        return ocean_function

    def GRDparameter(self, option=1):
        lln = LoveNumber().config(lmax=self.lmax, method=LLN_Data.PREM).get_Love_number()
        kl = lln.LLN[LLN_variable.k]
        hl = lln.LLN[LLN_variable.h]

        rho_water = GeoConstants.density_water
        rho_earth = GeoConstants.density_earth

        if option == 1:
            Green_N = []
            Green_U = []
            for i in np.arange(self.lmax + 1):
                for j in np.arange(-i, i + 1):
                    Green_N.append(3 * rho_water * (1 + kl[i]) / (rho_earth * (2 * i + 1)))
                    Green_U.append(3 * rho_water * (- hl[i]) / (rho_earth * (2 * i + 1)))
            Green_N = np.array(Green_N)
            Green_U = np.array(Green_U)
        else:
            Green_N = np.zeros((self.lmax + 1, self.lmax + 1))
            Green_U = np.zeros((self.lmax + 1, self.lmax + 1))
            for i in np.arange(self.lmax + 1):
                Green_N[i, 0:i + 1] = 3 * rho_water * (1 + kl[i]) / (rho_earth * (2 * i + 1))
                Green_U[i, 0:i + 1] = 3 * rho_water * (- hl[i]) / (rho_earth * (2 * i + 1))

        Green = {
            "N": Green_N,
            "U": Green_U,
        }
        return Green

    def GRDTerm(self, AL_SH, RSL_SH, rotation=False, option=1):
        Green_coef = self.GRDparameter(option=option)
        GreenN = Green_coef["N"]
        GreenU = Green_coef["U"]
        if option == 1:
            X_SH = SHC(c=GreenN * (AL_SH + RSL_SH))
            P_SH = SHC(c=GreenU * (AL_SH + RSL_SH))
        else:
            AL_C, AL_S = SHC(c=AL_SH).get_cs2d()
            RS_C, RS_S = SHC(c=RSL_SH).get_cs2d()
            X_C = GreenN * (AL_C + RS_C)
            X_S = GreenN * (AL_S + RS_S)
            P_C = GreenU * (AL_C + RS_C)
            P_S = GreenU * (AL_S + RS_S)
            X_SH = SHC(c=X_C, s=X_S)
            P_SH = SHC(c=P_C, s=P_S)

        if rotation:
            Rota_SH = self.RotationTerm(SH=AL_SH + RSL_SH)
            X_SH = X_SH.value + Rota_SH['N']
            X_SH = SHC(c=X_SH)
            P_SH = P_SH.value + Rota_SH['U']
            P_SH = SHC(c=P_SH)

        GRD_SH_value = X_SH.value + P_SH.value
        GRD_SH = SHC(c=GRD_SH_value)
        GRD_GRID = GRD_SH.to_grid(self.res)

        GRD = {
            "SH": GRD_SH.value,
            "N": X_SH.value,
            "U": P_SH.value,
            "Grid": GRD_GRID.value,
        }

        return GRD

    def BaryTerm(self, AL_SH, GRD ,mask=None):
        ocean_function = self.Ocean_function(mask=mask)
        ocean_mask = ocean_function["Grid"]
        Mask00 = ocean_function["SH"][0,0]
        AL_00 = AL_SH[:, 0]
        RO_SH = GRID(grid=GRD["Grid"] * ocean_mask, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value

        Baryterm = -(AL_00 + RO_SH[:, 0]) / Mask00
        # Baryterm = -(AL_00) / Mask00
        return Baryterm

    def QuasiRSLTerm(self,GRD,Bary,mask=None):
        """This is quasi-sea level"""
        ocean_function = self.Ocean_function(mask=mask)
        ocean_mask = ocean_function["Grid"]
        Bary = Bary*ocean_mask
        QuasiRSL = GRD["Grid"]+Bary
        QuasiRSL_SH = GRID(grid=QuasiRSL,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        Quasi = {
            "Grid":QuasiRSL,
            "SH":QuasiRSL_SH,
        }
        return Quasi

    def RSLTerm(self, GRD, Bary,mask=None):
        """This is relative sea level"""
        ocean_function = self.Ocean_function(mask=mask)
        ocean_mask = ocean_function["Grid"]
        Mask_SH = ocean_function["SH"]
        RSL = GRD["Grid"] * ocean_mask
        RSL_SH = GRID(grid=RSL, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        Bary_SH = Bary[:, None] * Mask_SH
        RSL_SH = RSL_SH + Bary_SH


        return RSL_SH

    def SLE(self,mask=None,rotation=False):
        print(f"=========Begin Quasi-Spectral SLF computing==========")
        start_time = time.time()
        ocean_function = self.Ocean_function(mask=mask)
        ocean_mask = ocean_function["Grid"]


        AL = self.shc.to_grid(grid_space=self.res).value * (1 - ocean_mask)
        AL_SH = GRID(grid=AL, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        # AL_SH = self.shc.value

        GRD = {"Grid":np.zeros_like(AL),
               "SH":np.zeros_like(AL_SH)}

        Bary = self.BaryTerm(AL_SH=AL_SH, GRD=GRD,mask=mask)
        RSL_SH = self.RSLTerm(GRD=GRD, Bary=Bary,mask=mask)
        for iter in np.arange(100):
            GRD = self.GRDTerm(AL_SH, RSL_SH,rotation=rotation)
            Bary = self.BaryTerm(AL_SH=AL_SH, GRD=GRD,mask=mask)
            new_RSL_SH = self.RSLTerm(GRD=GRD, Bary=Bary, mask=mask)

            delta = np.max(np.abs(new_RSL_SH - RSL_SH))
            print(f"The iteration is:{iter + 1},\n"
                  f"The delta is: {delta}")
            # print("delta:", delta, "Comparison:", delta < 1e-6, "All:", np.all(delta < 1e-6))
            if np.all(delta < 1e-6):
                break
            RSL_SH = new_RSL_SH
        RSL_SH = SHC(c=RSL_SH)
        RSL = RSL_SH.to_grid(self.res)
        SLE = {"RSL_SH": RSL_SH.value,
               "RSL": RSL.value,
               "BaryRSL": Bary,
               "GRD": RSL.value - Bary[:,None,None],
               "Load": RSL_SH.value+AL_SH}
        print(f"===Baryterm is: {Bary}\n")
        end_time = time.time()
        print(f"----------------------------------------------\n"
              f"-----time-consuming: {end_time - start_time:.4f} s-------\n"
              f"==============================================\n")
        return SLE

    def RotationTerm(self,SH):
        SH_00 = SH[:,0]
        SH_20 = SH[:,6]
        SH_21 = SH[:,7]
        SH_2m1 = SH[:,5]

        SH_vec = np.array([SH_00,SH_20,SH_21,SH_2m1])
        MatrixS = self.Rotation_matrix()
        # print(f"Matrix shape:{MatrixS['Psi'].shape,MatrixS['Gamma'].shape,MatrixS['Phi'].shape,MatrixS['T'].shape}")
        # print(f"SH shape is:{SH_vec.shape}")
        delta_J = MatrixS['Psi']@SH_vec

        m = MatrixS['Gamma']@delta_J

        delta_Lambda = MatrixS['Phi']@m

        S_rot = MatrixS['T']@delta_Lambda

        # print(f"new shape are: {delta_J.shape, m.shape, delta_Lambda.shape, S_rot.shape}")
        S_rot = S_rot.T
        SH_rotation = np.zeros_like(SH)
        SH_rotation[:,6] = S_rot[:,0]
        SH_rotation[:,7] = S_rot[:,1]
        SH_rotation[:,5] = S_rot[:,2]
        # print(f"rotation is:{SH_rotation.shape}")
        return SH_rotation
    def Rotation_matrix(self):
        a = EarthConstant.radiusm
        rho_w= EarthConstant.rhow
        Omega_E = EarthConstant.Omega
        A,C = EarthConstant.A, EarthConstant.C
        sigma_0 = EarthConstant.Chandler
        k2,h2 = EarthConstant.k2, EarthConstant.h2
        g = EarthConstant.grav
        lln = LoveNumber().config(lmax=self.lmax, method=LLN_Data.PREM).get_Love_number()
        kl2, hl2 = lln.LLN[LLN_variable.k][2], lln.LLN[LLN_variable.h][2]

        Psi_JT = np.array([
            [0, 0, -4/np.sqrt(15), 0],
            [0, 0, 0, -4/np.sqrt(15)],
            [8/3, -8/(3*np.sqrt(5)), 0, 0]
        ])*np.pi*rho_w*a**4

        Gamma_mJ = np.diag([
            Omega_E*(1+kl2)/(A*sigma_0),
            Omega_E*(1+kl2)/(A*sigma_0),
            -(1+kl2)/C
        ])

        Phi_lm = (a*Omega_E)**2*np.array([
            [0, 0, 2/3],
            [0, 0, -2/(3*np.sqrt(5))],
            [-1/np.sqrt(15), 0, 0],
            [0, -1/np.sqrt(15), 0]
        ])

        T_SL = (1+k2-h2)/g*np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        Maxtrixs = {
            "Psi":Psi_JT,
            "Gamma":Gamma_mJ,
            "Phi":Phi_lm,
            "T":T_SL
        }
        return Maxtrixs

class ConvolutionSLE:
    '''
    This method is referring the research article of Adhikari (2019);
    https://doi.org/10.5194/essd-11-629-2019
    '''
    def __init__(self,grid,lat,lon):
        '''
        Attention:
        lat here is latitude, not co-latitude, i.e., lat is from [-90,90];
        lon here is from [-180,180];
        '''

        self.tolerance = 1e-6
        self.lat,self.lon = lat,lon
        self.res = np.abs(lat[1]-lat[0])
        self.Load = GRID(grid=grid,lat=lat,lon=lon)
        self.lmax = int(180/self.res)

    def setmaxDegree(self,lmax):
        self.lmax = lmax
        print(f"The configuration information:\n"
              f"lmax:{self.lmax},resolution:{self.res},lat:{self.lat.shape},lon:{self.lon.shape}")
        return self
    def Ocean_function(self,loadfile="data/basin_mask/SH/Ocean_maskSH.dat"):
        OceanFuction_SH = FileTool.get_project_dir(loadfile)
        shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)  # load basin mask (in SHC)
        grid_basin = shc_OceanFunction.to_grid(grid_space=self.res)
        grid_basin.limiter(threshold=0.5)
        ocean_function = grid_basin.value[0]
        return ocean_function

    def BaryTerm(self,mask):
        ocean_mask = mask
        land_mask = 1-ocean_mask
        OceanArea = MathTool.get_acreage(basin=ocean_mask)
        E = -(self.Load.integral(mask=land_mask,average=False))/OceanArea
        return E

    def GRDTerm(self,WL,mask,rotation=False):
        ocean_mask = mask
        grids = PointLoad.Grids_generation.Equal_angular_distance(resolution=self.res)
        assert WL.ndim==3, "The dimension should be 3"
        Load = WL.reshape((len(WL),-1)).T
        grids['EWH'] = Load*EarthConstant.rhow

        lln = LoveNumber().config(lmax=self.lmax,method=LLN_Data.PREM).get_Love_number()
        gfa = PointLoad.GFA_regular_grid(lln=lln)
        gfa.configure(grids=grids)

        point_lon, point_lat = np.meshgrid(self.lon,self.lat)

        point = {
            'lat':point_lat.flatten(),
            'lon':point_lon.flatten(),
        }
        Bg = gfa.evaluation(points=point, variable=Displacement.Vertical, resolution=self.res).T
        EPg = gfa.evaluation(points=point,variable=Displacement.Geoheight,resolution=self.res).T
        Bg = Bg.reshape(len(WL), len(self.lat), len(self.lon))
        EPg = EPg.reshape(len(WL), len(self.lat), len(self.lon))
        # elif self.Green is GreenFunction.DiskLoad:
        #     rr = DiskLoad.grid2radius(lat_center=lat_2D,grid_size=self.res)
        #     grids = {
        #         'lat':lat_2D.flatten(),
        #         'lon':lon_2D.flatten(),
        #         'radius':rr[0].flatten(),
        #         'EWH': Load
        #     }
        #     gfa = DiskLoad.GFA_displacement(lln=lln)
        #     gfa.configure(grids=grids,cf=1000)
        #     Bg = gfa.evaluation(points=point,variable=Displacement.Vertical).T
        #     EPg = gfa.evaluation(points=point,variable=Displacement.Geoheight).T

        if rotation:
            Rota = self.RotationTerm(WL=WL)
            EPr = Rota["EP"]
            Br = Rota["B"]
            EP = EPr+EPg
            B = Br+Bg
        else:
            B = Bg
            EP = EPg

        Mean_GRD = GRID(grid=EP - B, lat=self.lat, lon=self.lon).integral(mask=ocean_mask, average=True)
        Mean_GRD = Mean_GRD[:,None,None]

        GRD = EP-B

        GRDterm = {"GRD":GRD,
                   "GRD_Mean":Mean_GRD,
                   "GHC":EP,
                   "VLM":B}
        return GRDterm

    def SLE(self,mask=None,rotation=False):
        print(f"=========Begin Green Convolution SLF computing==========")
        start_time = time.time()

        if mask is not None:
            ocean_mask = mask
        else:
            ocean_mask = self.Ocean_function()
        assert self.Load.value.ndim == 3, "The dimension should be 3"

        Baryterm = self.BaryTerm(mask=ocean_mask)
        RS = np.ones_like(self.Load.value) * Baryterm[:, None, None] * ocean_mask
        Baryterm = Baryterm[:, None, None]
        for iteration in np.arange(100):
            WL = self.Load.value*(1-ocean_mask)+RS
            GRDterm = self.GRDTerm(WL=WL,mask=ocean_mask,rotation=rotation)

            GRD = GRDterm['GRD']
            Mean_GRD = GRDterm['GRD_Mean']
            RS_new = (Baryterm+GRD-Mean_GRD)*ocean_mask
            delta = np.abs(np.linalg.norm(RS_new,axis=(1,2))-np.linalg.norm(RS,axis=(1,2)))
            print(f"The iteration is: {iteration+1},\n"
                  f"The delta is: {np.max(delta)}")
            if np.all(delta < 1e-6):
                break
            RS = RS_new
        RS = GRID(grid=RS,lat=self.lat,lon=self.lon)
        RS_SH = RS.to_SHC(lmax=self.lmax)

        SLE = {"RSL_SH":RS_SH.value,
               "RSL":RS.value,}

        print(f"===Baryterm is: {Baryterm}\n")
        end_time = time.time()
        print(f"----------------------------------------------\n"
              f"-----time-consuming: {end_time - start_time:.4f} s-------\n"
              f"==============================================\n")

        return SLE


    def RotationTerm(self,WL):
        h2 = EarthConstant.h2
        k2 = EarthConstant.k2
        grav = EarthConstant.grav
        self.lat,self.lon = np.array(self.lat),np.array(self.lon)
        SHF = Harmonic(lat=self.lat,lon=self.lon,lmax=self.lmax,option=1).get_spherical_harmonic_function()["Upsilon"]
        print(f"SHF shape is:{SHF.shape}")
        SH = self.InertiaTerm(WL=WL)["SH"]
        print(f"SH shape is:{SH.shape}")
        SH_copy = np.zeros_like(SH)
        SH_copy[:,0] = SH[:,0]

        SH[:,0] = 0

        grid2 = np.einsum("wjl,gl->gwj",SHF,SH)

        grid1 = np.einsum("wjl,gl->gwj",SHF,SH_copy)
        EPR = grid1+(1+k2)/grav*grid2
        BR = (h2/grav)*grid2
        print(f"The EP and B are: {EPR.shape} and {BR.shape}")
        rotation = {
            "EP": EPR,
            "B": BR
        }
        return rotation


    def InertiaTerm(self,WL):
        WL_SH = GRID(grid=WL, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        A,C = EarthConstant.A,EarthConstant.C
        Omega = EarthConstant.Omega
        varrho = EarthConstant.Chandler
        rhow = EarthConstant.rhow
        r = EarthConstant.radiusm
        lln = LoveNumber().config(lmax=self.lmax,method=LLN_Data.PREM).get_Love_number()
        kl2,hl2 = lln.LLN[LLN_variable.k][2], lln.LLN[LLN_variable.h][2]

        WL00, WL1_1, WL10, WL11, = WL_SH[:,0], WL_SH[:,1], WL_SH[:,2], WL_SH[:,3],
        WL2_2, WL2_1, WL20, WL21, W22 = WL_SH[:,4], WL_SH[:,5],WL_SH[:,6],WL_SH[:,7],WL_SH[:,8]

        factor1 = (r**2)*(Omega**2)
        factor2 = (Omega*(1+kl2))/(A*varrho)
        factor3 = rhow*(r**4)
        factor4 = -(1+kl2)/C

        Lambda21 = ((4*np.pi*factor1*factor2*factor3*WL21)/15)
        Lambda2_1 = ((4 * np.pi * factor1 * factor2 * factor3 * WL2_1) / 15)
        Lambda00 = ((16*np.pi*factor1*factor4*factor3*(WL00-WL20/np.sqrt(5)))/9)
        Lambda20 = ((-16 * np.pi * factor1 * factor4 * factor3*(WL00 - WL20 / np.sqrt(5))) / (9*np.sqrt(5)))
        print(
            f"Lambda shape and its len is: {Lambda00.shape},{len(Lambda00)},{Lambda00.reshape(len(Lambda00), -1).shape}")

        Lambda00 = Lambda00.reshape(len(Lambda00), -1)
        Lambda2_1 = Lambda2_1.reshape(len(Lambda2_1), -1)
        Lambda20 = Lambda20.reshape(len(Lambda20), -1)
        Lambda21 = Lambda21.reshape(len(Lambda21), -1)

        N = int((self.lmax+2)*(self.lmax+1)/2+(self.lmax)*(self.lmax+1)/2)
        Lambda = np.zeros((len(WL[:,0]),N))
        Lambda[:,0] = Lambda00
        Lambda[:,5] = Lambda2_1
        Lambda[:,6] = Lambda20
        Lambda[:,7] = Lambda21
        # factor2 = -(1+kl2)/C
        # factor3 = rhow*(r**4)*(8*np.pi/3)
        #
        # factor4 = Omega*(1+kl2)/(A*varrho)
        # factor5 = -4*np.pi*rhow*(r**4)/(np.sqrt(15))
        #
        # Lambda00 = 2/3*factor1*(factor2*factor3*(WL00-WL20/(np.sqrt(5))))
        # Lambda20 = -2*factor1/(3*np.sqrt(5))*factor2*factor3*(WL00-WL20/(np.sqrt(5)))
        #
        # Lambda2_1 = -factor1 / (np.sqrt(15)) * factor4 * (factor5 * WL2_1)
        # Lambda21 = -factor1 / (np.sqrt(15)) * factor4 * (factor5 * WL21)



        # # Inertia_arry = np.stack((Lambda00,Lambda2_1,Lambda00,Lambda21),axis=1)
        print(f"Inertia_arry shape is: {Lambda.shape}")

        Inertia= {
            "SH":Lambda
        }
        return Inertia

class PureSpectralSLE:
    '''
    This method is referring the PhD thesis of Roelof Rietbroek (2014);
    '''
    def __init__(self,SH,lmax=60):
        self.lmax = lmax
        self.shc = SHC(c=SH)
        self.res = 0.5
        self.tolerance = 1e-6
        self.lat, self.lon = MathTool.get_global_lat_lon_range(resolution=self.res)
    def setLatLon(self,lat,lon):
        self.lat,self.lon = lat,lon
        self.res = np.abs(lat[1]-lat[0])
        return self

    # def Ocean_function(self,,spatial=False):
    #     OceanFuction_SH = FileTool.get_project_dir(loadfile)
    #     # load basin mask (in SHC)
    #     shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)
    #     if spatial:
    #         grid_basin = shc_OceanFunction.to_grid(grid_space=self.res)
    #         grid_basin.limiter(threshold=0.5)
    #         ocean_function = grid_basin.value[0]
    #     else:
    #         ocean_function = shc_OceanFunction.value[0]
    #     return ocean_function

    def Ocean(self,loadfile="data/basin_mask/SH/Ocean_maskSH.dat",ocean_mask=None):

        if ocean_mask is not None:
            grid_OceanFunction = ocean_mask
            shc_OceanFunction = GRID(grid=grid_OceanFunction, lat=self.lat, lon=self.lon).to_SHC(lmax=self.lmax)

        else:
            OceanFuction_SH = FileTool.get_project_dir(loadfile)
            # load basin mask (in SHC)
            shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)
            grid_basin = shc_OceanFunction.to_grid(grid_space=self.res)
            grid_basin.limiter(threshold=0.5)
            grid_OceanFunction = grid_basin.value[0]
        Ocean = {
            "SH": shc_OceanFunction.value,
            "Grid": grid_OceanFunction
        }
        return Ocean

    def design_Matrix(self):
        lln = LoveNumber().config(lmax=self.lmax, method=LLN_Data.PREM).get_Love_number()
        kl = lln.LLN[LLN_variable.k]
        hl = lln.LLN[LLN_variable.h]

        rho_water = GeoConstants.density_water
        rho_earth = GeoConstants.density_earth
        Green_N_U = []
        for i in np.arange(self.lmax + 1):
            for j in np.arange(-i, i + 1):
                if i == 0:
                    Green_N_U.append(1)
                else:
                    Green_N_U.append(3 * rho_water * (1 + kl[i] - hl[i]) / (rho_earth * (2 * i + 1)))
        Green_N_U = np.array(Green_N_U)
        P = np.eye(len(Green_N_U))
        P[0,0]=0
        Green_NU = np.diag(Green_N_U)
        print(f"P and Green are:{P.shape,Green_NU.shape}")

        Matrix = {
            "Green_NU":Green_NU,
            "P":P
        }
        return Matrix

    def RSLTerm(self,QuasiRSL=None,Ocean=None):
        QuasiRSL_SH = SHC(c=QuasiRSL)
        QuasiRSL_Grid = QuasiRSL_SH.to_grid(grid_space=self.res)
        Os_Grid = QuasiRSL_Grid.value*Ocean["Grid"]
        Os_SH = GRID(grid=Os_Grid,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        Os = {
            "SH":Os_SH,
            "Grid":Os_Grid
        }
        return Os

    def RotationTerm(self, T):
        SH_00 = T[:, 0]
        SH_20 = T[:, 6]
        SH_21 = T[:, 7]
        SH_2m1 = T[:, 5]

        SH_vec = np.array([SH_00, SH_20, SH_21, SH_2m1])
        MatrixS = self.Rotation_matrix()
        # print(f"Matrix shape:{MatrixS['Psi'].shape,MatrixS['Gamma'].shape,MatrixS['Phi'].shape,MatrixS['T'].shape}")
        # print(f"SH shape is:{SH_vec.shape}")
        delta_J = MatrixS['Psi'] @ SH_vec

        m = MatrixS['Gamma'] @ delta_J

        delta_Lambda = MatrixS['Phi'] @ m

        S_rot = MatrixS['T'] @ delta_Lambda

        # print(f"new shape are: {delta_J.shape, m.shape, delta_Lambda.shape, S_rot.shape}")
        S_rot = S_rot.T
        E_NU = np.zeros((len(T[:,0]),len(T[0,:]),len(T[0,:])))
        E_NU[:, 6, 6] = S_rot[:, 0]
        E_NU[:, 7, 7] = S_rot[:, 1]
        E_NU[:, 5, 5] = S_rot[:, 2]
        print(f"rotation is:{E_NU.shape}")
        return E_NU

    def Rotation_matrix(self):
        a = EarthConstant.radiusm
        rho_w = EarthConstant.rhow
        Omega_E = EarthConstant.Omega
        A, C = EarthConstant.A, EarthConstant.C
        sigma_0 = EarthConstant.Chandler
        k2, h2 = EarthConstant.k2, EarthConstant.h2
        g = EarthConstant.grav
        lln = LoveNumber().config(lmax=self.lmax, method=LLN_Data.PREM).get_Love_number()
        kl2, hl2 = lln.LLN[LLN_variable.k][2], lln.LLN[LLN_variable.h][2]

        Psi_JT = np.array([
            [0, 0, -4 / np.sqrt(15), 0],
            [0, 0, 0, -4 / np.sqrt(15)],
            [8 / 3, -8 / (3 * np.sqrt(5)), 0, 0]
        ]) * np.pi * rho_w * a ** 4

        Gamma_mJ = np.diag([
            Omega_E * (1 + kl2) / (A * sigma_0),
            Omega_E * (1 + kl2) / (A * sigma_0),
            -(1 + kl2) / C
        ])

        Phi_lm = (a * Omega_E) ** 2 * np.array([
            [0, 0, 2 / 3],
            [0, 0, -2 / (3 * np.sqrt(5))],
            [-1 / np.sqrt(15), 0, 0],
            [0, -1 / np.sqrt(15), 0]
        ])

        T_SL = (1 + k2 - h2) / g * np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        Maxtrixs = {
            "Psi": Psi_JT,
            "Gamma": Gamma_mJ,
            "Phi": Phi_lm,
            "T": T_SL
        }
        return Maxtrixs
    def SLE(self,mask,rotation=False,OO=None):
        Matrix = self.design_Matrix()
        P = Matrix["P"]
        Green_NU = Matrix["Green_NU"]
        o = self.Ocean(ocean_mask=mask)["SH"]
        H = self.shc.value
        QuasiRSL = np.zeros_like(H)

        for iteration in np.arange(100):
            if rotation:
                E_NU = self.RotationTerm(T=self.shc.value + QuasiRSL @ OO)
                Gs = P[None, :, :] - ((Green_NU[None:, :] + E_NU) @ OO)
                Gs_inv = np.linalg.inv(Gs)
                temp = np.einsum("ik,ijk->ij", self.shc.value, Green_NU[None, :, :] - E_NU)
                new_QuasiRSL = np.einsum("ik,ijk->", temp, Gs_inv)
                # QuasiRSL = np.einsum('ik,ijk,ijl->il', self.shc.value, Gs_inv, Green_NU[None,:,:]-E_NU)
            else:
                Gs = P - Green_NU @ OO
                det = np.linalg.det(Gs)
                if np.isclose(det, 0):
                    raise ValueError("Matrix D is not invertible!")
                Gs_inv = np.linalg.inv(Gs)
                new_QuasiRSL = H  @ Green_NU.T @ Gs_inv.T

            # delta = H[:,0]+ new_QuasiRSL @ o.T
            delta = new_QuasiRSL-QuasiRSL
            if np.abs(np.max(delta))<1e-6:
                print(f"The iteration is: {iteration}")
                break
            QuasiRSL = new_QuasiRSL



    def design_Matrix_1D(self):
        lln = LoveNumber().config(lmax=self.lmax, method=LLN_Data.PREM).get_Love_number()
        kl = lln.LLN[LLN_variable.k]
        hl = lln.LLN[LLN_variable.h]

        rho_water = GeoConstants.density_water
        rho_earth = GeoConstants.density_earth
        Green_N_U = []
        for i in np.arange(self.lmax + 1):
            for j in np.arange(-i, i + 1):
                if i == 0:
                    Green_N_U.append(1)
                else:
                    Green_N_U.append(3 * rho_water * (1 + kl[i] - hl[i]) / (rho_earth * (2 * i + 1)))
        Green_N_U = np.array(Green_N_U)
        P = np.eye(len(Green_N_U))
        P[0,0]=0
        P = np.diag(P)
        print(P.shape,Green_N_U.shape)

        Matrix = {
            "Green_NU":Green_N_U,
            "P":P
        }
        return Matrix

class SeaLevelEquation_Old:
    def __init__(self):
        self.lmax = 60
        self.res = 2
        self.tolerance = 1e-6
        self.max_iter = 50
        self.lat,self.lon = MathTool.get_global_lat_lon_range(self.res)
        self.savefile = None
        pass
    def set_savefile(self,savefile):
        self.savefile = savefile
        return self
    def quasi_spectral_SLE(self,shc,ocean_mask):
        land_mask = 1-ocean_mask
        AL_grid = shc.to_grid(grid_space=self.res).value*(land_mask)
        AL_SH = GRID(grid=AL_grid,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        # Total_SH = shc.value

        # ocean_mask = self._get_Ocean_function()
        C_SH = GRID(grid=ocean_mask,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value



        Delta_SL = np.zeros_like(AL_grid)
        # Delta_SL_SH = GRID(grid=Delta_SL,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        Delta_Phi = self.compute_delta_phi(AL_SH=AL_SH,SL=Delta_SL,C_SH=C_SH)
        Delta_S_SH = self.compute_delta_S(SL=Delta_SL,Phi=Delta_Phi,C_SH=C_SH)
        print(f"Input information:\nSL-1: {Delta_SL.shape},\nPhi-1: {Delta_Phi.shape},{Delta_Phi[0]},\nSlm0: {Delta_S_SH.shape}")

        print(f"------Begin to iterating----------")
        for iter in tqdm(np.arange(self.max_iter)):
            Delta_SL = self.compute_delta_SL(AL_SH=AL_SH,Delta_S_SH=Delta_S_SH)
            Delta_Phi = self.compute_delta_phi(AL_SH=AL_SH,SL=Delta_SL,C_SH=C_SH)
            new_Delta_S_SH = self.compute_delta_S(SL=Delta_SL,Phi=Delta_Phi,C_SH=C_SH)

            if np.max(np.abs(new_Delta_S_SH-Delta_S_SH))<self.tolerance:
                print(f"The final iteration is: {iter}")
                break
            Delta_S_SH = new_Delta_S_SH
        print(f"----------Finished the iteration of Sea Level Equation---------\n")
        shc = SHC(c=Delta_S_SH)

        shc_grid = shc.to_grid(self.res)
        print(f"The shape of post-shc/grid is: {shc.value.shape},{shc_grid.value.shape}")
        # self._fig_verification(grid=shc_grid.value[0]*ocean_mask*100,maxvalue=2)

        return shc

    def compute_delta_SL(self,AL_SH,Delta_S_SH):
        KH_coef = self._get_KH_coef()
        # Delta_SL_clm = np.zeros_like(t)
        # Delta_SL_slm = np.zeros_like(S)
        AL_C,AL_S = SHC(c=AL_SH).get_cs2d()
        S_C,S_S = SHC(c=Delta_S_SH).get_cs2d()

        Delta_SL_C = KH_coef*(AL_C+S_C)
        Delta_SL_S = KH_coef*(AL_S+S_S)

        Delta_SL_SH = SHC(c=Delta_SL_C,s=Delta_SL_S)
        Delta_SL = Delta_SL_SH.to_grid(self.res).value
        # print(f"Delta_SL_SH is:{Delta_SL_SH.shape}")


        return Delta_SL

        pass

    def compute_delta_phi(self,AL_SH,SL,C_SH):
        AL_00 = AL_SH[:,0]
        C00 = C_SH[0,0]
        C = SHC(c=C_SH).to_grid(self.res).value[0]
        RO = SL*C
        RO_SH = GRID(grid=RO,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value

        Delta_phi = -(AL_00+RO_SH[:,0])/C00

        # RO_00 = SL_SH[0,0]*Mask_SH[0,0]

        # C_00 = Mask_SH[0,0]
        # Delta_phi = -(Delta_AL_00+RO_00)/C_00
        # print(f"Delta_phi is: {Delta_phi}")
        return Delta_phi

    def compute_delta_S(self,SL,Phi,C_SH):
        C = SHC(c=C_SH).to_grid(self.res).value[0]
        RO = SL*C
        RO_SH = GRID(grid=RO,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        PhiC_SH = Phi[:,np.newaxis]@C_SH
        new_Delta_S_SH = RO_SH+PhiC_SH
        # g = GeoConstants.g_wmo
        # Delta_SL_grid = SHC(c=Delta_SL_SH).to_grid(self.res).value
        # Mask_grid = SHC(c=Mask_SH).to_grid(self.res).value
        # RO_grid = Delta_SL_grid*Mask_grid
        # RO_SH = GRID(grid=RO_grid,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        #
        # Delta_phi_SH = Delta_phi/g*Mask_SH
        #
        # new_Delta_S_SH = RO_SH+Delta_phi_SH
        return new_Delta_S_SH

    def quasi_spatial_SLE(self,shc:SHC,ocean_mask):
        rho_water = GeoConstants.density_water
        land_mask = 1-ocean_mask
        lat,lon = MathTool.get_global_lat_lon_range(self.res)

        shc_grace = shc
        grid_grace = shc_grace.to_grid(self.res)

        Green_KH = self._get_KH_coef()
        Land_grid = GRID(grid=grid_grace.value * land_mask, lat=lat, lon=lon)
        Land_SH = Land_grid.to_SHC(self.lmax)

        SL_grid = GRID(grid=np.zeros_like(Land_grid.value),lat=lat,lon=lon)
        LandMass = rho_water*Land_grid.integral(mask=land_mask,average=False)
        SLCMass = rho_water*SL_grid.integral(mask=ocean_mask,average=False)
        OceanArea = MathTool.get_acreage(basin=ocean_mask)

        Phi = -(LandMass+SLCMass)/(OceanArea*rho_water)
        Delta_S_grid = (Phi[:,np.newaxis,np.newaxis]+SL_grid.value)*ocean_mask
        Delta_S_SH = GRID(grid=Delta_S_grid,lat=lat,lon=lon).to_SHC(self.lmax)
        print(f"Initial situation:\nSL-1: {SL_grid.value.shape};\nPhi-1: {Phi.shape},{Phi[0]};\nS0: {Delta_S_grid.shape}")
        print(f"-------Loading interation !!!!!-------")
        for iteration in np.arange(self.max_iter):
            delta_SL_C = Green_KH * (Land_SH.get_cs2d()[0][0] + Delta_S_SH.get_cs2d()[0][0])
            delta_SL_S = Green_KH * (Land_SH.get_cs2d()[1][0] + Delta_S_SH.get_cs2d()[1][0])

            delta_SL_SH = SHC(c=delta_SL_C, s=delta_SL_S)
            SL_grid = delta_SL_SH.to_grid(self.res)
            SLCMass = rho_water*SL_grid.integral(mask=ocean_mask,average=False)
            Phi = -(LandMass+SLCMass)/(OceanArea*rho_water)
            new_Delta_S_grid = (Phi[:,np.newaxis,np.newaxis]+SL_grid.value)*ocean_mask
            new_Delta_S_SH = GRID(grid=new_Delta_S_grid,lat=lat,lon=lon).to_SHC(self.lmax)

            sigma_stable = np.max(np.abs(new_Delta_S_SH.value - Delta_S_SH.value))
            if sigma_stable < self.tolerance:
                print(f"The convergence iteration is {iteration + 1}")
                break
            Delta_S_SH = new_Delta_S_SH
            # print(f"The shape of post-shc/grid is: {shc.value.shape},{shc_grid.value.shape}")
        # grid = Delta_S_SH.to_grid(self.grid_space)
        # self._fig_verification(grid=grid.value[0] * ocean_mask)
        return Delta_S_SH

    def _get_KH_coef(self, option=0):
        lln = LoveNumber().config(lmax=self.lmax,method=LLN_Data.PREM).get_Love_number()
        kl = lln.LLN[LLN_variable.k]
        hl = lln.LLN[LLN_variable.h]
        rho_water = GeoConstants.density_water
        rho_earth = GeoConstants.density_earth
        if option == 1:
            Green_N_U = []
            for i in np.arange(self.lmax + 1):
                for j in np.arange(-i, i + 1):
                    print(j)
                    if i == 0:
                        Green_N_U.append(1)
                    else:
                        Green_N_U.append(3 * rho_water * (1 + kl[i] - hl[i]) / (rho_earth * (2 * i + 1)))
            Green_N_U = np.array(Green_N_U)
        else:
            Green_N_U = np.zeros((self.lmax + 1, self.lmax + 1))
            for i in np.arange(self.lmax + 1):
                if i == 0:
                    Green_N_U[0, 0] = 1
                else:
                    Green_N_U[i, 0:i + 1] = 3 * rho_water * (1 + kl[i] - hl[i]) / (rho_earth * (2 * i + 1))
        return Green_N_U

    def _get_Ocean_function(self,loadfile="data/basin_mask/SH/Ocean_maskSH.dat"):
        OceanFuction_SH = FileTool.get_project_dir(loadfile)
        shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)  # load basin mask (in SHC)
        grid_basin = shc_OceanFunction.to_grid(grid_space=self.res)
        grid_basin.limiter(threshold=0.5)
        ocean_function = grid_basin.value[0]
        return ocean_function

def quick_fig(grid,lat=None,lon=None,maxvalue=2,savefile=None,unit="EWH (cm)"):
    import pygmt
    print(f"data of figure max/min:{np.max(grid)},{np.min(grid)}")

    fig_data = xr.DataArray(data=grid, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})

    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY="15p",FONT_LABEL="15p",MAP_FRAME_TYPE="plain",MAP_TITLE_OFFSET="-0.3c")
    pygmt.makecpt(cmap="haxby",series=[-maxvalue,maxvalue,maxvalue/10])
    fig.grdimage(grid=fig_data,projection="Q10c",cmap=True,frame=["a60f30"])
    fig.coast(shorelines="1/0.5p,black",resolution="f")
    fig.colorbar(position='JBC+o0c/1c+w8c+h',
                 frame=f"xa{maxvalue / 2}+l{unit}")
    # fig.text(position="BR",text=f"{var}",offset='-0.1c/0.2c', font='15p,Helvetica-Bold,black')
    if savefile:
        fig.savefig(savefile)
    fig.show()

def demo1():
    import netCDF4 as nc
    res = 2
    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"][:]
    dataset = nc.Dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WOUTrotation.nc")
    rotaset = nc.Dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WITHrotation.nc")
    lat = dataset["lat"][int(res)::int(res*2)]
    lon = dataset["lon"][int(res)::int(res*2)]
    Land_Load = dataset["weh"][0:1,int(res)::int(res*2),int(res)::int(res*2)]
    ref_rsl = dataset["rsl"][0,int(res)::int(res*2),int(res)::int(res*2)]
    ref_rota_rsl = rotaset["rsl"][0,int(res)::int(res*2),int(res)::int(res*2)]

    A = SpatialSLE(grid=Land_Load,lat=lat,lon=lon)
    A.setGreenFunctionType(kind=GreenFunction.PointLoad)
    # A.BaryTerm(mask=ocean_mask)
    # A.GRDTerm(WL=Land_Load[0:1,:,:],rotation=True,mask=None)
    # A.RotationTerm(WL=Land_Load[:,:,:])
    data = A.SLE(rotation=False)
    rsl = data['RSL'].value
    # baryterm = data['BaryRSL']
    # print(f"The sea level is {data.value.shape}")

    # A = ConvolutionSLE(grid=Land_Load,lat=lat,lon=lon).setmaxDegree(lmax=60)
    # data = A.SLE()
    # rsl = data['RSL'].value
    # # A.Rotationterm(WL=Land_Load)
    # quasi_rsl_rotation = A.SLE(rotation=True,mask=None)
    # A.GRDTerm(WL=Land_Load[None,:,:],mask=ocean_mask)
    # quasi_rsl = A.SLE(rotation=False)
    quick_fig(grid=100*(rsl[0]), lat=lat, lon=lon, maxvalue=2)
    # quick_fig(grid=100*baryterm[0], lat=lat, lon=lon,maxvalue=2)
    # land_ewh = dataset["weh"][0,int(res)::int(res*2),int(res)::int(res*2)]
    # land_ewh = dataset["weh"][0]
    # A = ConvolutionSLE(grid=land_ewh,lat=lat,lon=lon)
    # rsl = A.SLF(rotation=False,mask=ocean_mask)["RSL"].value
    # np.save("../../data/ref_sealevel/RSL_180.npy",rsl)
    # # A.Rotationterm(WL=land_ewh)
    # # A.InertiaTerm(WL=land_ewh)

    # ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")['ocean_mask'][:]
    # dataset = nc.Dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WOUTrotation.nc")
    # result = np.load("../../data/ref_sealevel/RSL_180.npy")
    # print(result.shape)
    # lat = dataset["lat"][:]
    # lon = dataset["lon"][:]
    # Gridrsl = dataset["rsl"][0]
    # Datas = [100 * Gridrsl * ocean_mask, 100 * result[0] * ocean_mask, 1000 * (Gridrsl - result[0]) * ocean_mask]
    #
    # Grids = []
    # for i in np.arange(len(Datas)):
    #     print(f"data of figure max/min:{np.max(Datas[i])},{np.min(Datas[i])}")
    #     grid = xr.DataArray(data=Datas[i], dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    #     Grids.append(grid)
    # frames = [["a60f30"], ["xa60f30", "ya0f30"], ["xa60f30", "ya0f30"]]
    # texts = ["(a) Reference", "(b) Green Convolution", "(c) a mius b"]
    # fig = pygmt.Figure()
    # pygmt.config(FONT_LABEL="15p", FONT_ANNOT_PRIMARY="15p", MAP_FRAME_TYPE='plain')
    # pygmt.makecpt(cmap="polar", series=[-2, 2, 0.2])
    # for i in np.arange(len(Grids)):
    #     if i == 2:
    #         fig.grdimage(grid=Grids[i], projection="Q10c", cmap=True, frame=frames[i])
    #         fig.coast(shorelines="1/0.5p,black", resolution='f')
    #         fig.colorbar(position="JBC+o-0.5c/1c+w9c/0.3c+h",
    #                      frame=["xa1f0.5", "y+lmm"])
    #         fig.text(position="BR", text=texts[i], offset='-0.1c/0.2c', font='17p,Helvetica-Bold,black')
    #     else:
    #         fig.grdimage(grid=Grids[i], projection="Q10c", cmap=True, frame=frames[i])
    #         fig.coast(shorelines="1/0.5p,black", resolution='f')
    #         fig.colorbar(position="JBC+o-0.5c/1c+w9c/0.3c+h",
    #                      frame=["xa1f0.5", "y+lcm"])
    #         fig.text(position="BR", text=texts[i], offset='-0.1c/0.2c', font='17p,Helvetica-Bold,black')
    #         fig.shift_origin(xshift="11c")
    # fig.savefig("I:/SeaLevel/ref_green.png")
    # fig.show()

def demo2():
    '''Input series SH but not load!'''
    # from demo.SLF.demo_SLF import Load_Grace
    # shc = Load_Grace()
    # SH = shc.value
    '''Input SH from spatial data'''
    # import netCDF4 as nc
    # dataset = nc.Dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WOUTrotation.nc")
    # ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"][:]
    # lat = dataset['lat'][:]
    # lon = dataset['lon'][:]
    # load = dataset['weh'][0]
    # load_SH = GRID(grid=load,lat=lat,lon=lon).to_SHC(lmax=180)
    # print(f"SH is: {SH.shape}")
    '''Input SH from reference'''
    from pysrc.LoadFile.LoadCS import LoadCS
    from pysrc.Auxiliary.EnumClasses import SLEReference
    import netCDF4 as nc
    from datetime import date
    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")['ocean_mask'][:]

    filepath = FileTool.get_project_dir('data/ref_sealevel/SLFsh_coefficients/GFZOP/CM/WOUTrotation/')
    begin_date, end_date = date(2003, 1, 1), date(2003, 2, 1)
    Load_SH = LoadCS().get_CS(filepath, begin_date=begin_date, end_date=end_date,
                              lmcs_in_queue=np.array([0, 1, 2, 4])).value

    SHrsl = LoadCS().get_CS(filepath,begin_date=begin_date,end_date=end_date,
                            lmcs_in_queue=np.array([0,1,6,8]))

    SGidrsl = SHrsl.to_grid(grid_space=0.5).value[0]

    # CG_RSL = nc.Dataset("../../data/ref_sealevel/GC_RSL_WOUTrotation.nc")["rsl"][:]
    CG_RSL = nc.Dataset("../../data/temp/Point_RSL_WOUTrotation.nc")["rsl"][:]
    # print(f"CG_RSL shape is: {CG_RSL.shape}")

    lat,lon = MathTool.get_global_lat_lon_range(0.5)

    # B = SpectralSLE_Old(SH=Load_SH,lmax=60).setLatLon(lat=lat,lon=lon)
    # GRDterm = {"Grid": np.zeros_like(SGidrsl),
    #        "SH": np.zeros_like(Load_SH)}
    # Bary = B.BaryTerm(AL_SH=Load_SH,GRD=GRDterm,mask=ocean_mask)
    # GRD_Old = B.GRDTerm(AL_SH=Load_SH,RSL_SH=GRDterm['SH'])['SH']
    # GRD_Old = SHC(c=GRD_Old).to_grid(grid_space=0.5).value[0]
    # print(f"Bary is:{Bary},")

    A = SpectralSLE(SH=Load_SH,lmax=60).setLatLon(lat=lat,lon=lon)
    # GRD_SH = np.zeros_like(Load_SH)
    # Bary = A.BaryTerm(AL_SH=Load_SH,GRD_SH=GRD_SH,mask=ocean_mask)
    # GRD = A.GRDTerm(AL_SH=Load_SH,RSL_SH=GRDterm['SH'])['SH']
    # GRD = SHC(c=GRD).to_grid(grid_space=0.5).value[0]
    # print(f"Barystatic is: {Bary['BarySH'].shape},{Bary['Baryterm']},{Bary['PureBary']}")

    # spatial_bary = SHC(c=Bary['BarySH']).to_grid(grid_space=0.5).value
    # quick_fig(grid=100*spatial_bary[0],lat=lat,lon=lon,maxvalue=0.5)
    # rslwith = A.SLE(rotation=True,mask=ocean_mask)["RSL"].value
    rslwout = A.SLE(rotation=False,mask=ocean_mask)['RSL'].value
    # quick_fig(grid=100*((rslwout[0]*ocean_mask)),lat=lat,lon=lon,maxvalue=2)
    # quick_fig(grid=1000*((rslwith[0]-rslwout[0])),lat=lat,lon=lon,maxvalue=2)
    quick_fig(grid=1000*((CG_RSL[0]-rslwout[0])*ocean_mask),lat=lat,lon=lon,maxvalue=2)

    # quick_fig(grid=100 * (GRD_Old), lat=lat, lon=lon, maxvalue=2)
    # quick_fig(grid=100 * (GRD), lat=lat, lon=lon, maxvalue=2)
    # quick_fig(grid=1000 * (GRD-GRD_Old), lat=lat, lon=lon, maxvalue=2)



    '''Rotation verify'''
    # print(f"Input sh is:{SH.shape}")
    # A.RotationTerm(SH=SH)

    # KH = A.SALparameter(option=1)
    # print(KH.shape)

def demo3():
    from pysrc.LoadFile.LoadCS import LoadCS
    from datetime import date
    filepath = FileTool.get_project_dir('data/ref_sealevel/SLFsh_coefficients/GFZOP/CM/WITHrotation/')
    begin_date,end_date = date(2003,1,1),date(2003,2,1)
    Load_SH = LoadCS().get_CS(filepath,begin_date=begin_date,end_date=end_date,lmcs_in_queue=np.array([0,1,2,4])).value

    print(f"Input SH is: {Load_SH.shape}")
    A = PureSpectralSLE(SH=Load_SH,lmax=60)
    Ma = A.design_Matrix()
    # A.Ocean()
    A.RotationTerm(T=Load_SH)
    # P = Ma["P"]
    # det = np.linalg.det(P)
    # if np.isclose(det,0):
    #     raise ValueError("P is not invertible")

def demo4():
    import netCDF4 as nc
    from pysrc.LoadFile.LoadCS import LoadCS
    from pysrc.Auxiliary.EnumClasses import SLEReference
    import pygmt
    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")['ocean_mask'][:]

    dataset = nc.Dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WOUTrotation.nc")
    Gridrsl = dataset["rsl"][0]
    SHrsl = LoadCS().get_CS(kind=SLEReference.RSL)
    SGidrsl = SHrsl.to_grid(grid_space=0.5).value[0]

    lat = dataset["lat"][:]
    lon = dataset["lon"][:]

    # quick_fig2(grid=100*(Gridrsl*ocean_mask),lat=lat,lon=lon,maxvalue=2)
    # quick_fig2(grid=100*(SGidrsl*ocean_mask),lat=lat,lon=lon,maxvalue=2)
    # quick_fig2(grid=1000*(Gridrsl-SGidrsl)*ocean_mask,lat=lat,lon=lon,maxvalue=2)
    Datas = [100*Gridrsl*ocean_mask,100*SGidrsl*ocean_mask,1000*(Gridrsl-SGidrsl)*ocean_mask]

    Grids = []
    for i in np.arange(len(Datas)):
        print(f"data of figure max/min:{np.max(Datas[i])},{np.min(Datas[i])}")
        grid = xr.DataArray(data=Datas[i],dims=["lat","lon"],coords={"lat":lat,"lon":lon})
        Grids.append(grid)
    frames = [["a60f30"],["xa60f30","ya0f30"],["xa60f30","ya0f30"]]
    texts = ["(a) Based on Grid","(b) Based on SH","(c) a mius b"]
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL="15p",FONT_ANNOT_PRIMARY="15p",MAP_FRAME_TYPE='plain')
    pygmt.makecpt(cmap="polar",series=[-2,2,0.2])
    for i in np.arange(len(Grids)):
        if i == 2:
            fig.grdimage(grid=Grids[i],projection="Q10c",cmap=True,frame=frames[i])
            fig.coast(shorelines="1/0.5p,black",resolution='f')
            fig.colorbar(position="JBC+o-0.5c/1c+w9c/0.3c+h",
                         frame=["xa1f0.5","y+lmm"])
            fig.text(position="BR",text=texts[i],offset='-0.1c/0.2c',font='17p,Helvetica-Bold,black')
        else:
            fig.grdimage(grid=Grids[i], projection="Q10c", cmap=True,frame=frames[i])
            fig.coast(shorelines="1/0.5p,black", resolution='f')
            fig.colorbar(position="JBC+o-0.5c/1c+w9c/0.3c+h",
                         frame=["xa1f0.5", "y+lcm"])
            fig.text(position="BR", text=texts[i], offset='-0.1c/0.2c', font='17p,Helvetica-Bold,black')
            fig.shift_origin(xshift="11c")
    fig.savefig("I:/SeaLevel/ref.png")
    fig.show()

def demo_rotation():
    import netCDF4 as nc
    dataset = nc.Dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WOUTrotation.nc")
    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")['ocean_mask'][:]
    lat = dataset["lat"][:]
    lon = dataset["lon"][:]
    land_ewh = dataset["weh"][0,:,:]
    print(f"Land EWH is: {land_ewh.shape}")
    A = ConvolutionSLE(grid=land_ewh,lat=lat,lon=lon).setmaxDegree(lmax=60)
    A.Rotationterm(WL=land_ewh)

def demo5(index=1):
    res = 2
    oceanmask = xr.open_dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"].values[int(res)::int(res * 2),
                int(res)::int(res * 2)]
    reference = xr.open_dataset('D:\Cheung\PyZWH\data/ref_sealevel/SLFgrids_GFZOP_CM_WOUTrotation.nc')
    lat = reference['lat'].values[int(res)::int(res * 2)]
    lon = reference['lon'].values[int(res)::int(res * 2)]
    time = reference['time'].values[0:index]
    input = reference['weh'].values[0:index, int(res)::int(res * 2), int(res)::int(res * 2)]
    RefRSL = reference['rsl'].values[0:index, int(res)::int(res * 2), int(res)::int(res * 2)]

    A = ConvolutionSLE(grid=input, lat=lat, lon=lon)
    result = A.SLE(mask=oceanmask, rotation=False)

    quick_fig(grid=100 * result['RSL'].value[0], lat=lat, lon=lon)






if __name__ == '__main__':
    # demo_rotation()
    # demo1()
    # demo2()
    demo5()
    # demo3()
    # demo4()
    # a = np.ones((31,100))
    # a1 = a[:,0]
    # print(a.shape)
    # b = np.array([a,a,a,a])
    # print(b.shape)
    # c = np.array([a1,a1,a1,a1])
    # print(c.shape)

    # lln = LoveNumber().config(lmax=60, method=LLN_Data.PREM).get_Love_number()
    # kl = lln.LLN[LLN_variable.k]
    # hl = lln.LLN[LLN_variable.h]
    # print(f"kl is: {kl[0]}")
