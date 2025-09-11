from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from SaGEA.auxiliary.preference.Constants import GeoConstants,EarthConstant
from pysrc.ancillary.geotools.LLN import LoveNumber,LLN_Data,LLN_variable,Frame
import numpy as np
from pysrc.ancillary.load_file.DataClass import SHC,GRID
import time
from SaGEA.post_processing.harmonic.Harmonic import Harmonic
from SaGEA.auxiliary.preference.EnumClasses import Displacement,GreenFunction
from pysrc.basis_func import PointLoad,DiskLoad



class PseudoSpectralSLE:
    def __init__(self, SH, lmax=60):
        self.shc = SHC(c=SH)
        self.lmax = lmax
        self.res = 1
        self.lat,self.lon = MathTool.get_global_lat_lon_range(self.res)
        self.lln = LoveNumber().config(lmax=self.lmax,method=LLN_Data.PREM).get_Love_number()
        self.OceanSpectral = None
        self.Frame = Frame.CE

    def setLatLon(self,lat=None,lon=None):
        self.lat,self.lon = lat,lon
        self.res = np.abs(self.lat[1]-self.lat[0])

        return self
    def setLoveNumber(self,lmax,method:LLN_Data.PREM,frame=Frame.CM):
        self.Frame = frame
        self.lln = LoveNumber().config(lmax=lmax,method=method).get_Love_number().convert(target=frame)
        return self
    def setOcean(self,ocean_mask=None):
        if ocean_mask is not None:
            mask_grid = ocean_mask
            mask_sh = GRID(grid=mask_grid,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        else:
            OceanFuction_SH = FileTool.get_project_dir("data/basin_mask/SH/Ocean_maskSH.dat")
            mask_shc = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)
            grid_basin = mask_shc.to_grid(grid_space=self.res)
            grid_basin.limiter(threshold=0.5)
            mask_grid = grid_basin.value[0]
            mask_sh = mask_shc.value

        mask = {"SH":mask_sh,"Grid":mask_grid}
        return mask
    def BaryTerm(self,AL_SH,GRD_Grid,mask=None):
        '''Baryterm means only the mass change leads to the sea level changes, which also called eustatic term'''
        ocean_function = self.setOcean(ocean_mask=mask)
        ocean_grid = ocean_function['Grid']
        ocean_sh = ocean_function['SH']

        Mask00 = ocean_sh[0,0]
        AL00 = AL_SH[:,0]
        RO_SH = GRID(grid=GRD_Grid*ocean_grid,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value

        Bary = -(AL00+RO_SH[:,0])/Mask00
        return Bary
    def GRDparameter(self, option=0):
        lln = self.lln
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

    def GRDTerm(self, AL_SH, RSL_SH, rotation=False, option=0):
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
        X_Grid = X_SH.to_grid(self.res)

        GRD = {
            "SH": GRD_SH.value,
            "N": X_SH.value,
            "U": -P_SH.value,
            "Grid": GRD_GRID.value,
            "N_Grid":X_Grid.value
        }

        return GRD
    def RotationTerm(self,SH):
        SH_00 = SH[:,0]
        SH_20 = SH[:,6]
        SH_21 = SH[:,7]
        SH_2m1 = SH[:,5]

        SH_vec = np.array([SH_00,SH_20,SH_21,SH_2m1])
        MatrixS = self.Rotation_matrix()
        delta_J = MatrixS['Psi']@SH_vec

        m = MatrixS['Gamma']@delta_J

        delta_Lambda = MatrixS['Phi']@m

        Y_rot = MatrixS['N']@delta_Lambda

        P_rot = MatrixS['U']@delta_Lambda

        Y_rot = Y_rot.T
        SH_rotation_Y = np.zeros_like(SH)
        SH_rotation_Y[:,6] = Y_rot[:,0]
        SH_rotation_Y[:,7] = Y_rot[:,1]
        SH_rotation_Y[:,5] = Y_rot[:,2]

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
    def RSLTerm(self,GRD,Bary,mask=None):
        ocean_function = self.setOcean(ocean_mask=mask)
        ocean_mask = ocean_function["Grid"]
        Mask_SH = ocean_function["SH"]

        RSL = GRD['Grid'] * ocean_mask
        RSL_SH = GRID(grid=RSL, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        Bary_SH = Bary[:, None] * Mask_SH
        RSL_SH = RSL_SH + Bary_SH

        Land_SH = GRID(grid=GRD['Grid']*(1-ocean_mask),lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        Quasi_RSL_SH = Land_SH+RSL_SH

        Bary_Grid = SHC(c=Bary_SH).to_grid(self.res).value
        # print(f"Bary_Grid and GRD_N:{Bary_Grid.shape,GRD['N_Grid'].shape}")
        GHC_Grid = GRD['N_Grid']+Bary_Grid
        GHC_SH = GRID(grid=GHC_Grid,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value

        SL_SH = {"RSL_SH":RSL_SH,
                 "Quasi_RSL_SH":Quasi_RSL_SH,
                 "GHC_Grid":GHC_Grid,
                 "GHC_SH":GHC_SH}
        return SL_SH

    def SLE(self,mask=None,rotation=None,isLand=True):
        start_time = time.time()
        print(f"=========Begin Spectral SLE computing==========")
        ocean_function = self.setOcean(ocean_mask=mask)
        ocean_mask = ocean_function['Grid']
        Mask_SH = ocean_function['SH']
        if isLand:
            input_Grid = self.shc.to_grid(self.res).value * (1 - ocean_mask)
            input_SH = GRID(grid=input_Grid, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        else:
            input_Grid = self.shc.to_grid(self.res).value
            input_SH = GRID(grid=input_Grid, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value

        GRD = {"Grid":np.zeros_like(input_Grid),
               "SH":np.zeros_like(input_SH),
               "N": np.zeros_like(input_SH),
               "U": np.zeros_like(input_SH),
               "N_Grid":np.zeros_like(input_Grid)
               }
        GHC,VLM = None,None

        BaryTerm = self.BaryTerm(AL_SH=input_SH,GRD_Grid=GRD['Grid'],mask=mask)
        SL_SH = self.RSLTerm(GRD=GRD,Bary=BaryTerm,mask=mask)
        RSL_SH = SL_SH['RSL_SH']
        for iter in np.arange(100):
            GRD = self.GRDTerm(AL_SH=input_SH, RSL_SH=RSL_SH, rotation=rotation)
            BaryTerm = self.BaryTerm(AL_SH=input_SH, GRD_Grid=GRD['Grid'], mask=mask)
            SL_SH = self.RSLTerm(GRD=GRD, Bary=BaryTerm, mask=mask)
            new_RSL_SH =  SL_SH['RSL_SH']
            VLM = GRD['U']
            # GHC = SL_SH['GHC_SH']
            GHC = GRD['N']

            delta = np.max(np.abs(new_RSL_SH - RSL_SH))
            # print(f"The iteration is:{iter + 1},\n"
            #       f"The delta is: {delta}")
            # print("delta:", delta, "Comparison:", delta < 1e-6, "All:", np.all(delta < 1e-6))
            if np.all(delta < 1e-6):
                break
            RSL_SH = new_RSL_SH
        RSL_SH = SHC(c=RSL_SH)
        RSL = RSL_SH.to_grid(self.res)
        SLE = {"Input": input_SH,
               "RSL_SH": RSL_SH.value,
               "Quasi_RSL_SH":SL_SH['Quasi_RSL_SH'],
               "RSL":RSL.value*ocean_mask,
               "GHC":GHC,
               "VLM":VLM,
               "mask":np.repeat(Mask_SH,repeats=len(GHC),axis=0)
               }
        # print(f"===Baryterm is: {BaryTerm}\n")
        end_time = time.time()
        print('-------------------------------------------------\n'
              'Sea Level Equation: PseudoSpectral method\n'
              '-------------------------------------------------')
        print('%-20s%-20s ' % ('Maxdegree:', f'{self.lmax}'))
        print('%-20s%-20s ' % ('Resolution:', f'{self.res}°,i.e.,({len(self.lat),len(self.lon)})'))
        print('%-20s%-20s ' % ('LoveNumber:', f'{self.lln.method}'))
        print('%-20s%-20s ' % ('Frame:', f'{self.Frame}'))
        print('%-20s%-20s ' % ('Rotation:', f'{rotation}'))
        print('%-20s%-20s ' % ('Iteration:', f'{iter+1}'))
        print('%-20s%-20s ' % ('Convergence:', f'{delta}'))
        print('%-20s%-20s ' % ('Time-consuming:', f'{end_time - start_time:.4f} s'))
        print("-------------------------------------------------\n")
        return SLE

class SpatialSLE:
    def __init__(self,grid,lat,lon):
        self.lat,self.lon = lat,lon
        self.res = np.abs(lat[1] - lat[0])
        self.Input = GRID(grid=grid,lat=lat,lon=lon)
        self.lmax = int(180/self.res)
        self.lln = LoveNumber().config(lmax=self.lmax,method=LLN_Data.PREM).get_Love_number()
        self.Green = GreenFunction.PointLoad
        self.Frame = Frame.CE
        # print(f"The initial configuration information:\n"
        #       f"lmax:{self.lmax}, resolution:{self.res}, lat:{self.lat.shape}, lon:{self.lon.shape}, LLN:{LLN_Data.PREM.name}, GreenFunction:{self.Green.name}, Frame:{self.Frame.name}")
    def setLoveNumber(self,lmax,method:LLN_Data.PREM,frame:Frame.CM):
        self.Frame = frame
        self.lln = LoveNumber().config(lmax=lmax,method=method).get_Love_number().convert(target=frame)
        # print(f"The Load Love Number here is up to degree {lmax}, method is {method.name}, and frame is {frame.name}")
        return self
    def setmaxDegree(self,lmax):
        self.lmax = lmax
        # print(f"The update configuration information:\n"
        #       f"lmax:{self.lmax}, resolution:{self.res}, lat:{self.lat.shape}, lon:{self.lon.shape}\n")
        return self
    def setGreenFunctionType(self,kind:GreenFunction.PointLoad):
        self.Green = kind
        # print(f"The GreenFunction here is {self.Green.name}.")
        return self
    def setOcean(self,loadfile="data/basin_mask/SH/Ocean_maskSH.dat"):
        OceanFuction_SH = FileTool.get_project_dir(loadfile)
        shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)  # load basin mask (in SHC)
        grid_basin = shc_OceanFunction.to_grid(grid_space=self.res)
        grid_basin.limiter(threshold=0.5)
        ocean_function = grid_basin.value[0]
        return ocean_function
    def BaryTerm(self, mask):
        '''Also called eustatic term'''
        ocean_mask = mask
        land_mask = 1 - ocean_mask
        OceanArea = MathTool.get_acreage(basin=ocean_mask)
        E = -(self.Input.integral(mask=land_mask, average=False)) / OceanArea
        # print(f"Barystatic term is {type(E)},{E.shape}")
        E = E[:, np.newaxis, np.newaxis] * mask
        return E

    def GRDTerm(self, WL, mask, rotation=False):
        ocean_mask = mask
        assert WL.ndim == 3, "The dimension should be 3"
        Load = WL.reshape((len(WL), -1)).T
        print(f"Load shape is:{Load.shape}")
        lln = self.lln
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
            gfa.configure(grids=grids, cf=1000)
            Bg = gfa.evaluation(points=point, variable=Displacement.Vertical).T
            EPg = gfa.evaluation(points=point, variable=Displacement.Geoheight).T
        else:
            Bg, EPg = None, None
            assert self.Green in [GreenFunction.DiskLoad, GreenFunction.PointLoad], \
                "Please input right Green-Function! Using like setGreenFunction(kind=GreenFunction.PointLoad)!"

        Bg = Bg.reshape(len(WL), len(self.lat), len(self.lon))
        EPg = EPg.reshape(len(WL), len(self.lat), len(self.lon))

        if rotation:
            Rota = self.RotationTerm(WL=WL)
            EPr = Rota["EP"]
            Br = Rota["B"]
            EP = EPr + EPg
            B = Br + Bg
        else:
            B = Bg
            EP = EPg

        Mean_GRD = GRID(grid=EP - B, lat=self.lat, lon=self.lon).integral(mask=ocean_mask, average=True)
        Mean_GRD = Mean_GRD[:, np.newaxis, np.newaxis] * ocean_mask

        GRD = EP - B - Mean_GRD
        # print(f"The GRD is: {GRD.shape},\n"
        #       f"The Mean GRD: {Mean_GRD.shape},\n"
        #       f"The GHC is: {(EP + self.BaryTerm(mask=mask) - Mean_GRD).shape},\n"
        #       f"The VLM is: {B.shape}")

        # GRDterm = {"GRD": GRD,
        #            "GRD_Mean": Mean_GRD,
        #            "GHC": EP + self.BaryTerm(mask=mask) - Mean_GRD,
        #            "VLM": B}
        GRDterm = {"GRD": GRD,
                   "GRD_Mean": Mean_GRD,
                   "GHC": EP,
                   "VLM": B}
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
        SH[:, 0] = delta_Lambda[0, :]
        grid1 = np.einsum("wjl,gl->gwj", SHF, SH)

        SH[:, 0] = 0
        SH[:, 6] = delta_Lambda[1, :]
        SH[:, 7] = delta_Lambda[2, :]
        SH[:, 5] = delta_Lambda[3, :]

        grid2 = np.einsum("wjl,gl->gwj", SHF, SH)

        EPR = (grid1 + (1 + k2) * grid2) / grav
        BR = (h2 / grav) * grid2
        print(f"The EP and B are: {EPR.shape} and {BR.shape}")
        rotation = {
            "EP": EPR,
            "B": BR
        }
        return rotation

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

    def SLE(self, mask=None, rotation=False):
        print(f"=========Begin Spatial SLE computing==========")
        start_time = time.time()

        if mask is not None:
            ocean_mask = mask
        else:
            ocean_mask = self.setOcean()
        assert self.Input.value.ndim == 3, "The dimension should be 3"
        # print(f"ocean_mask is:{ocean_mask.shape}")

        Baryterm = self.BaryTerm(mask=ocean_mask)
        S = np.zeros_like(self.Input.value) + Baryterm
        GHC, VLM = None, None
        for iteration in np.arange(100):
            WL = self.Input.value * (1 - ocean_mask) + S * ocean_mask
            GRDterm = self.GRDTerm(WL=WL, mask=ocean_mask, rotation=rotation)
            GHC, VLM = GRDterm['GHC'], GRDterm['VLM']
            GRD = GRDterm['GRD']
            S_new = Baryterm + GRD
            # delta = np.abs(
            #     np.linalg.norm(S_new * ocean_mask, axis=(1, 2)) - np.linalg.norm(S * ocean_mask, axis=(1, 2)))
            delta = np.abs(np.amax(S_new-S,axis=(1,2)))
            # print(f"The iteration is: {iteration + 1},\n"
            #       f"The delta is: {np.max(delta)}")
            if np.all(delta < 1e-6):
                break
            S = S_new

        S = GRID(grid=S, lat=self.lat, lon=self.lon)
        S_SH = S.to_SHC(lmax=self.lmax)

        SLE = {"Input": self.Input.value,
               "RSL_SH": S_SH.value,
               "RSL": S.value,
               "GHC": GHC,
               "VLM": VLM,
               # "BaryRSL": Baryterm,
               "mask": np.repeat(ocean_mask,repeats=len(GHC),axis=0)}

        # print(f"===Baryterm is: {np.amin(Baryterm,axis=(1,2))}\n")
        end_time = time.time()
        print('-------------------------------------------------\n'
              'Sea Level Equation: PseudoSpectral method\n'
              '-------------------------------------------------')
        print('%-20s%-20s ' % ('Maxdegree:', f'{self.lmax}'))
        print('%-20s%-20s ' % ('PointLoad:', f'{self.Green}'))
        print('%-20s%-20s ' % ('Resolution:', f'{self.res}°'))
        print('%-20s%-20s ' % ('LoveNumber:', f'{self.lln.method}'))
        print('%-20s%-20s ' % ('Frame:', f'{self.Frame}'))
        print('%-20s%-20s ' % ('Iteration:', f'{iteration + 1}'))
        print('%-20s%-20s ' % ('Convergence:', f'{np.max(delta)}'))
        print('%-20s%-20s ' % ('Time-consuming:', f'{end_time - start_time:.4f} s'))
        print("-------------------------------------------------\n")

        return SLE
