from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from SaGEA.auxiliary.preference.Constants import GeoConstants,EarthConstant
from pysrc.ancillary.geotools.LLN import LoveNumber,LLN_Data,LLN_variable,Frame
import numpy as np
from pysrc.ancillary.load_file.DataClass import SHC,GRID
import time


class PseudoSpectralSLE:
    def __init__(self, SH, lmax=60):
        self.shc = SHC(c=SH)
        self.lmax = lmax
        self.res = 1
        self.lat,self.lon = MathTool.get_global_lat_lon_range(self.res)
        self.lln = LoveNumber().config(lmax=self.lmax,method=LLN_Data.PREM).get_Love_number()
        self.OceanSpectral = None
        self.Frame = Frame.CE
        print(f"The initial configuration information:\n"
              f"lmax:{self.lmax}, resolution:{self.res},lat:{self.lat.shape},lon:{self.lon.shape},LLN:{LLN_Data.PREM.name}")
    def setLatLon(self,lat=None,lon=None):
        self.lat,self.lon = lat,lon
        self.res = np.abs(self.lat[1]-self.lat[0])
        print(f"\nThe update configuration information:\n"
              f"lmax:{self.lmax}, resolution:{self.res},lat:{self.lat.shape},lon:{self.lon.shape}")
        return self
    def setLoveNumber(self,lmax,method:LLN_Data.PREM,frame=Frame.CM):
        self.lln = LoveNumber().config(lmax=lmax,method=method).get_Love_number().convert(target=frame)
        print(f"\nThe Load Love Number here is up to degree {lmax}, method is {method.name}, and frame is {frame.name}")
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

    def SLE(self,mask=None,rotation=None,Gaunt=None,Psudo=True,isLand=True):
        print(f"=========Begin Spectral SLF computing==========")
        start_time = time.time()
        ocean_function = self.setOcean(ocean_mask=mask)
        ocean_mask = ocean_function['Grid']
        Mask_SH = ocean_function['SH']
        if isOnlyTWS:
            input_Grid = self.shc.to_grid(self.res).value*(1-ocean_mask)
            input_SH = GRID(grid=input_Grid,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        else:
            input_Grid = self.shc.to_grid(self.res).value
            input_SH = GRID(grid=input_Grid,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value

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
            print(f"The iteration is:{iter + 1},\n"
                  f"The delta is: {delta}")
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
        print(f"===Baryterm is: {BaryTerm}\n")
        end_time = time.time()
        print(f"----------------------------------------------\n"
              f"-----time-consuming: {end_time - start_time:.4f} s-------\n"
              f"==============================================\n")
        return SLE
def quick_fig(grid,lat=None,lon=None,maxvalue=2,savefile=None,unit="EWH (cm)"):
    import xarray as xr
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
    from pysrc.ancillary.load_file.LoadCS import LoadCS
    import netCDF4 as nc
    from datetime import date
    res=0.5
    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")['ocean_mask'][:]
    filepath = FileTool.get_project_dir('data/ref_sealevel/SLFsh_coefficients/GFZOP/CM/WOUTrotation/')
    begin_date, end_date = date(2003, 1, 1), date(2003, 2, 1)
    Load_SH = LoadCS().get_CS(filepath, begin_date=begin_date, end_date=end_date,
                              lmcs_in_queue=np.array([0, 1, 2, 4]))
    ReferenceRSL_SH = LoadCS().get_CS(filepath, begin_date=begin_date, end_date=end_date,
                            lmcs_in_queue=np.array([0, 1, 6, 8]))
    RefRSL_Grid = ReferenceRSL_SH.to_grid(grid_space=res).value[0]


    lat, lon = MathTool.get_global_lat_lon_range(res)

    A = PseudoSpectralSLE(SH=Load_SH.value, lmax=60).setLatLon(lat=lat, lon=lon)

    RSLwout = A.SLE(rotation=True, mask=ocean_mask)
    Quasi_SH = RSLwout['Quasi_RSL_SH']
    Quasi_Grid = SHC(c=Quasi_SH).to_grid(res)

    quick_fig(grid=100*(RSLwout['RSL'][0]),lat=lat,lon=lon,maxvalue=2)
    quick_fig(grid=100*(Quasi_Grid.value[0]),lat=lat,lon=lon,maxvalue=2)


    Load_Grid = Load_SH.to_grid(res).value[0]
    quick_fig(grid=(Load_Grid),lat=lat,lon=lon,maxvalue=1,unit="EWH (m)")

def demo2():
    from pysrc.sealevel_equation.SeaLevelEquation_Old import SpectralSLE
    from pysrc.ancillary.load_file.LoadCS import LoadCS
    import netCDF4 as nc
    from datetime import date
    res = 0.5
    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")['ocean_mask'][:]
    filepath = FileTool.get_project_dir('data/ref_sealevel/SLFsh_coefficients/GFZOP/CM/WOUTrotation/')
    begin_date, end_date = date(2003, 1, 1), date(2003, 2, 1)
    Load_SH = LoadCS().get_CS(filepath, begin_date=begin_date, end_date=end_date,
                              lmcs_in_queue=np.array([0, 1, 2, 4])).value
    ReferenceRSL_SH = LoadCS().get_CS(filepath, begin_date=begin_date, end_date=end_date,
                                      lmcs_in_queue=np.array([0, 1, 6, 8]))
    RefRSL_Grid = ReferenceRSL_SH.to_grid(grid_space=res).value[0]

    lat, lon = MathTool.get_global_lat_lon_range(res)

    A = SpectralSLE(SH=Load_SH, lmax=60).setLatLon(lat=lat, lon=lon)

    RSLwout = A.SLE(rotation=False, mask=ocean_mask)

    quick_fig(grid=100 * (RSLwout['RSL'].value[0]), lat=lat, lon=lon, maxvalue=2)
    # quick_fig(grid=100 * (Quasi_Grid.value[0]), lat=lat, lon=lon, maxvalue=2)


if __name__ == "__main__":
    demo1()





