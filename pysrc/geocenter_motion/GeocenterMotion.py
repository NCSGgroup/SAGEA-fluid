import numpy as np
from pysrc.ancillary.constant.GeoConstant import GCMConstant
from pysrc.ancillary.geotools.GeoMathKit import GeoMathKit
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from pysrc.ancillary.load_file.DataClass import SHC,GRID
import SaGEA.auxiliary.preference.EnumClasses as Enums
from pysrc.sealevel_equation.SeaLevelEquation import PseudoSpectralSLE
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.ancillary.geotools.LLN import LoveNumber
import time


def Convert_Mass_to_Coordinates(C10, C11, S11):
    k1 = 0.021
    rho_earth = GCMConstant.rho_earth
    X = np.sqrt(3) * (1 + k1) * C11 / rho_earth
    Y = np.sqrt(3) * (1 + k1) * S11 / rho_earth
    Z = np.sqrt(3) * (1 + k1) * C10 / rho_earth
    Coordinate = {"X": X, "Y": Y, "Z": Z}
    return Coordinate

def Convert_Stokes_to_Coordinates(C10, C11, S11):
    X = np.sqrt(3) * GCMConstant.radius * C11
    Y = np.sqrt(3) * GCMConstant.radius * S11
    Z = np.sqrt(3) * GCMConstant.radius * C10
    Coordinate = {"X": X, "Y": Y, "Z": Z}
    return Coordinate

class GeocenterMotion:
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
        self.frame = Enums.Frame.CM
    def setResolution(self,resolution):
        self.res = resolution
        self.lat,self.lon = MathTool.get_global_lat_lon_range(resolution)

        return self
    def setLatLon(self,lat,lon):
        self.lat,self.lon = lat,lon
        self.res = np.abs(self.lat[1]-self.lat[0])
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
    def __I_Matrix_Term(self, mask=None,buffer=0):
        N = len(self.GRACE.value)
        I = np.zeros((N, 3, 3))
        ocean_mask = self.setOcean(ocean_mask=mask)
        ocean_mask = GeoMathKit.leakage(ocean_mask=ocean_mask,lats=self.lat,buffer_width_km=buffer)
        theta, phi = MathTool.get_colat_lon_rad(lat=self.lat, lon=self.lon)
        Pilm = MathTool.get_Legendre(lat=theta, lmax=self.lmax, option=0)

        cosC10 = np.cos(0 * phi)
        cosC11 = np.cos(1 * phi)
        sinS11 = np.sin(1 * phi)


        CoreI10C = Pilm[:, 1, 0][:, None] * ocean_mask * cosC10[None, :]
        CoreI11C = Pilm[:, 1, 1][:, None] * ocean_mask * cosC11[None, :]
        CoreI11S = Pilm[:, 1, 1][:, None] * ocean_mask * sinS11[None, :]


        I_10C = GRID(grid=CoreI10C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_11C = GRID(grid=CoreI11C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_11S = GRID(grid=CoreI11S, lat=self.lat, lon=self.lon).to_SHC(self.lmax)

        I[:, 0, 0], I[:, 0, 1], I[:, 0, 2] = I_10C.value[:, 2], I_11C.value[:, 2], I_11S.value[:, 2]
        I[:, 1, 0], I[:, 1, 1], I[:, 1, 2] = I_10C.value[:, 3], I_11C.value[:, 3], I_11S.value[:, 3]
        I[:, 2, 0], I[:, 2, 1], I[:, 2, 2] = I_10C.value[:, 1], I_11C.value[:, 1], I_11S.value[:, 1]

        I = I
        # print("-------------Finished I Matrix computation-------------")
        return I
    def __G_Matrix_Term(self,mask=None,buffer=0):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:,0:4] = 0
        ocean_mask = (self.setOcean(ocean_mask=mask))
        ocean_mask = GeoMathKit.leakage(ocean_mask=ocean_mask,lats=self.lat,buffer_width_km=buffer)

        kernal = (SHC(c=GRACE_SH).to_grid(self.res).value)*ocean_mask
        G_SH = GRID(grid=kernal,lat=self.lat,lon=self.lon).to_SHC(self.lmax).value
        G = np.zeros((len(GRACE_SH),3))
        G[:,0] = G_SH[:,2]
        G[:,1] = G_SH[:,3]
        G[:,2] = G_SH[:,1]
        G = G
        # print("-------------Finished G Matrix computation-------------")
        return G
    def __Ocean_Model_Term(self,C10,C11,S11):
        GAD_Correct = self.GAD.value
        OM_SH = self.OceanSH.value
        OM = np.zeros((len(OM_SH),3))
        OM[:,0] = OM_SH[:,2]-GAD_Correct[:,2]+C10
        OM[:,1] = OM_SH[:,3]-GAD_Correct[:,3]+C11
        OM[:,2] = OM_SH[:,1]-GAD_Correct[:,1]+S11

        return OM
    def __GRD_Term(self,C10=None,C11=None,S11=None,mask=None,GRD=False,rotation=True):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:,1]=S11
        GRACE_SH[:,2]=C10
        GRACE_SH[:,3]=C11

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
        return UpdateTerm[:,2],UpdateTerm[:,3],UpdateTerm[:,1]
    def Low_Degree_Term(self,mask=None,GRD=False,rotation=True,buffer=0):
        """
        the series of Stokes coefficients follow: C10, C11, S11, C20, C21, S21
        that means, index 0->C10, 1->C11, 2->S11, 3->C20, 4->C21, 5->S21
        """
        print(f"======Begin GRACE Low Degree Terms computing=======")
        start_time = time.time()
        GRACE_SH = self.GRACE.value
        I_C10,I_C11,I_S11 = [np.zeros(len(GRACE_SH))]*3
        OM = self.__Ocean_Model_Term(C10=I_C10,C11=I_C11,S11=I_S11)
        I = self.__I_Matrix_Term(mask=mask,buffer=buffer)
        G = self.__G_Matrix_Term(mask=mask,buffer=buffer)

        I_inv = np.linalg.inv(I)
        C = np.einsum('nij,nj->ni',I_inv,OM-G)

        GRD_Ocean_Term = self.__GRD_Term(C10=C[:,0],C11=C[:,1],S11=C[:,2],
                                         mask=mask,GRD=GRD,rotation=rotation)
        for iter in np.arange(100):
            OM_new = self.__Ocean_Model_Term(C10=GRD_Ocean_Term[0],C11=GRD_Ocean_Term[1],S11=GRD_Ocean_Term[2])
            C_new = np.einsum('nij,nj->ni', I_inv, OM_new - G)
            delta = np.abs(C_new-C).flatten()
            if np.max(delta) < 10e-4:
                break
            C = C_new

        lln = LoveNumber().config(lmax=self.lmax, method=self.LLN_method).get_Love_number()
        lln.convert(target=self.frame)
        k = lln.LLN[Enums.LLN_variable.k]

        factor = 1.021/(GCMConstant.rho_earth*GCMConstant.radius)
        factor2 = (3+3*k[2])/(5*GCMConstant.rho_earth*GCMConstant.radius)
        # factor3 = (3+3*k[3])/(7*EarthConstant.rhoear*EarthConstant.radiusm)

        Mass_Coef = {"C10":C[:,0],"C11":C[:,1],"S11":C[:,2]}
        Stokes_Coef = {"C10":C[:,0]*factor,"C11":C[:,1]*factor,"S11":C[:,2]*factor}

        SH = {"Mass":Mass_Coef,"Stokes":Stokes_Coef}
        end_time = time.time()
        print('---------------------------------------------------\n'
              'GRACE-OBP: Geocenter motion estimation\n'
              '---------------------------------------------------')
        print('%-20s%-20s ' % ('Maxdegree:', f'{self.lmax}'))
        print('%-20s%-20s ' % ('Resolution:', f'{self.res}°'))
        print('%-20s%-20s ' % ('LoveNumber:', f'{self.LLN_method}'))
        print('%-20s%-20s ' % ('Frame:', f'{self.frame}'))
        print("%-20s%-20s " % ('SAL:',f'{GRD} (if False, omit rotation)'))
        print("%-20s%-20s " % ('Rotation feedback:', f'{rotation}'))
        print('%-20s%-20s ' % ('Iteration:', f'{iter + 1}'))
        print('%-20s%-20s ' % ('Convergence:', f'{np.max(delta)}'))
        print('%-20s%-20s ' % ('Time-consuming:', f'{end_time - start_time:.4f} s'))
        print(f"---------------------------------------------------")
        return SH
    def GSM_Like(self,mask=None,GRD=False,rotation=True,buffer=0):
        SH = self.Low_Degree_Term(mask=mask,GRD=GRD,rotation=rotation,buffer=buffer)
        C = SH['Mass']
        Coordinate = Convert_Mass_to_Coordinates(C10=C["C10"],C11=C["C11"],S11=C["S11"])
        print("-----------Finished GSM-like computation-----------\n"
              "===================================================\n")
        return Coordinate
    def Full_Geocenter(self,GAC=None,mask=None,GRD=False,rotation=True,buffer=0):
        GAC = SHC(c=GAC)
        GAC_Coordinate = Convert_Stokes_to_Coordinates(C10=GAC.value[:,2],C11=GAC.value[:,3],S11=GAC.value[:,1])
        SH = self.Low_Degree_Term(mask=mask,GRD=GRD,rotation=rotation,buffer=buffer)
        C = SH['Mass']
        GSM_Coordinate = Convert_Mass_to_Coordinates(C10=C["C10"], C11=C["C11"], S11=C["S11"])
        X = GAC_Coordinate['X']+GSM_Coordinate['X']
        Y = GAC_Coordinate['Y']+GSM_Coordinate['Y']
        Z = GAC_Coordinate['Z']+GSM_Coordinate['Z']
        full_geocenter = {"X":X,"Y":Y,"Z":Z}
        print("--------Finished Full-Geocenter computation--------\n"
              "===================================================\n")
        return full_geocenter

