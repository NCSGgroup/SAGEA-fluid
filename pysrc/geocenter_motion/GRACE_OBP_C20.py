import numpy as np
from pysrc.load_file.DataClass import SHC,GRID
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.sealevel_equation.SeaLevelEquation import PseudoSpectralSLE
import SaGEA.auxiliary.preference.EnumClasses as Enums
from pysrc.Auxiliary.LLN import LoveNumber
from SaGEA.auxiliary.preference.Constants import EarthConstant
import time


def Convert_Mass_to_Coordinates(C10, C11, S11):
    k1 = 0.021
    rho_earth = EarthConstant.rhoear
    X = np.sqrt(3) * (1 + k1) * C11 / rho_earth
    Y = np.sqrt(3) * (1 + k1) * S11 / rho_earth
    Z = np.sqrt(3) * (1 + k1) * C10 / rho_earth
    Coordinate = {"X": X, "Y": Y, "Z": Z}
    return Coordinate


def Convert_Stokes_to_Coordinates(C10, C11, S11):
    X = np.sqrt(3) * EarthConstant.radiusm * C11
    Y = np.sqrt(3) * EarthConstant.radiusm * S11
    Z = np.sqrt(3) * EarthConstant.radiusm * C10
    Coordinate = {"X": X, "Y": Y, "Z": Z}
    return Coordinate


class GRACE_OBP_V2:
    '''
    Attention:
    the coefficients type of input data about GRACE and OceanModel is mass coefficients;
    the coefficients type of GAD and GAC are original coefficients, i.e., Stokes coefficients (gravity spherical harmonics);
    '''
    def __init__(self,GRACE,OceanSH=None,GAD=None,GAC=None,lmax=60):
        '''
        GRACE and OceanSH are the mass coefficients,
        GAD and GAC are the Stokes coefficients;
        '''
        self.GRACE = SHC(GRACE)
        self.OceanModel = SHC(OceanSH)
        self.GAD = SHC(GAD)
        self.lmax = lmax
        self.GAC = SHC(GAC)
        self.res = 1
        self.lat,self.lon = MathTool.get_global_lat_lon_range(self.res)

        self.LLN_method = Enums.LLN_Data.PREM
        self.frame = Enums.Frame.CF

    def setLatLon(self, lat=None, lon=None):
        self.lat, self.lon = lat, lon
        self.res = np.abs(self.lat[1] - self.lat[0])
        print(f"The update configuration information:\n"
              f"lmax:{self.lmax}, resolution:{self.res},lat:{self.lat.shape},lon:{self.lon.shape}")
        return self

    def setResolution(self,resolution):
        self.res = resolution
        self.lat, self.lon = MathTool.get_global_lat_lon_range(resolution)
        print(f"\nThe update configuration information:\n"
              f"lmax:{self.lmax}, resolution:{self.res},lat:{self.lat.shape},lon:{self.lon.shape}")
        return self

    def setOcean(self,ocean_mask=None):
        if ocean_mask is not None:
            mask_grid = ocean_mask
            # mask_sh = GRID(grid=mask_grid, lat=self.lat, lon=self.lon).to_SHC(self.lmax).value
        else:
            OceanFuction_SH = FileTool.get_project_dir("data/basin_mask/SH/Ocean_maskSH.dat")
            mask_shc = load_SHC(OceanFuction_SH, key='', lmax=self.lmax)
            grid_basin = mask_shc.to_grid(grid_space=self.res)
            grid_basin.limiter(threshold=0.5)
            mask_grid = grid_basin.value[0]
            # mask_sh = mask_shc.value
        # mask = {"SH": mask_sh, "Grid": mask_grid}
        return mask_grid

    def setLoveNumber(self, method: Enums.LLN_Data.PREM, frame: Enums.Frame.CF):
        self.LLN_method = method
        self.frame = frame
        return self

    def GAC_Convert_Term(self):
        GAC_CF = np.zeros((len(self.GAC.value), 5))
        lln = LoveNumber().config(lmax=self.lmax,method=self.LLN_method).get_Love_number()
        lln.convert(target=self.frame)
        k = lln.LLN[Enums.LLN_variable.k]
        GAC_CF[:,0] = self.GAC.value[:,2]*(1+k[1])
        GAC_CF[:,1] = self.GAC.value[:,3]*(1+k[1])
        GAC_CF[:,2] = self.GAC.value[:,1]*(1+k[1])
        GAC_CF[:,3] = self.GAC.value[:,6]
        GAC_CF[:,4] = self.GAC.value[:,12]
        # print(f"GAC Term: {GAC_CF[0]}")

        return GAC_CF

    def GAD_Convert_Term(self):
        GAD = self.GAD.value
        GAD_CF = np.zeros((len(GAD),5))
        lln = LoveNumber().config(lmax=self.lmax, method=self.LLN_method).get_Love_number()
        lln.convert(target=self.frame)
        k = lln.LLN[Enums.LLN_variable.k]
        GAD_CF[:,0] = GAD[:,2] * (EarthConstant.radiusm * EarthConstant.rhoear) / (1 + k[1])
        GAD_CF[:,1] = GAD[:,3] * (EarthConstant.radiusm * EarthConstant.rhoear) / (1 + k[1])
        GAD_CF[:,2] = GAD[:,1] * (EarthConstant.radiusm * EarthConstant.rhoear) / (1 + k[1])
        GAD_CF[:,3] = GAD[:,6] * (5*EarthConstant.radiusm * EarthConstant.rhoear) / (3 + 3*k[2])
        GAD_CF[:,4] = GAD[:,12] * (7*EarthConstant.radiusm * EarthConstant.rhoear) / (3 + 3*k[3])
        # print(f"GAD Term: {GAD_CF[0]}")

        return GAD_CF

    def I_Matrix_Term(self,mask=None):
        N = len(self.GRACE.value)
        I = np.zeros((N,4,4))
        ocean_mask = self.setOcean(ocean_mask=mask)
        theta, phi = MathTool.get_colat_lon_rad(lat=self.lat, lon=self.lon)
        Pilm = MathTool.get_Legendre(lat=theta, lmax=self.lmax, option=0)

        cosC10 = np.cos(0 * phi)
        cosC11 = np.cos(1 * phi)
        sinS11 = np.sin(1 * phi)
        cosC20 = np.cos(0 * phi)
        # cosC30 = np.cos(0 * phi)

        CoreI10C = Pilm[:, 1, 0][:, None] * ocean_mask * cosC10[None, :]
        CoreI11C = Pilm[:, 1, 1][:, None] * ocean_mask * cosC11[None, :]
        CoreI11S = Pilm[:, 1, 1][:, None] * ocean_mask * sinS11[None, :]
        CoreI20C = Pilm[:, 2, 0][:, None] * ocean_mask * cosC20[None, :]
        # CoreI30C = Pilm[:, 3, 0][:, None] * ocean_mask * cosC30[None, :]

        I_10C = GRID(grid=CoreI10C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_11C = GRID(grid=CoreI11C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_11S = GRID(grid=CoreI11S, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        I_20C = GRID(grid=CoreI20C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)
        # I_30C = GRID(grid=CoreI30C, lat=self.lat, lon=self.lon).to_SHC(self.lmax)

        I[:,0,0],I[:,0,1],I[:,0,2],I[:,0,3] = I_10C.value[:,2],I_11C.value[:,2],I_11S.value[:,2],I_20C.value[:,2]

        I[:,1,0],I[:,1,1],I[:,1,2],I[:,1,3] = I_10C.value[:,3],I_11C.value[:,3],I_11S.value[:,3],I_20C.value[:,3]

        I[:,2,0],I[:,2,1],I[:,2,2],I[:,2,3] = I_10C.value[:,1],I_11C.value[:,1],I_11S.value[:,1],I_20C.value[:,1]

        I[:,3,0],I[:,3,1],I[:,3,2],I[:,3,3] = I_10C.value[:,6],I_11C.value[:,6],I_11S.value[:,6],I_20C.value[:,6]

        # I[:, 0, 0], I[:, 0, 1], I[:, 0, 2], I[:, 0, 3], I[:, 0, 4]  = \
        #     I_10C.value[:, 2], I_11C.value[:, 2], I_11S.value[:, 2], I_20C.value[:,2], I_30C.value[:,2]

        # I[:, 1, 0], I[:, 1, 1], I[:, 1, 2], I[:, 1, 3], I[:, 1, 4] = \
        #     I_10C.value[:, 3], I_11C.value[:, 3], I_11S.value[:, 3], I_20C.value[:,3], I_30C.value[:,3]

        # I[:, 2, 0], I[:, 2, 1], I[:, 2, 2], I[:, 2, 3], I[:, 2, 4]= \
        #     I_10C.value[:, 1], I_11C.value[:, 1], I_11S.value[:, 1], I_20C.value[:,1], I_30C.value[:,1]

        # I[:, 3, 0], I[:, 3, 1], I[:, 3, 2], I[:, 3, 3], I[:, 3, 4]= \
        #     I_10C.value[:, 4], I_11C.value[:, 4], I_11S.value[:, 4], I_20C.value[:,4], I_30C.value[:,4]

        # I[:, 4, 0], I[:, 4, 1], I[:, 4, 2], I[:, 4, 3], I[:, 4, 4] = \
        #     I_10C.value[:, 12], I_11C.value[:, 12], I_11S.value[:, 12], I_20C.value[:,12], I_30C.value[:, 12]
        I = I
        # print(f"I Matrix shape is: {I.shape}")
        # print(f'I Matrix is: \n{I[0]},\n{I[1]}')
        print("-------------Finished I Matrix computation-------------")
        return I

    def G_Matrix_Term(self,mask=None,SLE=False):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:,0:4] = 0
        GRACE_SH[:,6] = 0
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
        G = np.zeros((len(GRACE_SH),4))
        G[:,0] = G_SH[:,2]
        G[:,1] = G_SH[:,3]
        G[:,2] = G_SH[:,1]
        G[:,3] = G_SH[:,6]
        # G[:,4] = G_SH[:,12]
        G = G
        # print(f"G V2 is: {G[0]}")
        print("-------------Finished G Matrix computation-------------")
        return G

    def Ocean_Model_Term(self,C10,C11,S11,C20):
        GAD_Correct = self.GAD_Convert_Term()
        OM_SH = self.OceanModel.value
        OM = np.zeros((len(OM_SH),4))
        # print(OM.shape)
        OM[:,0] = OM_SH[:,2]-GAD_Correct[:,0]+C10
        OM[:,1] = OM_SH[:,3]-GAD_Correct[:,1]+C11
        OM[:,2] = OM_SH[:,1]-GAD_Correct[:,2]+S11
        OM[:,3] = OM_SH[:,6]-GAD_Correct[:,3]+C20
        # OM[:,4] = OM_SH[:,12]-GAD_Correct[:,4]+C30
        # print(f"OceanModel Term: {OM[0]}")
        return OM

    def GRD_Term(self,C10=None,C11=None,S11=None,C20=None,mask=None,GRD=False,rotation=True):
        GRACE_SH = self.GRACE.value
        GRACE_SH[:,1]=S11
        GRACE_SH[:,2]=C10
        GRACE_SH[:,3]=C11
        GRACE_SH[:,6]=C20
        # GRACE_SH[:,12]=C30
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
        return UpdateTerm[:,2],UpdateTerm[:,3],UpdateTerm[:,1],UpdateTerm[:,6]
        # return UpdateTerm[:, 2], UpdateTerm[:, 3], UpdateTerm[:, 1], UpdateTerm[:, 4], UpdateTerm[:, 12]

    def Low_Degree_Term(self,mask=None,GRD=False,rotation=True,SLE=False):
        """
        the series of Stokes coefficients follow: C10, C11, S11, C20,
        that means, index 0->C10, 1->C11, 2->S11, 3->C20
        """
        print(f"=========Begin Geocenter Motion computing==========")
        start_time = time.time()
        GRACE_SH = self.GRACE.value
        I_C10,I_C11,I_S11,I_C20 = [np.zeros(len(GRACE_SH))]*4
        OM = self.Ocean_Model_Term(C10=I_C10,C11=I_C11,S11=I_S11,C20=I_C20)
        I = self.I_Matrix_Term(mask=mask)
        G = self.G_Matrix_Term(mask=mask,SLE=SLE)

        I_inv = np.linalg.inv(I)
        # print(f"I and I_inv:\n{I[0]}\n\n{I_inv[0]}")
        # print(f"verfiy: {I[0]@I_inv[0]}")
        C = np.einsum('nij,nj->ni',I_inv,OM-G)

        GRD_Ocean_Term = self.GRD_Term(C10=C[:,0],C11=C[:,1],S11=C[:,2],C20=C[:,3],
                                       mask=mask,GRD=GRD,rotation=rotation)
        for iter in np.arange(100):
            OM_new = self.Ocean_Model_Term(C10=GRD_Ocean_Term[0],C11=GRD_Ocean_Term[1],
                                           S11=GRD_Ocean_Term[2],C20=GRD_Ocean_Term[3])
            C_new = np.einsum('nij,nj->ni', I_inv, OM_new - G)
            delta = np.abs(C_new-C).flatten()
            if np.max(delta) < 10e-4:
                print(f"----------------------------------------------------\nIterative number is: {iter + 1}")
                break
            C = C_new

        lln = LoveNumber().config(lmax=self.lmax, method=self.LLN_method).get_Love_number()
        lln.convert(target=self.frame)
        k = lln.LLN[Enums.LLN_variable.k]

        factor = 1.021/(EarthConstant.rhoear*EarthConstant.radiusm)
        factor2 = (3+3*k[2])/(5*EarthConstant.radiusm*EarthConstant.rhoear)
        factor3 = (3+3*k[3])/(7*EarthConstant.rhoear*EarthConstant.radiusm)
        print(f"Love numbers degree-1:{k[1]},degre-2:{k[2]},degree-3:{k[3]}")
        Mass_Coef = {"C10": C[:, 0], "C11": C[:, 1],
                     "S11": C[:, 2], "C20": C[:,3]}
        Stokes_Coef = {"C10": C[:, 0] * factor, "C11": C[:, 1] * factor,
                       "S11": C[:, 2] * factor," C20": C[:, 3] * factor2}
        # Mass_Coef = {"C10":C[:,0],"C11":C[:,1],"S11":C[:,2],"C20":C[:,3],"C30":C[:,4]}
        # Stokes_Coef = {"C10":C[:,0]*factor,"C11":C[:,1]*factor,"S11":C[:,2]*factor,
        #                "C20":C[:,3]*factor2,"C30":C[:,4]*factor3}

        SH = {"Mass":Mass_Coef,"Stokes":Stokes_Coef}
        end_time = time.time()
        print(f"----------------------------------------------\n"
              f"-----time-consuming: {end_time - start_time:.4f} s-------\n"
              f"==============================================\n")
        return SH




    def GSM_Like(self,mask=None,GRD=False,rotation=True,SLE=False):
        SH = self.Low_Degree_Term(mask=mask,GRD=GRD,rotation=rotation,SLE=SLE)
        C = SH['Mass']
        Coordinate = Convert_Mass_to_Coordinates(C10=C["C10"],C11=C["C11"],S11=C["S11"])
        print("-------------Finished GSM-like computation-------------\n"
              "==========================================================")
        return Coordinate

    def Full_Geocenter(self,mask=None,GRD=False,rotation=True,SLE=False):
        GAC = self.GAC_Convert_Term()
        GAC_Coordinate = Convert_Stokes_to_Coordinates(C10=GAC[:, 0], C11=GAC[:, 1], S11=GAC[:, 2])
        # print(f"X/Y/Z:\n{coordinate['X']}\n\n{coordinate['Y']}\n\n{coordinate['Z']}")
        SH = self.Low_Degree_Term(mask=mask, GRD=GRD, rotation=rotation,SLE=SLE)
        C = SH['Mass']
        GSM_Coordinate = Convert_Mass_to_Coordinates(C10=C["C10"], C11=C["C11"], S11=C["S11"])
        X = GAC_Coordinate['X']+GSM_Coordinate['X']
        Y = GAC_Coordinate['Y']+GSM_Coordinate['Y']
        Z = GAC_Coordinate['Z']+GSM_Coordinate['Z']
        full_geocenter = {"X":X,"Y":Y,"Z":Z}
        print("-------------Finished Full-Geocenter computation-------------\n"
              "=============================================================")
        return full_geocenter

def demo1():
    from datetime import date
    import SaGEA.auxiliary.preference.EnumClasses as Enums
    from SaGEA.auxiliary.aux_tool.FileTool import FileTool
    from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
    lmax = 60
    grid_space = 1
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    gsm_dir, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    gad_dir, gad_key = FileTool.get_project_dir("data/L2_SH_products/GAD/GFZ/RL06/BC01/"), "GRCOF2"
    gac_dir, gac_key = FileTool.get_project_dir("data/L2_SH_products/GAC/GFZ/RL06/BC01/"), "GRCOF2"

    shc, dates_begin, dates_end = load_SHC(gsm_dir, key=gsm_key, lmax=lmax, begin_date=begin_date, end_date=end_date,
                                           get_dates=True, )
    shc_gad = load_SHC(gad_dir, key=gad_key, lmax=lmax, begin_date=begin_date, end_date=end_date)  # load GAD
    shc_gac = load_SHC(gac_dir, key=gac_key, lmax=lmax, begin_date=begin_date, end_date=end_date)
    shc.de_background()
    shc_gad.de_background()

    shc.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,to_type=Enums.PhysicalDimensions.Density)

    OceanSH = np.zeros_like(shc.value)
    A = GRACE_OBP_V2(GRACE=shc.value,OceanSH=OceanSH,GAD=shc_gad.value,lmax=60)
    A.G_Matrix_Term()
    A.I_Matrix_Term()
    # A.CorrectGAD()
    # GCM = A.GCM(mask=None,GRD=False,rotation=True)




if __name__ == '__main__':
    demo1()


