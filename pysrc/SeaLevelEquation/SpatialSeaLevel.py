from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from SaGEA.auxiliary.preference.Constants import EarthConstant
from pysrc.Auxiliary.LLN import LoveNumber,LLN_Data,LLN_variable,Frame
from SaGEA.auxiliary.preference.EnumClasses import Displacement,GreenFunction
from pysrc.BasisFunction import PointLoad,DiskLoad
import numpy as np
from pysrc.LoadFile.DataClass import GRID
import time
from SaGEA.post_processing.harmonic.Harmonic import Harmonic

class SpatialSLE:
    def __init__(self,grid,lat,lon):
        self.lat,self.lon = lat,lon
        self.res = np.abs(lat[1] - lat[0])
        self.Input = GRID(grid=grid,lat=lat,lon=lon)
        self.lmax = int(180/self.res)
        self.lln = LoveNumber().config(lmax=self.lmax,method=LLN_Data.PREM).get_Love_number()
        self.Green = GreenFunction.PointLoad
        self.Frame = Frame.CE
        print(f"The initial configuration information:\n"
              f"lmax:{self.lmax}, resolution:{self.res}, lat:{self.lat.shape}, lon:{self.lon.shape}, LLN:{LLN_Data.PREM.name}, GreenFunction:{self.Green.name}, Frame:{self.Frame.name}")
    def setLoveNumber(self,lmax,method:LLN_Data.PREM,frame:Frame.CM):
        self.lln = LoveNumber().config(lmax=lmax,method=method).get_Love_number().convert(target=frame)
        print(f"\nThe Load Love Number here is up to degree {lmax}, method is {method.name}, and frame is {frame.name}")
        return self
    def setmaxDegree(self,lmax):
        self.lmax = lmax
        print(f"\nThe update configuration information:\n"
              f"lmax:{self.lmax}, resolution:{self.res}, lat:{self.lat.shape}, lon:{self.lon.shape}\n")
        return self
    def setGreenFunctionType(self,kind:GreenFunction.PointLoad):
        self.Green = kind
        print(f"\nThe GreenFunction here is {self.Green.name}.")
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
            print(f"The iteration is: {iteration + 1},\n"
                  f"The delta is: {np.max(delta)}")
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

        print(f"===Baryterm is: {np.amin(Baryterm,axis=(1,2))}\n")
        end_time = time.time()
        print(f"----------------------------------------------\n"
              f"-----time-consuming: {end_time - start_time:.4f} s-------\n"
              f"==============================================\n")

        return SLE



def Save_SLE(savefile,input,rsl,ghc,vlm,lat,lon,time):
    import xarray as xr
    ds = xr.Dataset(
        {
            "input":(("time","lat","lon"),input),
            "rsl":(("time","lat","lon"),rsl),
            "ghc":(("time","lat","lon"),ghc),
            "vlm":(("time","lat","lon"),vlm),
        },
        coords={"time":time,"lat":lat,"lon":lon},
    )
    ds.to_netcdf(savefile)
    print(f"Save-path is:{savefile}\n"
          f"---Successfully save nc file---")
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

def demo1(index=1):
    import xarray as xr
    res = 0.5
    oceanmask = xr.open_dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"].values[int(res)::int(res*2),int(res)::int(res*2)]
    reference = xr.open_dataset('D:\Cheung\PyZWH\data/ref_sealevel/SLFgrids_GFZOP_CM_WOUTrotation.nc')
    lat = reference['lat'].values[int(res)::int(res*2)]
    lon = reference['lon'].values[int(res)::int(res*2)]
    time = reference['time'].values[0:index]
    input = reference['weh'].values[0:index,int(res)::int(res*2),int(res)::int(res*2)]
    RefRSL = reference['rsl'].values[0:index,int(res)::int(res*2),int(res)::int(res*2)]

    A = SpatialSLE(grid=input,lat=lat,lon=lon)
    A.setGreenFunctionType(kind=GreenFunction.DiskLoad)
    A.setmaxDegree(lmax=60)
    A.setLoveNumber(lmax=60,method=LLN_Data.PREM)
    result = A.SLE(mask=oceanmask,rotation=False)

    Save_SLE(savefile="I:/SLE/stage4/Disk_RSL_WOUTrotation.nc",
             input=result['Input'], rsl=result["RSL"], ghc=result['GHC'], vlm=result['VLM'],
             lat=lat, lon=lon, time=time)
    quick_fig(grid=100*result['RSL'][0],lat=lat,lon=lon)

def demo2(index=10):
    import xarray as xr
    res = 0.5
    oceanmask = xr.open_dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"].values[int(res)::int(res*2),int(res)::int(res*2)]
    reference = xr.open_dataset('D:\Cheung\PyZWH\data/ref_sealevel/SLFgrids_GFZOP_CM_WITHrotation.nc')
    lat = reference['lat'].values[int(res)::int(res*2)]
    lon = reference['lon'].values[int(res)::int(res*2)]
    time = reference['time'].values[0:index]
    input = reference['weh'].values[0:index,int(res)::int(res*2),int(res)::int(res*2)]

    A = SpatialSLE(grid=input,lat=lat,lon=lon)
    A.setGreenFunctionType(kind=GreenFunction.PointLoad)
    A.setmaxDegree(lmax=60)
    A.setLoveNumber(lmax=60,method=LLN_Data.PREM,frame=Frame.CM)
    result = A.SLE(mask=oceanmask,rotation=True)
    Save_SLE(savefile=f"../../result/SLFgrid_spatialmethod/Point_RSL_WITTrotation_{index}.nc",
             input=result['Input'], rsl=result["RSL"], ghc=result['GHC'], vlm=result['VLM'],
             lat=lat, lon=lon, time=time)

    A.setGreenFunctionType(kind=GreenFunction.DiskLoad)
    A.setmaxDegree(lmax=60)
    A.setLoveNumber(lmax=60, method=LLN_Data.PREM,frame=Frame.CM)
    result = A.SLE(mask=oceanmask, rotation=True)

    Save_SLE(savefile=f"../../result/SLFgrid_spatialmethod/Disk_RSL_WITHrotation_{index}.nc",
             input=result['Input'], rsl=result["RSL"], ghc=result['GHC'], vlm=result['VLM'],
             lat=lat, lon=lon, time=time)

def demo_all(index=1):
    import xarray as xr
    import pygmt
    res = 2
    ocean_mask = xr.open_dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"].values[int(res)::int(res*2),int(res)::int(res*2)]
    grid = xr.open_dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WOUTrotation.nc")
    lat = grid["lat"].values[int(res)::int(res*2)]
    lon = grid["lon"].values[int(res)::int(res*2)]
    rsl_grid = grid["rsl"].values[0,int(res)::int(res*2),int(res)::int(res*2)]
    ghc_grid = grid["ghc"].values[0,int(res)::int(res*2),int(res)::int(res*2)]
    vlm_grid = grid["vlm"].values[0,int(res)::int(res*2),int(res)::int(res*2)]
    input = grid["weh"].values[0,int(res)::int(res*2),int(res)::int(res*2)]

    A = SpatialSLE(grid=input, lat=lat, lon=lon)
    A.setGreenFunctionType(kind=GreenFunction.PointLoad)
    A.setmaxDegree(lmax=60)
    A.setLoveNumber(lmax=60, method=LLN_Data.PREM)
    result = A.SLE(mask=ocean_mask, rotation=False)

    # my_grid = xr.open_dataset("I:/SLE/stage4/Point_RSL_WOUTrotation.nc")
    my_rsl = result["RSL"][0]
    my_ghc = result['GHC'][0]
    my_vlm = result['VLM'][0]
    print(my_ghc.shape, my_rsl.shape, my_vlm.shape)

    Datasets = [100 * rsl_grid, 100 * ghc_grid, 100 * vlm_grid, 100 * my_rsl, 100 * my_ghc, 100 * my_vlm]
    Gridsets = []
    for i in np.arange(len(Datasets)):
        grid = xr.DataArray(data=Datasets[i], dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        print(
            f"Max/Min and Mean/RMS :{np.max(grid.values)},{np.min(grid.values)},{np.mean(grid.values)},{np.sqrt(np.mean(grid.values.flatten() ** 2))}")
        Gridsets.append(grid)
    #
    titles = ["(a) Reference RSL", "(b) Reference GHC", "(c) Reference VLM", "(d) Spatial RSL", "(e) Spatial GHC",
              "(f) Spatial VLM"]
    frames = [["a60f30"], ["a0f30"], ["a0f30"], ["a60f30"], ["xa60f30", "ya0f30"], ["xa60f30", "ya0f30"]]
    units = ["EWH (cm)", "EWH (cm)", "EWH (cm)", "EWH (cm)", "EWH (cm)", "EWH (cm)"]
    cmaps = []
    #
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL="15p", FONT_ANNOT_PRIMARY="15p", MAP_FRAME_TYPE="plain")
    pygmt.makecpt(cmap="haxby", series=[-2, 2, 0.2])
    for i in np.arange(len(Gridsets)):
        fig.grdimage(grid=Gridsets[i], projection="Q10c", cmap=True, frame=frames[i])
        fig.coast(shorelines="1/0.5p,black", resolution="f")
        fig.text(position="BR", text=titles[i], offset="-0.1c/0.2c", font="17p,Helvetica-Bold,black")
        if i in [3, 4, 5]:
            fig.colorbar(position="JBC+o-1c/1c+w8c/0.3c+h",
                         frame=["xa1f0.5", f"y+l{units[i]}"])
        if i == 2:
            fig.shift_origin(yshift="-6c")
            fig.shift_origin(xshift="-22c")
        else:
            fig.shift_origin(xshift="11c")

    fig.savefig("I:/temp/SLE/figure4.png")
    fig.show()

def demo_pure(index=1):
    import xarray as xr
    # res = 0.5
    oceanmask = xr.open_dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"].values
    reference = xr.open_dataset('D:\Cheung\PyZWH\data/ref_sealevel/SLFgrids_GFZOP_CM_WOUTrotation.nc')
    lat = reference['lat'].values
    lon = reference['lon'].values
    time = reference['time'].values[0:index]
    input = reference['weh'].values[0:index]
    # RefRSL = reference['rsl'].values[0:index]

    A = SpatialSLE(grid=input,lat=lat,lon=lon)
    A.setmaxDegree(lmax=60)
    A.setLoveNumber(lmax=60,method=LLN_Data.PREM)
    result = A.SLE(mask=oceanmask,rotation=False)

    Save_SLE(savefile="I:/SLE/stage4/Point_RSL_WOUTrotation_pure.nc",
             input=result['Input'], rsl=result["RSL"], ghc=result['GHC'], vlm=result['VLM'],
             lat=lat, lon=lon, time=time)

def demo_inter():
    import xarray as xr
    data1 = xr.open_dataset('I:/SLE/stage4/Point_RSL_WOUTrotation.nc')
    data2 = xr.open_dataset('I:/SLE/stage4/Point_RSL_WOUTrotation_pure.nc')
    lat=data1['lat'].values
    lon=data1['lon'].values
    rsl1 = data1['rsl'].values[0]
    rsl2 = data2['rsl'].values[0]
    quick_fig(grid=1000*(rsl2-rsl1),lat=lat,lon=lon,unit='EWH (mm)')

if __name__ == '__main__':
    demo2()
    # demo_all()
    # demo1()
    # demo_inter()