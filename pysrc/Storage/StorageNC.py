from pysrc.SeaLevelEquation.SeaLevelEquation_Old import SpatialSLE
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
import numpy as np
import netCDF4 as nc
import xarray as xr


def Save_SLE(savefile,input,rsl,ghc,vlm,mask,lat,lon,time):
    ds = xr.Dataset(
        {
            "input":(("time","lat","lon"),input),
            "rsl":(("time","lat","lon"),rsl),
            "ghc":(("time","lat","lon"),ghc),
            "vlm":(("time","lat","lon"),vlm),
            "mask":(("time","lat","lon"),mask)
        },
        coords={"time":time,"lat":lat,"lon":lon},
    )
    ds.to_netcdf(savefile)
    print(f"Save-path is:{savefile}\n"
          f"---Successfully save nc file---")
class Storage:
    def __init__(self):
        self.savefile = "I:/SLE/GCI_RSL.nc"
        pass
    def setfile(self,savefile):
        self.savefile = savefile
        return self
    def Ocean_Mask(self,lmax,res,loadfile="data/basin_mask/SH/Ocean_maskSH.dat"):
        OceanFuction_SH = FileTool.get_project_dir(loadfile)
        shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=lmax)  # load basin mask (in SHC)
        grid_basin = shc_OceanFunction.to_grid(grid_space=res)
        grid_basin.limiter(threshold=0.5)
        ocean_function = grid_basin.value[0]
        return ocean_function
    def Save_nc_3D(self,data,lat,lon,time):
        ds = xr.Dataset(
            {
                "rsl":(("time","lat","lon"),data),
            },
            coords={
                "time":time,
                "lat":lat,
                "lon":lon,
            },
        )
        ds.to_netcdf(self.savefile)
        print(f"Save-path is:{self.savefile}\n"
              f"---Successfully save nc file---")
    def Save_SLE(self,input,rsl,ghc,vlm,lat,lon,time):
        ds = xr.Dataset(
            {
                "input":(("time","lat","lon"),input),
                "rsl":(("time","lat","lon"),rsl),
                "ghc":(("time","lat","lon"),ghc),
                "vlm":(("time","lat","lon"),vlm),
            },
            coords={"time":time,"lat":lat,"lon":lon},
        )
        ds.to_netcdf(self.savefile)
        print(f"Save-path is:{self.savefile}\n"
              f"---Successfully save nc file---")


def demo1():
    from SaGEA.auxiliary.preference.EnumClasses import GreenFunction
    res = 0.5
    DataSet = xr.open_dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WITHrotation.nc")
    lat = np.array(DataSet["lat"].values[int(res)::int(res * 2)])
    lon = np.array(DataSet["lon"].values[int(res)::int(res * 2)])
    time = DataSet["time"].values[0]
    time = np.array([time])
    Land_Load = DataSet["weh"].values[0, int(res)::int(res * 2), int(res)::int(res * 2)]

    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"][:]

    A = SpatialSLE(grid=Land_Load,lat=lat,lon=lon).setGreenFunctionType(kind=GreenFunction.PointLoad)
    data = A.SLE(rotation=True,mask=ocean_mask)['RSL']

    # data = np.load('../../data/ref_sealevel/RSL_180.npy')
    print(f"data and lat and lon are:{Land_Load.shape,lat.shape,lon.shape,time.shape,time}")

    B = Storage().setfile(savefile="../../data/temp/Point_RSL_WITHrotation.nc")
    B.Save_nc_3D(data=data.value,lat=lat,lon=lon,time=time)

def demo2():
    from SaGEA.auxiliary.preference.EnumClasses import GreenFunction
    res = 0.5
    DataSet = xr.open_dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WITHrotation.nc")
    lat = np.array(DataSet["lat"].values[int(res)::int(res * 2)])
    lon = np.array(DataSet["lon"].values[int(res)::int(res * 2)])
    time = DataSet["time"].values[0]
    time = np.array([time])
    Land_Load = DataSet["weh"].values[0, int(res)::int(res * 2), int(res)::int(res * 2)]

    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"][:]

    A = SpatialSLE(grid=Land_Load, lat=lat, lon=lon).setGreenFunctionType(kind=GreenFunction.DiskLoad)
    data = A.SLE(rotation=True,mask=ocean_mask)['RSL']

    # data = np.load('../../data/ref_sealevel/RSL_180.npy')
    print(f"data and lat and lon are:{Land_Load.shape, lat.shape, lon.shape, time.shape, time}")

    B = Storage().setfile(savefile="../../data/temp/Disk_RSL_WITHrotation.nc")
    B.Save_nc_3D(data=data.value, lat=lat, lon=lon, time=time)


def demo3():
    from SaGEA.auxiliary.preference.EnumClasses import GreenFunction
    # res = 0.5
    DataSet = xr.open_dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WITHrotation.nc")
    lat = np.array(DataSet["lat"].values[:])
    lon = np.array(DataSet["lon"].values[:])
    time = DataSet["time"].values[:]
    time = np.array([time])
    Land_Load = DataSet["weh"].values[:,:,:]

    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"][:]

    A = SpatialSLE(grid=Land_Load,lat=lat,lon=lon).setGreenFunctionType(kind=GreenFunction.PointLoad)
    data = A.SLE(rotation=False,mask=ocean_mask)

    # data = np.load('../../data/ref_sealevel/RSL_180.npy')
    print(f"data and lat and lon are:{Land_Load.shape,lat.shape,lon.shape,time.shape,time}")

    B = Storage().setfile(savefile="../../data/temp/SLE/Point_RSL_WOUTrotation_all.nc")
    B.Save_SLE(input=data['Input'],rsl=data["RSL"].value,ghc=data['GHC'],vlm=data['VLM'],lat=lat,lon=lon,time=time)

def demo4():
    from SaGEA.auxiliary.preference.EnumClasses import GreenFunction
    # res = 0.5
    DataSet = xr.open_dataset("../../data/ref_sealevel/SLFgrids_GFZOP_CM_WITHrotation.nc")
    lat = np.array(DataSet["lat"].values[:])
    lon = np.array(DataSet["lon"].values[:])
    time = DataSet["time"].values[:]
    time = np.array([time])
    Land_Load = DataSet["weh"].values[:,:,:]

    ocean_mask = nc.Dataset("../../data/ref_sealevel/ocean_mask.nc")["ocean_mask"][:]

    A = SpatialSLE(grid=Land_Load,lat=lat,lon=lon).setGreenFunctionType(kind=GreenFunction.DiskLoad)
    data = A.SLE(rotation=False,mask=ocean_mask)

    # data = np.load('../../data/ref_sealevel/RSL_180.npy')
    print(f"data and lat and lon are:{Land_Load.shape,lat.shape,lon.shape,time.shape,time}")

    B = Storage().setfile(savefile="../../data/temp/SLE/Disk_RSL_WOUTrotation_all.nc")
    B.Save_SLE(input=data['Input'],rsl=data["RSL"].value,ghc=data['GHC'],vlm=data['VLM'],lat=lat,lon=lon,time=time)

if __name__ == "__main__":
    # demo1()
    # demo2()
    demo3()
    demo4()


