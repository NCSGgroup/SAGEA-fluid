import xarray as xr
import numpy as np

class StorageNC:
    def __init__(self,time,lat,lon,grid_data):
        self.time = time
        self.lat = lat
        self.lon = lon
        self.data = grid_data
        pass

    def save_down(self,output="grid_data.nc"):
        data_array = xr.DataArray(
            self.data,
            dims = ["time","lat","lon"],
            coords={"time":self.time,
                    "lat":self.lat,
                    "lon":self.lon},
            name="rsl"
        )

        ds = xr.Dataset(
            {"rsl":data_array},
            attrs = {
                "description":"relative sea level from SAGEA-fluid",
                "time_period":"Same with GRACE",
                "unit":"m",
            }
        )
        ##ds['time'].attrs['units']="day since 2002-01"
        ##ds['lat'].attrs['units']="degree_north"

        ds.to_netcdf(output)
        print(f"文件已成功保存至：{output}")
        print(f"文件维度信息：\n{ds}")


def demo():
    from lib.SaGEA.auxiliary.aux_tool.MathTool import MathTool
    lat,lon = MathTool.get_global_lat_lon_range(resolution=1)

    x_epoch = np.linspace(2002,2003,12)
    data_grid = np.random.rand(len(x_epoch),len(lat),len(lon))

    StorageNC(time=x_epoch,lat=lat,lon=lon,grid_data=data_grid).save_down(output="J:/Research/demo_test.nc")
    print(len(x_epoch))

if __name__ == "__main__":
    demo()