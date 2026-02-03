import xarray as xr

data_ecco = xr.open_dataset("I:\ECCO\ECCO2/daily.nc")
# print(data_ecco['phibot'])
# print(data_ecco['time'].values)
print(data_ecco)