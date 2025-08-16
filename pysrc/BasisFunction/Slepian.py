import numpy as np
# from scipy.linalg import eigh
from pysrc.Auxiliary.FileTool import FileTool
from pysrc.Auxiliary.MathTool import MathTool
from pysrc.LoadFile.LoadL2SH import load_SHC
from tqdm import tqdm
from pysrc.BasisFunction.Harmonic import Harmonic

class Slepian:
    def __init__(self):
        self.lmax = 60
        self.res = 1
        self.savefile = None

    def setMaxDegree(self, Nmax):
        self.lmax = Nmax
        return self

    def setResolution(self, res):
        self.res = res
        return self

    def setSaveFile(self, file):
        self.savefile = file
        return self

    def load_mask(self, loadfile="data/basin_mask/SH/Greenland_maskSH.dat"):
        """加载mask文件（假设为numpy二进制格式）"""
        Mask_SH = FileTool.get_project_dir(loadfile)
        shc_Mask = load_SHC(Mask_SH, key='', lmax=self.lmax)  # load basin mask (in SHC)
        grid_basin = shc_Mask.to_grid(grid_space=self.res)
        grid_basin.limiter(threshold=0.5)
        mask_function = grid_basin.value[0]
        return mask_function
    def Slepian_Feature_Matrix(self):
        mask = self.load_mask()
        lat,lon = MathTool.get_global_lat_lon_range(resolution=self.res)
        H = Harmonic(lat=lat, lon=lon, lmax=self.lmax,option=1).get_spherical_harmonic_function()
        Upsilon_co = H["Upsilon_co"]
        Upsilon_so = H["Upsilon_so"]
        print(f"Upsilon is:{Upsilon_co.shape} and {Upsilon_so.shape}")
        theta,phi = MathTool.get_colat_lon_rad(lat=lat,lon=lon)
        theta_grid,phi_grid = np.meshgrid(theta,phi,indexing="ij")
        dtheta,dphi = np.deg2rad(self.res),np.deg2rad(self.res)
        area_weights = dtheta*dphi*np.sin(theta_grid)*mask
        valid_points = np.where(mask.ravel()==1)

        Upsilon_co = Upsilon_co.reshape((len(mask.ravel()),self.lmax+1,self.lmax+1))
        Upsilon_so = Upsilon_so.reshape((len(mask.ravel()), self.lmax + 1, self.lmax + 1))
        print(f"Upsilon is:{Upsilon_co.shape} and {Upsilon_so.shape}")
        Upsilon_co = Upsilon_co[valid_points]
        Upsilon_so = Upsilon_so[valid_points]
        print(f"Upsilon is:{Upsilon_co.shape} and {Upsilon_so.shape}")

        # theta_valid = theta_grid.ravel()[valid_points]
        # phi_valid = phi_grid.ravel()[valid_points]
        area_weights_valid = area_weights.ravel()[valid_points]

        K = (self.lmax+1)**2
        Upsilon_valid = np.zeros((len(Upsilon_co),K))
        idx= 0
        for n in np.arange(self.lmax+1):
            for m in np.arange(-n,n+1):
                if m<0:
                    Upsilon_valid[:,idx] = Upsilon_so[:,n,-m]
                    idx+=1
                else:
                    Upsilon_valid[:,idx] = Upsilon_co[:,n,m]
                    idx+=1
        W_sqrt = np.sqrt(area_weights_valid)
        Y_weighted = Upsilon_valid * W_sqrt[:, np.newaxis]
        D = (Y_weighted.T @ Y_weighted)/(4*np.pi)
        # '''Compute the eigenvalue β and eigenvector Slepian_lm'''
        # eigenvalues, eigenvectors = eigh(D)
        eigenvalues, eigenvectors = np.linalg.eigh(D)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        print(f"eigenvalues: {eigenvalues[sorted_idx]},{eigenvalues.shape}")

        return {
            'eigenvalues': eigenvalues[sorted_idx],
            'eigenvectors': eigenvectors[:, sorted_idx],
            'lmax': self.lmax,
            'mask': mask,
            'grid_shape': mask.shape,
            'Slepian_coefficient':eigenvectors.T
        }

    def reconstrcut_basis(self, threshold=0.01):
        '''From eigenvectors to reconstruct to spatial slepian basis'''
        slepian_result = self.Slepian_Feature_Matrix()
        eigenvalues = slepian_result['eigenvalues']
        eigenvectors = slepian_result['eigenvectors']
        mask = slepian_result['mask']
        '''Get the Ncut'''
        Ncut = 0
        for i in np.arange(len(eigenvalues)):
            if eigenvalues[i] <= threshold:
                Ncut = i
                break
            else:
                continue
        print(f"When the threshold is {threshold}, the Ncut is: {Ncut}")
        '''Generate global grid'''
        lat, lon = MathTool.get_global_lat_lon_range(self.res)
        theta, phi = MathTool.get_colat_lon_rad(lat=lat, lon=lon)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

        print(f"theta_grid flatten is: {len(theta_grid.ravel())}")


        print(f"Eigenvectors is :{eigenvectors.shape}")
        H = Harmonic(lat=lat, lon=lon, lmax=self.lmax, option=1)
        Slepian_Fuction = []
        Slepian_Coefficient = eigenvectors[:,:].T
        for i in tqdm(np.arange(Ncut)):
            coeffs = Slepian_Coefficient[i,:].reshape(1,-1)
            idx = 0
            grid = H.synthesis_1D(cqlm=coeffs)
            Slepian_Fuction.append(grid[0])
            idx += 1
        Slepian_Fuction = np.array(Slepian_Fuction)

        Slepian = {
            "Slepian_function":Slepian_Fuction,
            "Slepian_coefficient":Slepian_Coefficient,
        }
        return Slepian

    def save_to_nc(self,data,savefile):
        import xarray as xr
        lat,lon = MathTool.get_global_lat_lon_range(self.res)
        idx = np.arange(1,len(data[:,0,0])+1)
        ds = xr.Dataset(
            {
                "Slepian":(("idx","lat","lon"),data),
            },
            coords={"idx":idx,"lat":lat,"lon":lon},)
        ds.to_netcdf(savefile)
        print("====save file in .nc format successfully===========")

    def eigenvalues_fig(self):
        import pygmt
        data = self.Slepian_Feature_Matrix()['eigenvalues']
        xaixs = np.arange(1,len(data)+1)
        print(xaixs.shape)

        fig = pygmt.Figure()
        pygmt.config(FONT_ANNOT_PRIMARY="15p",FONT_LABEL="15p")
        fig.plot(
            region=[1,10e2,-0.05,1.05],
            projection="X10cl/6c",
            frame=["WSne","xa1pf3+lIndex i","ya0.2f0.1+lEigenvalue β"],
            x = xaixs,
            y = data,
            pen = f"2p,blue",
        )
        fig.plot(
            x=xaixs,
            y=data,
            pen=f"2p,blue",
            style="c0.2c",
            fill="blue"
        )
        fig.plot(
            x = [34,34],
            y = [-0.05,1.05],
            pen = f"1p,black,--"
        )
        fig.plot(
            x = xaixs[33],
            y = data[33],
            pen = "2p,red",
            style = "c0.2c",
            fill = "red"
        )
        fig.text(x=170,y=0.4,text="β<0.01,Ncut=34",font="17p,Helvetica,red",pen="1p,red")
        fig.savefig("I:/temp/Figure/Slepian_index_i.png")
        fig.show()

    def quick_fig(self,file,maxvalue=10,idx=0):
        import pygmt
        import xarray as xr
        latitudes, longitudes = MathTool.get_global_lat_lon_range(self.res)
        grid_set = []
        for i in np.arange(len(file)):
            grid = xr.DataArray(data=file[i],dims=['lat','lon'],coords={'lat':latitudes,'lon':longitudes})
            print(f"{i}th extremes are: {grid.data.max()},{grid.data.min()}")
            grid_set.append(grid)
        # grid_set = np.array(grid_set)

        fig = pygmt.Figure()
        pygmt.config(FONT_LABEL='15p',FONT_ANNOT_PRIMARY='15p',FONT_TITLE='17p',MAP_TITLE_OFFSET='-0.3c')
        pygmt.makecpt(cmap='haxby',series=[-maxvalue,maxvalue,maxvalue/10])

        # fig.grdimage(grid=grid_set[idx],projection='Q10c',region=[280,350,55,90],cmap=True,frame={"af",f"+t{idx+1}th Slepian"})
        fig.grdimage(grid=grid_set[idx], projection='Q10c', cmap=True,
                     frame={"af", f"+t{idx + 1}th Slepian"})
        fig.coast(shorelines='1/0.5p,black',resolution='f')
        fig.colorbar(position=["JBL+o-8c/1c+w8c/0.3c+h"],
                     frame=[f"xa{maxvalue/2}"])
        if self.savefile:
            fig.savefig(self.savefile)
        fig.show()
def demo1():
    a = Slepian()
    data = a.reconstrcut_basis()

    slepian = data["Slepian_function"]
    a.save_to_nc(data=slepian,savefile='../../data/temp/slepian/slepian_function1.nc')

def demo2(idx=1):
    import xarray as xr
    slepian_function = xr.open_dataset("../../data/temp/slepian/slepian_function.nc")
    slepian_function1 = xr.open_dataset("../../data/temp/slepian/slepian_function1.nc")
    lat = slepian_function["lat"].values
    lon = slepian_function["lon"].values
    data = slepian_function["Slepian"].values
    data1 = slepian_function1["Slepian"].values
    A =Slepian()
    A.quick_fig(file=data,idx=idx,maxvalue=10)
    print(f"Slepian1=========\n")
    # A.quick_fig(file=data1, idx=idx, maxvalue=10)

def demo3():
    A = Slepian()
    A.eigenvalues_fig()



if __name__ == "__main__":
    demo2()