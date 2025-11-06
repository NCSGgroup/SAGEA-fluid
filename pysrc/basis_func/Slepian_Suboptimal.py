import numpy as np
from scipy.special import sph_harm
from scipy.linalg import eigh
from lib.SaGEA.auxiliary.aux_tool.FileTool import FileTool
from lib.SaGEA.auxiliary.aux_tool.MathTool import MathTool
from lib.SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from tqdm import tqdm
import pygmt
import xarray as xr

lmax = 60
res = 1
class Slepian:
    def __init__(self):
        self.lmax = 60
        self.res = 1
        self.savefile = None

    def setMaxDegree(self,Nmax):
        self.lmax = Nmax
        return self

    def setResolution(self,res):
        self.res = res
        return self

    def setSaveFile(self,file):
        self.savefile = file
        return self

    def load_mask(self,loadfile="data/basin_mask/SH/Greenland_maskSH.dat"):
        """加载mask文件（假设为numpy二进制格式）"""
        Mask_SH = FileTool.get_project_dir(loadfile)
        shc_Mask = load_SHC(Mask_SH, key='', lmax=lmax)  # load basin mask (in SHC)
        grid_basin = shc_Mask.to_grid(grid_space=res)
        grid_basin.limiter(threshold=0.5)
        mask_function = grid_basin.value[0]
        return mask_function

    def compute_slepian_basis(self):
        mask = self.load_mask()
        latitudes,longitudes = MathTool.get_global_lat_lon_range(self.res)
        theta = np.deg2rad(90 - latitudes)
        phi = np.deg2rad(longitudes)
        theta_grid,phi_grid = np.meshgrid(theta,phi,indexing='ij')
        '''Compute the weight of area'''
        dtheta = np.deg2rad(self.res)
        dphi = np.deg2rad(self.res)
        weights = dtheta*dphi*np.sin(theta_grid)*mask
        '''Expand the data to extra the valid points'''
        valid_points = np.where(mask.ravel() == 1)
        theta_valid = theta_grid.ravel()[valid_points]
        phi_valid = phi_grid.ravel()[valid_points]
        weights_valid = weights.ravel()[valid_points]
        '''Setting the shape of spherical harmonic matrix and compute it'''
        K = (self.lmax + 1) ** 2
        Y = np.zeros((len(theta_valid), K), dtype=np.complex128)
        idx = 0
        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                Y[:, idx] = sph_harm(m, l, phi_valid, theta_valid)
                idx += 1

        '''Build the density matrix'''
        W_sqrt = np.sqrt(weights_valid)
        Y_weighted = Y * W_sqrt[:, np.newaxis]
        D = Y_weighted.T.conj() @ Y_weighted

        '''Compute the eigenvalue β and eigenvector Slepian_lm'''
        eigenvalues, eigenvectors = eigh(D)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        print(f"eigenvalues: {eigenvalues[sorted_idx]},{eigenvalues.shape}")

        # 返回结果
        return {
            'eigenvalues': eigenvalues[sorted_idx],
            'eigenvectors': eigenvectors[:, sorted_idx],
            'lmax': self.lmax,
            'mask': mask,
            'grid_shape': mask.shape
        }

    def reconstrcut_basis(self,threshold=0.01):
        '''From eigenvectors to reconstruct to spatial slepian basis'''
        slepian_result = self.compute_slepian_basis()
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
        latitudes,longitudes = MathTool.get_global_lat_lon_range(self.res)
        theta,phi = MathTool.get_colat_lon_rad(lat=latitudes,lon=longitudes)
        theta_grid,phi_grid = np.meshgrid(theta,phi,indexing='ij')

        basis_functions = []
        for i in tqdm(np.arange(Ncut)):
            coeffs = eigenvectors[:, i]
            basis = np.zeros_like(theta_grid, dtype=np.complex128)

            idx = 0
            for l in np.arange(self.lmax + 1):
                for m in np.arange(-l, l + 1):
                    Y_lm = sph_harm(m, l, phi_grid.ravel(), theta_grid.ravel())
                    basis += coeffs[idx] * Y_lm.reshape(basis.shape)
                    idx += 1
            print(basis)
            basis = basis.real
            # basis = basis.real * mask
            basis_functions.append(basis)

        return np.array(basis_functions)

    def fig(self,file,maxvalue=10,idx=0):
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

        fig.grdimage(grid=grid_set[idx],projection='Q10c',region=[280,350,55,90],cmap=True,frame={"af",f"+t{idx+1}th Slepian"})
        fig.coast(shorelines='1/0.5p,black',resolution='f')
        fig.colorbar(position=["JBL+o-8c/1c+w8c/0.3c+h"],
                     frame=[f"xa{maxvalue/2}"])
        if self.savefile:
            fig.savefig(self.savefile)
            fig.show()
        else:
            fig.show()


def demo1():
    a = Slepian()
    # mask = a.load_mask()
    # basis_functions = a.reconstrcut_basis(threshold=0.01)
    # print(f"The shape of basis_function: {basis_functions.shape}")
    # np.save("I:/slepian_basis_nomask.npy", basis_functions)

    idx=33
    a.setSaveFile(file=f"I:/temp/Figure/Slepian_{idx}.png")
    grid = np.load("I:/slepian_basis_nomask.npy")
    grid = grid
    a.fig(file=grid,idx=idx)

def demo2():
    a = Slepian()
    idx = 33



if __name__ == "__main__":
    demo1()