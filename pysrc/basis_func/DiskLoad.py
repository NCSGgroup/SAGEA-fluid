import numpy as np

from pysrc.ancillary.geotools.LLN import LoveNumber, LLN_Data, LLN_variable
from pysrc.basis_func.Legendre import Legendre_polynomial
from SaGEA.auxiliary.preference.Constants import EarthConstant
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from SaGEA.auxiliary.preference.EnumClasses import Displacement
from tqdm import tqdm

class DiskLoad:

    def __init__(self, lln: LoveNumber, radius, thickness=1):
        """

        :param lln:
        :param radius: radius of the disk, unit: degree
        """
        self._residence = None
        self._lln = lln

        '''compute sigma'''
        leg = Legendre_polynomial(alpha=radius, lmax=self._lln.lmax + 1)
        leg_back_2 = np.roll(leg, shift=2, axis=-1)

        sigma_n = leg_back_2 - leg
        sigma_n = np.delete(sigma_n, obj=1, axis=-1)
        sigma_n[0] = 1 - np.cos(np.deg2rad(radius))
        sigma_n *= 0.5 * thickness * EarthConstant.rhow
        self._sigma = sigma_n
        pass

    def configure(self, residence):
        """

        :param residence: deg, angular distance between two nodes
        :return:
        """
        self._residence = residence
        self._legendre()

        return self

    def _legendre(self):
        self._Pl = Legendre_polynomial(alpha=self._residence, lmax=self._lln.lmax)
        pass

    def getVertical(self):
        n = np.arange(self._lln.lmax + 1)
        item = self._Pl * self._lln.LLN[LLN_variable.h] * self._sigma / (1 + 2 * n)

        return np.sum(item, axis=-1) * 3 / EarthConstant.rhoear

    def getHorizental(self):
        """
        It could be generating NaN if theta (residence) is zero. Not the case for vertical displacement and geoid height
        :return:
        """
        n = np.arange(self._lln.lmax + 1)

        sinTheta = np.sin(np.deg2rad(self._residence))
        cosTheta = np.cos(np.deg2rad(self._residence))

        Plb = np.roll(self._Pl, 1, axis=-1)

        if len(np.shape(sinTheta)) != 0:
            sinTheta = sinTheta[:, None]
            cosTheta = cosTheta[:, None]

        differential = 1 / sinTheta * (cosTheta * self._Pl - Plb) * n

        item = differential * self._lln.LLN[LLN_variable.l] * self._sigma / (1 + 2 * n)

        return np.sum(item, axis=-1) * 3 / EarthConstant.rhoear

    def getGeoidheight(self):
        n = np.arange(self._lln.lmax + 1)
        item = self._Pl * (1 + self._lln.LLN[LLN_variable.k]) * self._sigma / (1 + 2 * n)

        return np.sum(item, axis=-1) * 3 / EarthConstant.rhoear

    def release(self):
        """
        This is for reducing memory! Very important for global convolution
        :return:
        """
        self._residence = None
        self._Pl = None
        pass

class DiskLoad_constrain(DiskLoad):
    """
    When the angular distance is above K*radius, the impact of this load could be considered as negligible.
    """

    def __init__(self, lln: LoveNumber, radius, thickness=1, constrain_factor=10):
        super().__init__(lln, radius, thickness)
        self._residence_full = None
        self._threshold = constrain_factor * radius
        pass

    def configure(self, residence):
        """

        :param residence: deg, angular distance between two nodes
        :return:
        """
        self._residence_full = residence
        self._residence = residence[residence < self._threshold]
        self._legendre()

        return self

    def getVertical(self):
        a = super().getVertical()
        b = np.zeros_like(self._residence_full)
        b[self._residence_full < self._threshold] = a

        return b

    def getHorizental(self):
        a = super().getHorizental()
        b = np.zeros_like(self._residence_full)
        b[self._residence_full < self._threshold] = a

        return b

    def getGeoidheight(self):
        a=super().getGeoidheight()
        b = np.zeros_like(self._residence_full)
        b[self._residence_full < self._threshold] = a

        return b

    def release(self):
        """
        This is for reducing memory! Very important for global convolution
        :return:
        """
        super().release()
        self._residence_full = None
        pass


class GFA_displacement:
    """
    Green Function Approach (GFA): based on disk load green function;
    """

    def __init__(self, lln: LoveNumber):
        self._grids = None
        self._DL = None
        self._lln = lln

    def configure(self, grids: dict, cf=15):
        """
        define the unit uniform disk loads (thickness = 1 meter), which is dependent on the radius.
        Avoid repeating definition of the disk by telling the radius.
        :param grids: ('lat', 'lon', 'radius', 'EWH') of the grid; radius [degree], lat, lon [degree], EWH [meter].
        EWH could be actually a matrix containing the dimension of time to speed up computation.
        :return:
        """

        Unit_Disk_loads = {}
        for rr in grids['radius']:
            if rr not in Unit_Disk_loads.keys():
                Unit_Disk_loads[rr] = DiskLoad_constrain(lln=self._lln, radius=rr, thickness=1, constrain_factor=cf)

        self._DL = Unit_Disk_loads
        self._grids = grids
        return self

    def evaluation(self, points: dict, variable=Displacement.Vertical):
        """
        Evaluation of desired variable at specified points
        :param points: ('lat', 'lon')
        :param variable:
        :return:
        """
        grids = self._grids
        dl = self._DL
        dis = np.zeros_like(grids['EWH'])
        print(f"the cycle-index is: {len(list(grids['radius']))}")
        for id, rr in tqdm(enumerate(list(grids['radius']))):
            # print(id)
            theta = MathTool.angular_distance(grids['lat'][id], grids['lon'][id], points['lat'],
                                              points['lon'])

            '''To avoid singularity'''
            theta[theta == 0] += 1e-6
            theta[theta == 180] += 1e-6

            '''calculate the displacement'''
            dis += self._getFunc(DL=dl[rr].configure(residence=theta), variable=variable)[:, None] @ grids['EWH'][id][
                                                                                                     None, :]

            '''release memory'''
            dl[rr].release()

        return dis

    def _getFunc(self, variable: Displacement, DL: DiskLoad_constrain):
        if variable == Displacement.Vertical:
            return DL.getVertical()
        elif variable == Displacement.Horizontal:
            return DL.getHorizental()
        elif variable == Displacement.Geoheight:
            return DL.getGeoidheight()


class GFA_regular_grid(GFA_displacement):

    def __init__(self, lln: LoveNumber):
        super().__init__(lln)

    def evaluation(self, points: dict, variable=Displacement.Vertical, resolution=2):
        """
        Evaluation of desired variable at specified points
        :param resolution:
        :param points: ('lat', 'lon')
        :param variable:
        :return:
        """
        grids = self._grids
        dl = self._DL
        dis = np.zeros_like(grids['EWH'])

        Num = int(180 / resolution)
        lat0 = -1000
        # print(f"the cycle-index is: {len(list(grids['radius']))}")
        for id, rr in tqdm(enumerate(list(grids['radius']))):
            # print(id)
            if grids['lat'][id] != lat0:
                lat0 = grids['lat'][id]
                theta = MathTool.angular_distance(grids['lat'][id], grids['lon'][id], points['lat'],
                                                  points['lon'])
                '''To avoid singularity'''
                theta[theta == 0] += 1e-6
                theta[theta == 180] += 1e-6

                '''calculate the displacement'''
                temp = self._getFunc(DL=dl[rr].configure(residence=theta), variable=variable)

            else:
                temp = np.roll(temp.reshape((Num, -1)), shift=1, axis=1)
                temp = temp.flatten()

            dis += temp[:, None] @ grids['EWH'][id][None, :]

            '''release memory'''
            dl[rr].release()

        return dis


def grid2radius(lat_center, grid_size):
    """
    calculate the radius of disk load that equals to the area of pixel
    :param lat_center: center of the pixel, latitude, [degree]
    :param grid_size: equal-distance grid, grid interval, e.g., 1 degree
    :return: theta_radius [degree], length_radius [m]
    """
    from SaGEA.auxiliary.preference.Constants import EarthConstant
    example = 30  # longitude, but indeed the choice could be arbitrary.

    a = MathTool.angular_distance(point1_lat=lat_center + grid_size / 2, point1_lon=example,
                                  point2_lat=lat_center - grid_size / 2, point2_lon=example)

    b = MathTool.angular_distance(point1_lat=lat_center, point1_lon=example, point2_lat=lat_center,
                                  point2_lon=example + grid_size)

    # print(np.deg2rad(a) * np.deg2rad(b))
    r = np.sqrt(np.deg2rad(a) * np.deg2rad(b) / np.pi)

    return np.rad2deg(r), EarthConstant.radiusm * r


def grid2radius_type2(lat_center, grid_size):
    """
    calculate the radius of disk load that equals to the area of pixel
    :param lat_center: center of the pixel, latitude, [degree]
    :param grid_size: equal-distance grid, grid interval, e.g., 1 degree
    :return: theta_radius [degree], length_radius [m]
    """
    from SaGEA.auxiliary.preference.Constants import EarthConstant

    area = np.cos(np.deg2rad(lat_center)) * np.deg2rad(grid_size) ** 2

    # print(area)
    x = 1 - area / (2 * np.pi)

    r = np.arccos(x)

    return np.rad2deg(r), EarthConstant.radiusm * r


def Mass(r, thickness):
    """
    calculate the radius of disk load that equals to the area of pixel
    :param lat_center: center of the pixel, latitude, [degree]
    :param grid_size: equal-distance grid, grid interval, e.g., 1 degree
    :return: theta_radius [degree], length_radius [m]
    """
    from SaGEA.auxiliary.preference.Constants import EarthConstant

    M1 = 2 * np.pi * EarthConstant.radiusm ** 2 * (
                1 - np.cos(r / EarthConstant.radiusm)) * EarthConstant.rhow * thickness

    M2 = np.pi * r ** 2 * EarthConstant.rhow * thickness / (1e12)

    return M1, M2
def demo():
    lln = LoveNumber().config(lmax=40000, method=LLN_Data.REF).get_Love_number()

    # file = FileTool.get_project_dir() / 'data' / 'LLN' / 'PREM-LGFs.dat'
    # theta = np.loadtxt(file, skiprows=1, usecols=0)
    # theta = 0.001100220044009
    theta = np.array([0.001000200040008, 0.001100220044009])
    load = DiskLoad(lln=lln, radius=0.10).configure(residence=theta)
    u = load.getVertical()
    v = load.getHorizental()
    g = load.getGeoidheight()
    print(u)
    print(v)
    print(g)

    load = DiskLoad_constrain(lln=lln, radius=0.10, constrain_factor=0.01).configure(residence=theta)
    u = load.getVertical()
    v = load.getHorizental()
    g = load.getGeoidheight()
    print(u)
    print(v)
    print(g)
    pass
def demo2_Green():
    """Use CSR-Mascon to do the tests"""
    import netCDF4 as nc
    from pathlib import Path
    lln = LoveNumber().config(lmax=10000, method=LLN_Data.PREM).get_Love_number()
    gfa = GFA_displacement(lln=lln)

    '''===============handling mascon data========================'''
    filename = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'

    dir_in = 'I:\CSR\CSR_Mascons/'
    csr = nc.Dataset(Path(dir_in) / filename)

    """upscale to 1 degree"""
    res = 2
    index = 180
    mascon_data = np.array(csr['lwe_thickness'])[index, res * 2::res * 4, res * 2::res * 4]

    mascon_data = np.roll(mascon_data, shift=int(-180 / res), axis=-1)
    print(f"mascon_data is :{mascon_data.shape}")

    mascon_data = mascon_data.reshape((1, -1)).T
    print(f"mascon_data is :{mascon_data.shape}")

    lat, lon = MathTool.get_global_lat_lon_range(resolution=res)
    lon, lat = np.meshgrid(lon, lat)

    rr = grid2radius(lat_center=lat, grid_size=res)

    grids = {
        'lat': lat.flatten(),
        'lon': lon.flatten(),
        'radius': rr[0].flatten(),
        'EWH': mascon_data * 0.01  # cm to meter
    }

    gfa.configure(grids=grids, cf=1000)

    point = {
        'lat': lat.flatten(),
        'lon': lon.flatten(),
    }

    dist = gfa.evaluation(points=point, variable=Displacement.Vertical)
    print(f"dist is: {dist.shape}")

def demo2_Green_fast():
    '''===============handling mascon data========================'''
    import netCDF4 as nc
    from pathlib import Path
    lln = LoveNumber().config(lmax=10000, method=LLN_Data.PREM).get_Love_number()
    gfa = GFA_regular_grid(lln=lln)

    '''===============handling mascon data========================'''
    filename = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'

    dir_in = 'I:\CSR\CSR_Mascons/'
    csr = nc.Dataset(Path(dir_in) / filename)

    """upscale to 1 degree"""
    res = 2
    index = 180
    mascon_data = np.array(csr['lwe_thickness'])[index, res * 2::res * 4, res * 2::res * 4]
    # mascon_data = np.array(csr['lwe_thickness'])[index, res * 2::res * 4, res * 2::res * 4]# unit: cm

    mascon_data = np.roll(mascon_data, shift=int(-180 / res), axis=-1)

    # mascon_data = mascon_data.flatten()
    mascon_data = mascon_data.reshape((1, -1)).T
    # mascon_data = mascon_data.reshape((index+1, -1)).T

    lat, lon = MathTool.get_global_lat_lon_range(resolution=res)
    lon, lat = np.meshgrid(lon, lat)

    # sn = np.array([lat.flatten(),lon.flatten(),mascon_data.flatten()])
    #
    # np.savetxt('../temp/mascon.txt', sn.T)

    rr = grid2radius(lat_center=lat, grid_size=res)

    grids = {
        'lat': lat.flatten(),
        'lon': lon.flatten(),
        'radius': rr[0].flatten(),
        'EWH': mascon_data * 0.01  # cm to meter
    }

    gfa.configure(grids=grids, cf=1000)

    point = {
        'lat': lat.flatten(),
        'lon': lon.flatten(),
    }

    dist = gfa.evaluation(points=point, variable=Displacement.Vertical)
    print(dist.shape)

    dist2 = gfa.evaluation(points=point, variable=Displacement.Geoheight)
    print(dist2.shape)

if __name__ == '__main__':
    # demo()
    # demo2_Green()
    demo2_Green_fast()
