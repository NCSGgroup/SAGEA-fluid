import numpy as np
from pysrc.ancillary.geotools.LLN import LoveNumber, LLN_variable
from pysrc.basis_func.Legendre import Legendre_polynomial
from SaGEA.auxiliary.preference.Constants import EarthConstant
from SaGEA.auxiliary.preference.EnumClasses import Displacement
from SaGEA.auxiliary.aux_tool.MathTool import MathTool
from tqdm import tqdm


class LGF_truncation:
    """
    This is theoretically right but practical wrong because of limited truncation. Therefore, a compensation should be applied, see LGF
    """

    def __init__(self, lln: LoveNumber):
        self._residence = None
        self._lln = lln

    def configure(self, residence):
        '''

        :param residence: deg, angular distance between two nodes
        :return:
        '''

        residence[residence < 1e-5] = 1e-5  # to avoid singularity
        self._residence = residence
        self._legendre()
        return self

    def _legendre(self):
        self._Pl = Legendre_polynomial(alpha=self._residence, lmax=self._lln.lmax)
        pass

    def getVertical(self):
        item = self._Pl * self._lln.LLN[LLN_variable.h]

        return np.sum(item, axis=-1) * EarthConstant.radiusm / EarthConstant.Mass

    def getHorizental(self):
        n = np.arange(self._lln.lmax + 1)

        sinTheta = np.sin(np.deg2rad(self._residence))
        cosTheta = np.cos(np.deg2rad(self._residence))

        Plb = np.roll(self._Pl, 1, axis=-1)
        differential = n / sinTheta * (cosTheta * self._Pl - Plb)

        return differential[:, 1:] @ self._lln.LLN[LLN_variable.l][1:,
                                     None] * EarthConstant.radiusm / EarthConstant.Mass

    def getGeoidheight(self):
        return (self._Pl @ (self._lln.LLN[LLN_variable.k][:, None] + 1) * EarthConstant.radiusm / EarthConstant.Mass).flatten()
class GFA_displacement:
    """
       Green Function Approach (GFA): based on truncated point mass load green function;
       !! The maximal degree of LLN should be consistent with the resolution of load!!!
       For example, for a load at 2 degree, the lmax of LLN should be 180/2 = 90; otherwise, it would introduce large errors.
       """

    def __init__(self, lln: LoveNumber):
        self._grids = None
        # self._PL = LGF(lln=lln)
        self._PL = LGF_truncation(lln=lln)
        self._lln = lln

    def configure(self, grids: dict):
        """
        define the unit uniform disk loads (thickness = 1 meter), which is dependent on the radius.
        Avoid repeating definition of the disk by telling the radius.
        :param grids: ('lat', 'lon', 'area', 'EWH') of the grid; area [m**2], lat, lon [degree], EWH [meter].
        EWH could be actually a matrix containing the dimension of time to speed up computation.
        :return:
        """
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
        pl = self._PL
        dis = np.zeros_like(grids['EWH'])
        for id, rr in tqdm(enumerate(list(grids['area']))):
            # print(id)
            theta = MathTool.angular_distance(grids['lat'][id], grids['lon'][id], points['lat'],
                                                points['lon'])
            '''To avoid singularity'''
            # theta[theta == 0] += 1e-6
            # theta[theta == 180] += 1e-6

            '''calculate the displacement'''
            a= self._getFunc(PL=pl.configure(residence=theta), variable=variable)
            temp = (a * rr)[:, None]
            dis += temp @ grids['EWH'][id][None, :]

        return dis

    def _getFunc(self, variable: Displacement, PL: LGF_truncation):
        if variable == Displacement.Vertical:
            return PL.getVertical()
        elif variable == Displacement.Horizontal:
            return PL.getHorizental()
        elif variable == Displacement.Geoheight:
            return PL.getGeoidheight()

class GFA_regular_grid(GFA_displacement):
    """
    This fast version only works when the grid network is defined as equal angular distance grid.
    """

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
        pl = self._PL
        dis = np.zeros_like(grids['EWH'])
        Num = int(180 / resolution)
        lat0 = -1000
        for id, rr in tqdm(enumerate(list(grids['area']))):
            # print(id)
            if grids['lat'][id] != lat0:
                lat0 = grids['lat'][id]
                theta = MathTool.angular_distance(grids['lat'][id], grids['lon'][id], points['lat'],
                                                  points['lon'])

                '''calculate the displacement'''
                a = self._getFunc(PL=pl.configure(residence=theta), variable=variable)
                temp = a * rr
            else:
                temp = np.roll(temp.reshape((Num, -1)), shift=1, axis=1)
                temp = temp.flatten()

            dis += temp[:,None] @ grids['EWH'][id][None, :]

        return dis

class Grids_generation:
    @staticmethod
    def Equal_angular_distance(resolution=0.5, Earth_radius=EarthConstant.radiusm):
        """

        :param Earth_radius: [meter]
        :param resolution: [degree]
        :return: grid, dict
        """
        lat, lon = MathTool.get_global_lat_lon_range(resolution=resolution)
        lon, lat = np.meshgrid(lon, lat)
        '''S = a*a*cosθ*dθ*dλ 全球格网的面积'''
        area = np.cos(np.deg2rad(lat)) * np.deg2rad(resolution) ** 2 * Earth_radius ** 2
        '''通通转为(180*360,)便于后期计算'''
        grids = {
            'lat': lat.flatten(),
            'lon': lon.flatten(),
            'area': area.flatten()
        }

        return grids

