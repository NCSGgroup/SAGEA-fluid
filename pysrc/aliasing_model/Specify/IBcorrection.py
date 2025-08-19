"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/11 18:55
@Description:
"""

import numpy as np
# import pyshtools as pysh
# from netCDF4 import Dataset

from SaGEA.post_processing.geometric_correction.old.GeoMathKit import GeoMathKit
from pysrc.aliasing_model.Specify.Harmonic import Harmonic, SynthesisType
from pysrc.load_file.LoadAOD import SimpleSH
from SaGEA.post_processing.geometric_correction.old.LoveNumber import LoveNumberType, LoveNumber


class LandSeaMask:

    def __init__(self, lat, lon, method: int = 3):
        """
        definition
        :param method: define which method to be used to derive the land-sea mask
        :param lat: one-dimension latitude in [degree]  [dimension N]
        :param lon: one-dimension longitude in [degree] [dimension M]
        """
        assert method in [1, 2, 3]
        self.__lat = lat
        self.__lon = lon
        self.__ocean = None
        if method == 1:
            self.__method1()
        elif method == 2:
            self.__method2()
        elif method == 3:
            self.__method3()

        pass

    def getOcean(self):
        """
        get the index of ocean grids from the whole defined by lat, lon
        :return: A vector that contains value in boolean, where TRUE denotes the ocean and FALSE denotes the land
        """
        return self.__ocean.copy()

    def __method1(self):
        """
        Use the topography file to distinguish the land and sea
        :return:
        """
        # lmax = 300
        # topo_coeff = pysh.SHCoeffs.from_file('../data//Auxiliary/srtmp300.msl.txt', lmax=lmax)
        #
        # # lat = np.arange(89.75, -90, -0.5)
        # # lon = np.arange(0.25, 360, 0.5)
        #
        # LN = LoveNumber('../data/Auxiliary/')
        # hm = Harmonic(LN, Parallel=5).setLoveNumMethod(LoveNumberType.Wang)
        #
        # SHC = GeoMathKit.CS_2dTo1d(topo_coeff.coeffs[0, :, :])
        # SHS = GeoMathKit.CS_2dTo1d(topo_coeff.coeffs[1, :, :])
        #
        # grids = hm.synthesis(Cnm=SHC, Snm=SHS, lat=self.__lat, lon=self.__lon, Nmax=lmax, kind=SynthesisType.synthesis)
        #
        # self.__ocean = grids <= 0

        pass

    def __method2(self):
        """
        Use the land-sea netcdf file from ECMWF to derive the land-sea mask
        https://confluence.ecmwf.int//display/TIGGE/Land-sea+mask
        The land-sea mask is a field that contains, for every grid point, the PROPORTION of land in the grid box.
        The values are between 0 (sea) and 1 (land).
        :return:
        """
        # OceanMask = Dataset('../data/Auxiliary/OceanMask.nc')
        #
        # lat = np.array(OceanMask.variables['latitude'][:], dtype=np.float)
        # lon = np.array(OceanMask.variables['longitude'][:], dtype=np.float)
        # Val = np.array(OceanMask.variables['lsm'][0, :, :], dtype=np.float)
        #
        # LN = LoveNumber('../data/Auxiliary/')
        # hm = Harmonic(LN, Parallel=5).setLoveNumMethod(LoveNumberType.Wang)
        #
        # Nmax = 200
        # Pnm = GeoMathKit.getPnm(lat, Nmax, 1)
        # cnm, snm = hm.analysis(Nmax, [Val.flatten()], lat, lon, Pnm, kind=HarAnalysisType.analysis)
        #
        # # lat1 = np.arange(89.75, -90, -0.5)
        # # lon1 = np.arange(0.25, 360, 0.5)
        # grids = hm.synthesis(Cnm=cnm, Snm=snm, lat=self.__lat, lon=self.__lon, Nmax=Nmax, kind=SynthesisType.synthesis)
        #
        # self.__ocean = grids < 0.3
        pass

    def __method3(self):
        """
        Stokes coefficients of a land-sea mask generated from the shape file, up to degree/order = 360.
        :return:
        """
        lmax = 360
        SHC, SHS = SimpleSH().load('H:/Paper3/paper_data/Auxiliary/ocean360_grndline.sh').getCS(lmax)

        # lat = np.arange(89.75, -90, -0.5)
        # lon = np.arange(0.25, 360, 0.5)

        LN = LoveNumber('H:/Paper3/paper_data/Auxiliary/')
        hm = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)
        PnmMat = GeoMathKit.getPnmMatrix(self.__lat, lmax, 2)
        grids = hm.synthesis(Cqlm=SHC, Sqlm=SHS, lat=self.__lat, lon=self.__lon, Nmax=lmax, PnmMat=PnmMat,kind=SynthesisType.synthesis)

        self.__ocean = grids > 0.5

        pass


class IBcorrection(LandSeaMask):

    def __init__(self, lat, lon, method: int = 3):
        """
        definition of these parameters refers to the super class
        :param lat:
        :param lon:
        :param method:
        """
        LandSeaMask.__init__(self, lat, lon, method)
        self.__lat = lat
        self.__lon = lon

        pass

    def correct(self, grids):
        """
        Inverse Barometric correction.
        See AOD RL06 handbook.
        :param grids: [N*M] meshgrids that will be posed with IB correction
        :return:
        """

        lonMesh, latMesh = np.meshgrid(self.__lon, self.__lat)
        '''Mask the land area to get pure ocean grid'''
        ocean = self.getOcean()
        '''get the lat, lon and gridVal of these ocean grids'''
        latOcean = np.deg2rad(latMesh[ocean].flatten())
        lonOcean = np.deg2rad(lonMesh[ocean].flatten())
        gridOcean = grids[ocean.flatten()].flatten()
        '''replace the pressure over oceans with the area-mean'''
        area = np.sum(np.cos(latOcean))
        tot = np.sum(np.cos(latOcean) * gridOcean)
        grids[ocean.flatten()] = tot / area

        return grids


