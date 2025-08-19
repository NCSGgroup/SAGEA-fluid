
import os
import warnings

import numpy as np
import xarray

from SaGEA.auxiliary.preference.EnumClasses import AODtype,TidesType,DataType
from pysrc.LoadFile.DataClass import SHC
from SaGEA.post_processing.geometric_correction.old.GeoMathKit import GeoMathKit



class LoadFields:
    """
    This class deals with loading all necessary inputs for one cycle of AOD computation.
    """
    TimeEpoch = ["00:00:00","03:00:00" ,"06:00:00","09:00:00","12:00:00","15:00:00","18:00:00", "21:00:00"]

    def __init__(self, data_path='H:/ERA5/model level/',GeoHeight='model'):
        self.__epoch_data = {}
        self.__meanfield = None
        self.__modelLevel = None
        self.__path = data_path
        self.__lat, self.__lon = None, None
        self.__nLat, self.__nLon = None, None
        self.__level = None
        self.__q_level = None
        self.a = np.load('H:/Paper3/paper_data/Auxiliary/ABCoefficients137.npy')[0]
        self.b = np.load('H:/Paper3/paper_data/Auxiliary/ABCoefficients137.npy')[1]
        self.GeoHeight = GeoHeight

        pass

    def setTime(self, date='2010-01-01', time='00:00:00', OnlyPressure=False):
        """
        set the time epoch of data that will be used in later computation.
        :param OnlyPressure: if True, then only pressure is read.
        :param date:
        :param time:
        :return:
        """

        assert time in self.TimeEpoch
        str1 = date.split('-')
        str2 = time.split(':')

        sstr1 = str1[0] + str1[1] + str1[2]
        sstr2 = sstr1 + str2[0]


        path_dir = os.path.join(self.__path, str1[0], sstr1)

        flag_sp = 'sp-' + sstr2 + '.nc'
        flag_t = 'TEM-' + sstr2 + '.nc'
        flag_q = 'SHM-' + sstr2 + '.nc'
        flag_z = 'GPT-' + str1[0] + '.nc'


        '''name of each data'''
        PHISFC = os.path.join(self.__path, flag_z)
        PSFC = os.path.join(path_dir, flag_sp)
        TEMP = os.path.join(path_dir, flag_t)
        SHUM = os.path.join(path_dir, flag_q)


        f1 = xarray.open_dataset(PHISFC)
        lat0, lon0 = f1['latitude'].values, f1['longitude'].values
        phisfc = f1['z'].values[0].flatten()
        self.__epoch_data[DataType.PHISFC] = phisfc

        f = xarray.open_dataset(PSFC)
        lat1, lon1 = f['latitude'].values, f['longitude'].values
        ps = f['sp'].values[0].flatten()
        self.__epoch_data[DataType.PSFC] = ps

        if OnlyPressure:
            assert (lat0 == lat1).all()
            assert (lon0 == lon1).all()
            self.__lat, self.__lon = lat0, lon0
            print(PHISFC)
            print(PSFC)
            return self
        if self.GeoHeight in ['pressure','pressure level','Pressure']:

            ft = xarray.open_dataset(TEMP)
            lat2, lon2 = ft['latitude'].values, ft['longitude'].values
            if 'level' in ft.variables:
                TEMP_level = ft['level'].values * 100
            elif 'pressure_level' in ft.variables:
                TEMP_level = ft['pressure_level'].values[::-1]*100
            data_TEMP = []
            for i in range(len(ft['t'][0, :, 0, 0])):
                data_TEMP.append(ft['t'].values[0, i].flatten())
            data_TEMP = np.array(data_TEMP)
            self.__epoch_data[DataType.TEMP] = data_TEMP

            fs = xarray.open_dataset(SHUM)
            lat3, lon3 = fs['latitude'].values, fs['longitude'].values
            if 'level' in fs.variables:
                SHUM_level = fs['level'].values * 100
            elif 'pressure_level' in fs.variables:
                SHUM_level = fs['pressure_level'].values[::-1]*100
            data_SHUM = []
            for i in range(len(fs['q'][0, :, 0, 0])):
                data_SHUM.append(fs['q'].values[0, i].flatten())
            data_SHUM = np.array(data_SHUM)
            self.__epoch_data[DataType.SHUM] = data_SHUM

            assert (lat0 == lat1).all() and (lat0 == lat2).all() and (lat0 == lat3).all()
            assert (lon0 == lon1).all() and (lon0 == lon2).all() and (lon0 == lon3).all()

            iso_pres = []

            for i in range(len(TEMP_level)):
                pres_level = np.ones(len(ps)) * TEMP_level[i]
                if (TEMP_level[i] - ps < 0).all():
                    iso_pres.append(pres_level)
                    continue
                index = TEMP_level[i] - ps > 0
                pres_level[index] = ps[index]
                iso_pres.append(pres_level)

        elif self.GeoHeight in ['model','model level','Model']:
            ft = xarray.open_dataset(TEMP)
            lat2, lon2 = ft['latitude'].values, ft['longitude'].values
            if 'level' in ft.variables:
                TEMP_level = ft['level'].values
            elif 'model_level' in ft.variables:
                TEMP_level = ft['model_level'].values
            data_TEMP = []
            for i in range(len(ft['t'][0,:,0,0])):
                data_TEMP.append(ft['t'].values[0,i].flatten())
            data_TEMP = np.array(data_TEMP)
            self.__epoch_data[DataType.TEMP] = data_TEMP

            fs = xarray.open_dataset(SHUM)
            lat3, lon3 = fs['latitude'].values, fs['longitude'].values
            if 'level' in fs.variables:
                SHUM_level = fs['level'].values
            elif 'model_level' in fs.variables:
                SHUM_level = fs['model_level'].values
            data_SHUM = []
            for i in range(len(fs['q'][0,:,0,0])):
                data_SHUM.append(fs['q'].values[0,i].flatten())
            data_SHUM = np.array(data_SHUM)
            self.__epoch_data[DataType.SHUM] = data_SHUM
            assert (lat0 == lat1).all() and (lat0 == lat2).all() and (lat0 == lat3).all()
            assert (lon0 == lon1).all() and (lon0 == lon2).all() and (lon0 == lon3).all()

            iso_pres = []
            for i in range(len(TEMP_level)):
                pres_level = self.a[i]+self.b[i]*ps
                iso_pres.append(pres_level)
            pass

        self.__level = len(TEMP_level)
        self.__q_level = len(SHUM_level)
        self.__epoch_data['PsLevel'] = np.array(iso_pres)
        self.__lat, self.__lon = lat0, lon0
        print(PHISFC)
        print(PSFC)
        print(TEMP)
        print(SHUM)





        return self

    def setTime_for_tide(self, date='2010-01-01', time='06:00:00'):
        """
        set the time epoch of data that will be used in later computation.
        :param OnlyPressure: if True, then only pressure is read.
        :param date:
        :param time:
        :return:
        """

        assert time in self.TimeEpoch
        str1 = date.split('-')
        str2 = time.split(':')

        sstr1 = str1[0] + str1[1] + str1[2]
        sstr2 = sstr1 + str2[0]


        path_dir = os.path.join(self.__path, str1[0], sstr1)

        flag_sp = 'sp-'+sstr2+'.nc'
        PSFC = os.path.join(path_dir, flag_sp)

        f = xarray.open_dataset(PSFC)
        self.__lat,self.__lon = f['latitude'].values,f['longitude'].values
        ps = f['sp'].values[0]
        self.__epoch_data[DataType.PSFC] = ps

        return self

    def getField(self, kind: DataType = DataType.TEMP):
        return self.__epoch_data[kind]

    def getPressureLevel(self):
        return self.__epoch_data['PsLevel']

    def getLevel(self):
        return self.__level

    def getQLevel(self):
        return self.__q_level

    def getLatLon(self):
        """
        position of N*M points
        :return: geodetic latitude [dimension: N*M] and longitude [dimension: N*M] in degree
        """

        lon, lat = np.meshgrid(self.__lon, self.__lat)

        return lat.flatten(), lon.flatten()

    def getLatLon_v2(self):
        """
        position of N*M points
        :return: geodetic latitude [dimension: N] and longitude [dimension: M] in degree
        """
        return self.__lat, self.__lon

    def getNlatNlon(self):
        """
        position of N*M points
        :return: N, M
        """
        return len(self.__lat), len(self.__lon)



class LoadSH:
    """
    This class is a base class that deals with the gravity models reading.
    """

    def __init__(self):
        self.product_type = None
        self.modelname = None
        self.GM = None
        self.Radius = None
        self.maxDegree = None
        self.zero_tide = None
        self.errors = None
        self.norm = None

        self._sigmaC, self._sigmaS = None, None
        self._C, self._S = None, None
        pass

    def load(self, fileIn: str):
        return self

    def getCS(self, Nmax):

        assert Nmax >= 0

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        if end > len(self._C):
            warnings.warn('Nmax is too big')
        C_coef = GeoMathKit.CS_1dTo2d(CS=np.array(self._C[0:end].copy()))[None,:,:]
        S_coef = GeoMathKit.CS_1dTo2d(CS=np.array(self._S[0:end].copy()))[None,:,:]
        # C_coef = MathTool.cs_1dto2d(cs=self._C[0:end].copy())[None,:,:]
        # S_coef = MathTool.cs_1dto2d(cs=self._S[0:end].copy())[None,:,:]
        # print(f"C_coef is:{C_coef.shape}")
        shc = SHC(c=C_coef,s=S_coef)
        return shc

    def getCS_old(self, Nmax):

        assert Nmax >= 0

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        if end > len(self._C):
            warnings.warn('Nmax is too big')

        return self._C[0:end].copy(), self._S[0:end].copy()

    def getSigmaCS(self, Nmax):

        assert Nmax >= 0

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        if end > len(self._C):
            warnings.warn('Nmax is too big')

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        return self._sigmaC[0:end], self._sigmaS[0:end]

class Gif48_NoSigma(LoadSH):
    """
    specified to read gif48 gravity fields. Plus, the file that formats the same as GIF48 can be read as well.
    """

    def __init__(self):
        LoadSH.__init__(self)
        # self.__sigmaC, self.__sigmaS = None, None
        # self.__C, self.__S = None, None

    def load(self, fileIn: str):
        """
        load gif48 fields
        :param fileIn: gif48 file and its path
        :return:
        """

        flag = 0

        with open(fileIn) as f:
            content = f.readlines()
            pass

        flag = self.__header(content)

        self.__read(content[flag + 1:])

        return self

    def __header(self, content: list):
        """
        Read and identify the header of the gif48 gravity fields file
        :param content: content of the file
        :return: the line of "end_of_header"
        """

        flag = 0

        for i in range(len(content)):
            value = content[i].split()
            if len(value) == 0: continue

            if value[0] == 'product_type':
                self.product_type = value[1]
            elif value[0] == 'modelname':
                self.modelname = value[1]
            elif value[0] == 'earth_gravity_constant':
                self.GM = float(value[1])
            elif value[0] == 'radius':
                self.Radius = float(value[1])
            elif value[0] == 'max_degree':
                self.maxDegree = int(value[1])
            elif value[0] == 'errors':
                self.errors = value[1]
            elif value[0] == 'norm':
                self.norm = value[1]
            elif value[0] == 'tide_system':
                self.zero_tide = (value[1] == 'zero_tide')
            elif value[0] == 'end_of_head':
                flag = i
                break

        return flag

    def __read(self, content: list):
        """
        Start reading gif48 like line by line like below
        key    L    M    C    S
        :param content:
        :return: all the C, S coefficients in order
        """
        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0: continue

            l.append(value[1])
            m.append(value[2])
            C.append(value[3])
            S.append(value[4])

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)


        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))


        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S


class Gif48(LoadSH):
    """
    specified to read gif48 gravity fields. Plus, the file that formats the same as GIF48 can be read as well.
    """

    def __init__(self):
        LoadSH.__init__(self)
        # self.__sigmaC, self.__sigmaS = None, None
        # self.__C, self.__S = None, None

    def load(self, fileIn: str):
        """
        load gif48 fields
        :param fileIn: gif48 file and its path
        :return:
        """

        flag = 0

        with open(fileIn) as f:
            content = f.readlines()
            pass

        flag = self.__header(content)

        self.__read(content[flag + 1:])

        return self

    def __header(self, content: list):
        """
        Read and identify the header of the gif48 gravity fields file
        :param content: content of the file
        :return: the line of "end_of_header"
        """

        flag = 0

        for i in range(len(content)):
            value = content[i].split()
            if len(value) == 0: continue

            if value[0] == 'product_type':
                self.product_type = value[1]
            elif value[0] == 'modelname':
                self.modelname = value[1]
            elif value[0] == 'earth_gravity_constant':
                self.GM = float(value[1])
            elif value[0] == 'radius':
                self.Radius = float(value[1])
            elif value[0] == 'max_degree':
                self.maxDegree = int(value[1])
            elif value[0] == 'errors':
                self.errors = value[1]
            elif value[0] == 'norm':
                self.norm = value[1]
            elif value[0] == 'tide_system':
                self.zero_tide = (value[1] == 'zero_tide')
            elif value[0] == 'end_of_head':
                flag = i
                break

        return flag

    def __read(self, content: list):
        """
        Start reading gif48 like line by line like below
        key    L    M    C    S    sigma C    sigma S
        :param content:
        :return: all the C, S coefficients in order
        """
        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0: continue

            l.append(value[1])
            m.append(value[2])
            C.append(value[3])
            S.append(value[4])
            sigmaC.append(value[5])
            sigmaS.append(value[6])

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)
        sigmaC = np.array(sigmaC).astype(np.float64)
        sigmaS = np.array(sigmaS).astype(np.float64)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))
        self._sigmaC = np.zeros(len(l))
        self._sigmaS = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S
        self._sigmaC[(l * (l + 1) / 2 + m).astype(np.int64)] = sigmaC
        self._sigmaS[(l * (l + 1) / 2 + m).astype(np.int64)] = sigmaS


class SimpleSH(LoadSH):
    """
    specified to the ones in the simplest format (No header) like below:

    0  0  c s
    1  0  c s
    1  1  c s
    """

    def __init__(self):
        LoadSH.__init__(self)

    def load(self, fileIn: str):

        with open(fileIn) as f:
            content = f.readlines()
            pass

        l, m, C, S = [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0: continue

            l.append(value[0])
            m.append(value[1])
            C.append(value[2])
            S.append(value[3])

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)

        n = np.round(np.sqrt(len(l) * 2)) - 1
        assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))
        self._sigmaC = np.zeros(len(l))
        self._sigmaS = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S

        return self


class AOD_GFZ(LoadSH):
    """
    This class is used to read stokes coefficients at given epoch from AOD product.
    """

    def __init__(self):
        LoadSH.__init__(self)
        self.__kind = None
        self.__AODdir = None
        self.__box = {AODtype.ATM: 'atm',
                      AODtype.OCN: 'ocn',
                      AODtype.OBA: 'oba',
                      AODtype.GLO: 'glo'}
        self.__epoch = None

    def load(self, fileIn: str):
        """
        set the directory of AOD products to be read
        :param fileIn:
        :return:
        """
        self.__AODdir = fileIn
        return self

    def setTime(self, date: str, epoch: str):
        """
        specify the epoch of AOD file to be read
        :param date: eg., '2009-01-01'
        :param epoch: eg., '06:00:00'
        :return:
        """
        self.__epoch = epoch
        '''search the directory by month information'''
        root_path = self.__AODdir
        target = None

        class Break(Exception):
            pass

        try:
            for root, dirs, files in os.walk(root_path):
                for name in files:
                    if date in name:
                        target = os.path.join(root, name)
                        raise Break
        except Break as e:
            pass

        flag = 0

        if target is None:
            raise FileNotFoundError

        with open(target) as f:
            content = f.readlines()
            pass

        flag = self.__header(content)

        self.__read(content[flag + 1:])

        '''search the file by day information'''

        return self


    def setType(self, kind: AODtype = AODtype.ATM):
        """
        set the type to be read: ATM, OCN, GLO, OBA
        :return:
        """
        self.__kind = kind
        return self

    def __header(self, content: list):
        """
        Read and identify the header of the gif48 gravity fields file
        :param content: content of the file
        :return: the line of "end_of_header"
        """

        flag = 0

        for i in range(len(content)):

            value = content[i].split(':')
            if len(value) == 0:
                continue
            value[0] = value[0].strip()
            try:
                value[1] = value[1].replace('\n', '')
            except Exception as e:
                pass


            if value[0] == 'SOFTWARE VERSION':
                self.product_type = value[1]
            elif value[0] == 'PRODUCER AGENCY':
                self.modelname = value[1]
            elif value[0] == 'CONSTANT GM [M^3/S^2]':
                self.GM = float(value[1])
            elif value[0] == 'CONSTANT A [M]':
                self.Radius = float(value[1])
            elif value[0] == 'MAXIMUM DEGREE':
                self.maxDegree = int(value[1])
            elif value[0] == 'COEFFICIENT ERRORS (YES/NO)':
                self.errors = value[1]
            elif value[0] == 'norm':
                self.norm = value[1]
            elif value[0] == 'tide_system':
                self.zero_tide = (value[1] == 'zero_tide')
            elif (self.__box[self.__kind] in content[i]) and (self.__epoch in content[i]):
                flag = i
                break

        return flag

    def __read(self, content: list):
        """
        Start reading gif48 like line by line like below
        key    L    M    C    S    sigma C    sigma S
        :param content:
        :return: all the C, S coefficients in order
        """
        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0:
                continue
            if len(item.split(':')) > 1:
                break

            l.append(value[0])
            m.append(value[1])
            C.append(value[2])
            S.append(value[3])

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S


class AODtides(LoadSH):
    """
    This class is used to read stokes coefficients at given epoch from AOD product.
    """

    def __init__(self):
        LoadSH.__init__(self)
        self.__AODdir = None

    def load(self, fileIn: str):
        """
        set the directory of AOD products to be read
        :param fileIn:
        :return:
        """
        self.__AODdir = fileIn
        return self

    def setInfo(self, tide: TidesType, kind=AODtype.ATM, sincos='sin'):
        assert kind == AODtype.ATM or kind == AODtype.OCN
        assert sincos in ['sin', 'cos']

        '''search the directory by above information'''
        root_path = self.__AODdir
        target = None

        class Break(Exception):
            pass

        try:
            for root, dirs, files in os.walk(root_path):
                for name in files:
                    if (kind.name in name) and (tide.name in name):
                        target = os.path.join(root, name)
                        raise Break
        except Break as e:
            pass

        flag = 0

        if target is None:
            raise FileNotFoundError

        with open(target) as f:
            content = f.readlines()
            pass

        flag = self.__header(content, sincos)

        self.__read(content[flag + 1:])

        return self

    def __header(self, content: list, sincos: str):
        """
        Read and identify the header of the gif48 gravity fields file
        :param content: content of the file
        :return: the line of "end_of_header"
        """

        flag = 0

        for i in range(len(content)):
            value = content[i].split(':')
            if len(value) == 0:
                continue
            value[0] = value[0].strip()
            try:
                value[1] = value[1].replace('\n', '')
            except Exception as e:
                pass

            if value[0] == 'SOFTWARE VERSION':
                self.product_type = value[1]
            elif value[0] == 'PRODUCER AGENCY':
                self.modelname = value[1]
            elif value[0] == 'CONSTANT GM [M^3/S^2]':
                self.GM = float(value[1])
            elif value[0] == 'CONSTANT A [M]':
                self.Radius = float(value[1])
            elif value[0] == 'MAXIMUM DEGREE':
                self.maxDegree = int(value[1])
            elif value[0] == 'COEFFICIENT ERRORS (YES/NO)':
                self.errors = value[1]
            elif value[0] == 'norm':
                self.norm = value[1]
            elif value[0] == 'tide_system':
                self.zero_tide = (value[1] == 'zero_tide')
            elif sincos in content[i]:
                flag = i
                break

        return flag

    def __read(self, content: list):
        """
        Start reading gif48 like line by line like below
        key    L    M    C    S    sigma C    sigma S
        :param content:
        :return: all the C, S coefficients in order
        """
        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0:
                continue
            if len(item.split(':')) > 1:
                break

            l.append(value[0])
            m.append(value[1])
            C.append(value[2])
            S.append(value[3])

        l = np.array(l).astype(np.int64)
        m = np.array(m).astype(np.int64)
        C = np.array(C).astype(np.float64)
        S = np.array(S).astype(np.float64)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int64)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int64)] = S


def demo1():
    # a = Gif48().load('../data/Auxiliary/GIF48.gfc')
    a = Gif48().load('../data/Auxiliary/ITG-Grace2010s.gfc')
    C0, S0 = a.getCS(100)
    C1, S1 = a.getSigmaCS(100)
    pass


def demo2():
    a = SimpleSH().load('../data/Auxiliary/ocean360_grndline.sh')
    C0, S0 = a.getCS(360)
    pass


def demo3():
    """
    read AOD
    :return:
    """

    ad = AOD_GFZ().load('I:\GFZ\AOD1B07/').setType(AODtype.ATM).setTime('2005-01-01', '12:00:00')
    # shc = ad.getCS(ad.maxDegree)
    CS = ad.getCS(Nmax=180)
    # print(CS[0].shape)
    # print(shc.value.shape)


def demo4():
    """
    read AOD tides
    :return:
    """
    ad = AODtides().load('../data/Products/RL06_tides').setInfo(tide=TidesType.S1, kind=AODtype.OCN, sincos='sin')
    C, S = ad.getCS(ad.maxDegree)
    pass


if __name__ == '__main__':
    demo3()
