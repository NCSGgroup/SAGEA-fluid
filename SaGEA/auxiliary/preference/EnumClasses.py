from enum import Enum


def match_string(name, obj, ignore_case=False):
    obj_list = list(obj)
    names = [obj_list[i].name for i in range(len(obj_list))]

    if ignore_case:
        names = [names[i].lower() for i in range(len(names))]
        name = name.lower()

    assert name in names

    return obj_list[names.index(name)]

'''A'''
class AODtype(Enum):
    ATM = 0
    OCN = 1
    GLO = 2
    OBA = 3


'''B'''
class BasinName(Enum):
    Amazon = 1
    Amur = 2
    Antarctica = 3
    Aral = 4
    Brahmaputra = 5
    Caspian = 6
    Colorado = 7
    Congo = 8
    Danube = 9
    Dnieper = 10
    Euphrates = 11
    Eyre = 12
    Ganges = 13
    Greenland = 14
    Indus = 15
    Lena = 16
    Mackenzie = 17
    Mekong = 18
    Mississippi = 19
    Murray = 20
    Nelson = 21
    Niger = 22
    Nile = 23
    Ob = 24
    Okavango = 25
    Orange = 26
    Orinoco = 27
    Parana = 28
    Sahara = 29
    St_Lawrence = 30
    Tocantins = 31
    Yangtze = 32
    Yellow = 33
    Yenisey = 34
    Yukon = 35
    Zambeze = 36
    Ocean = 37


'''C'''


'''D'''

class DataType(Enum):
    TEMP = 0
    SHUM = 1
    PSFC = 2
    PHISFC = 3
class Displacement(Enum):
    Vertical = 0
    Horizontal = 1
    Geoheight = 2
    Gravity = 3


'''E'''
class EmpiricalDecorrelationType(Enum):
    PnMm = 1
    window_stable = 2
    window_Wahr2006 = 3
    window_Duan2009 = 4


'''F'''
class Frame(Enum):
    CM = 0
    CF = 1
    CE = 2


'''G'''
class GIAModel(Enum):
    Caron2018 = 1
    Caron2019 = 2
    ICE6GC = 3
    ICE6GD = 4

class GeometricCorrectionAssumption(Enum):
    Sphere = 1
    Ellipsoid = 2
    ActualEarth = 3

class GridFilterType(Enum):
    VGC = 1

class GreenFunction(Enum):
    PointLoad = 1
    DiskLoad = 2
    # FastPoint = 3
    # FastDisk = 4

'''H'''

'''I'''

'''J'''

'''K'''

'''L'''
class L2DataServer(Enum):
    GFZ = 1
    ITSG = 2

class L2ProductType(Enum):
    GSM = 1
    GAA = 2
    GAB = 3
    GAC = 4
    GAD = 5

class L2InstituteType(Enum):
    CSR = 1
    GFZ = 2
    JPL = 3
    COST_G = 4
    ITSG = 5

class L2Release(Enum):
    RL05 = 5
    RL06 = 6
    RL061 = 61
    RL062 = 62
    RL063 = 62

    ITSGGrace2014 = 1002014
    ITSGGrace2016 = 1002016
    ITSGGrace2018 = 1002018
    ITSGGrace_operational = 2002018

class L2ProductMaxDegree(Enum):
    Degree60 = 60
    Degree90 = 90
    Degree96 = 96
    Degree120 = 120

class L2LowDegreeType(Enum):
    Deg1 = 1
    C20 = 2
    C30 = 3

class L2LowDegreeFileID(Enum):
    TN11 = 11
    TN13 = 13
    TN14 = 14

class LoveNumberMethod(Enum):
    PREM = 1
    AOD04 = 2
    Wang = 3
    IERS = 4

class LoveNumberType(Enum):
    VerticalDisplacement = 1
    HorizontalDisplacement = 2
    GravitationalPotential = 3
    # Do not change the order

class LLN_Data(Enum):
    PREM = 1
    REF = 2
    Wang = 3
    iasp91 = 4
    ak135 = 5
    iasp91hard = 6
    ak135hard = 7
    PREMhard = 8
    PREMsoft = 9

class LLN_variable(Enum):
    h = 1
    k = 2
    l = 3

class LeakageMethod(Enum):
    Addictive = 1
    Multiplicative = 2
    Scaling = 3
    ScalingGrid = 4
    Iterative = 5
    DataDriven = 6
    ForwardModeling = 7
    BufferZone = 8


'''M'''


'''N'''


'''O'''


'''P'''
class PhysicalDimensions(Enum):
    Dimensionless = 0
    EWH = 1
    Pressure = 2
    Density = 3
    Geoid = 4
    Gravity = 5
    HorizontalDisplacementEast = 6
    HorizontalDisplacementNorth = 7
    VerticalDisplacement = 8
    InnerIntegral = 9


'''Q'''


'''R'''
class SLEReference(Enum):
    LandLoad = 1
    LandLoadSigma = 2
    RSL = 3
    RSLSigma = 4
    GHC = 5
    GHCSigma = 6
    VLM = 7
    VLMSigma = 8

'''S'''
class Satellite(Enum):
    GRACE = 1
    GRACE_FO = 2


class SeaLevelMethod(Enum):
    GreenConvolution = 1
    QuasiSpectral = 2

class SHCDecorrelationType(Enum):
    PnMm = 1
    SlideWindowSwenson2006 = 2
    SlideWindowStable = 3

class SHCFilterType(Enum):
    Gaussian = 1
    Fan = 2
    AnisotropicGaussianHan = 3
    DDK = 4
    VGC = 5



'''T'''
class TidesType(Enum):
    P1 = 0
    S1 = 1
    K1 = 2
    T2 = 3
    S2 = 4
    R2 = 5
    T3 = 6
    S3 = 7
    R3 = 8
    N2 = 9
    M2 = 10
    L2 = 11

'''U'''


'''V'''
class VaryRadiusWay(Enum):
    """for VGC filter"""
    sin = 1
    sin2 = 2


'''W'''


'''X'''


'''Y'''


'''Z'''
