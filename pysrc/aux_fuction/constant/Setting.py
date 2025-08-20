"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/3 10:44
@Description:
"""

from enum import Enum


class ForceFields(Enum):
    ERAinterim = 0
    ERA5 = 1


class IntegralChoice(Enum):
    GFZ06VI = 1
    GFZ05VI = 2
    GFZ04VI = 3
    EFVI = 4
    SphereSP = 5
    NonsphereSP = 6


class DataType(Enum):
    TEMP = 0
    SHUM = 1
    PSFC = 2
    PHISFC = 3


class InterpOption(Enum):
    Blockmean = 0
    Bilinear = 1
    IDW = 2
    Nearest = 3
    NoInterp = 4


class SynthesisType(Enum):
    """pure analysis or the geo-potential"""
    synthesis = 0
    '''Equivalent water height'''
    EWH = 1
    '''Geoid height (undulation)'''
    Geoidheight = 2
    '''Surface pressure'''
    Pressure = 3
    '''!Gravity disturbance, NOT the free air anomaly!!!'''
    GravityDisturbance = 4
    '''!Free air gravity anomaly, NOT the gravity disturbance!!!'''
    FreeAirGraviyAnomaly = 5
    '''!Potential (anomaly) - Blakely (7.2)'''
    Potential = 6


class HarAnalysisType(Enum):
    """pure analysis or the geo-potential"""
    analysis = 0
    '''Equivalent water height'''
    EWH = 1
    '''Geoid height'''
    GeoidHeight = 2
    '''Surface pressure'''
    Pressure = 3
    '''Inner integral: pressure/g'''
    InnerIntegral = 4


class Constants:
    """gas constant for dry air"""
    Rd = 287.00
    # Rd = 287.06
    '''gravity constant g defined by WMO'''
    g_wmo = 9.80665
    ''' water density'''
    rho_water = 1025.0


class EllipsoidType(Enum):
    """
    we provide two types of reference ellipsoid.
    """
    GRS80 = 0
    WGS84 = 1
    GRS80_IERS2010 = 2
    gif48 = 3  # mostly used for gravity field application


class LoveNumberType(Enum):
    PREM = 1
    AOD04 = 2
    Wang = 3
    IERS = 4


class AODtype(Enum):
    ATM = 0
    OCN = 1
    GLO = 2
    OBA = 3



class TidesType(Enum):
    # S1 = 0
    # S2 = 1
    # S3 = 2
    # M2 = 3
    # P1 = 4
    # K1 = 5
    # N2 = 6
    # L2 = 7
    # T2 = 8
    # R2 = 9
    # T3 = 10
    # R3 = 11
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

'''Made by Cheung'''
class FileType(Enum):
    SP = 0
    UPPER = 1
    MEAN = 2
    AOD = 3
    CRA = 4
    CRA1 = 5
    CRA2 = 6
    Delta = 7

class CStype(Enum):
    C = 0
    S = 1

class setting:
    """
    system setting
    """
