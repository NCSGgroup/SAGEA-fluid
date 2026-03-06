from enum import Enum


class AODtype(Enum):
    ATM = 0
    OCN = 1
    GLO = 2
    OBA = 3


class Displacement(Enum):
    Vertical = 0
    Horizontal = 1
    Geoheight = 2
    Gravity = 3

class EAMtype(Enum):
    AAM = 1
    OAM = 2
    HAM = 3
    SLAM = 4


class Frame(Enum):
    CM = 0
    CF = 1
    CE = 2

class GreenFunction(Enum):
    PointLoad = 1
    DiskLoad = 2
    # FastPoint = 3
    # FastDisk = 4

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
