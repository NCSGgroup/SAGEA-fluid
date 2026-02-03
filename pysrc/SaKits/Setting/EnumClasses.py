from enum import Enum

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