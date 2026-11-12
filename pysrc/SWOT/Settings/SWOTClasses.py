from enum import Enum

class KaRInL2LRSSH(Enum):
    Basic =0
    Expert = 1
    WaveWind = 2
    Unsmooth = 3

class L2Nadir(Enum):
    GPR = 0
    GPN = 1
    GPS = 2

class TrackType(Enum):
    Along_Track = 0
    Cross_Track = 1

class Scending(Enum):
    Descending = 0
    Ascending = 1

class RefineType(Enum):
    Yu = 0
    Zhang = 1
    Nothing = 2

class BGtype(Enum):
    MSSCLS = 0
    EGM2008 = 1
    MSS = 2
    N = 3


class VariableType(Enum):
    ssh = 0
    ssha = 1
    swh = 2
    xover = 3
    mss_cnes = 4
    mss_dtu = 5
    mdt = 6
    ged = 7
    tides = 8
    difGed = 9        # SSH-EGM2008-MDT
    difMSS = 10       # SSH-MSS-MDT
    difYu = 11        # MSS-EGM2008-MDT