from enum import Enum

class KaRInL2LRSSH(Enum):
    Basic =0
    Expert = 1
    WaveWind = 2
    Unsmooth = 3

class TrackType(Enum):
    Along_Track = 0
    Cross_Track = 1

class Scending(Enum):
    Descending = 0
    Ascending = 1

class RefineMethod(Enum):
    Yu = 0
    Zhang = 1
    Nothing = 2

class SSHData(Enum):
    ssh = 0
    ssha = 1
    swh = 2
    xover = 3
    mss_cnes = 4
    mss_dtu = 5
    mdt = 6
    geoid = 7
    tides = 8