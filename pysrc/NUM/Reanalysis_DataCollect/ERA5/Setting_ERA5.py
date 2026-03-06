from enum import Enum


class MADML_Type(Enum):
    Geopotential = 0
    U_wind = 1
    V_wind = 2
    Temporature = 3
    Humidity = 4

class MADPL_Type(Enum):
    Geopotential = 0
    U_wind = 1
    V_wind = 2
    Temporature = 3
    Humidity = 4

class MADSL_Type(Enum):
    SurPressure = 0
    U_wind = 1
    V_wind = 2
    Geopotential = 3

class HDPL_Type(Enum):
    Geopotential = 0
    Humidity = 1
    Temporature = 2
    U_wind = 3
    V_wind = 4
    Vertical_Velocity = 5


class HDSL_Type(Enum):
    Geopotential = 0
    SurPressure = 1
    U_wind = 2
    V_wind = 3
    Temperature = 4
