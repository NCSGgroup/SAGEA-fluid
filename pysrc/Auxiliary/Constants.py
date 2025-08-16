class GeoConstants:
    density_earth = 5517  # unit[kg/m3]
    # radius_earth = 6378136.3  # unit[m]
    radius_earth =6378137.0
    GM = 3.9860044150E+14  # unit[m3/s2]

    """gas constant for dry air"""
    Rd = 287.00
    # Rd = 287.06

    '''gravity constant g defined by WMO'''
    g_wmo = 9.80665
    # g_wmo = 9.7

    ''' water density'''
    # density_water = 1000.0
    density_water = 1025.0
    ggg = 6.67384e-11

    Mass = g_wmo*radius_earth**2/ggg

class EarthConstant:
    ggg = 6.67384e-11  # % Newton's constant (SI units)
    grav = 9.80665  # % Surface gravity (m/s/s)
    rhow = 1025.0  # % density of pure water(kg/m^3)
    rhoear = 5517  # unit[kg/m3]
    radiusm = 6378136.3  # unit[m]
    Mass = grav * radiusm ** 2 / ggg

    Omega = 7.2921e-5   # % Earth's mean rotation velocity (rad/s)
    A = 8.0077e37   # % Average moment of inertia at the equator (kg·m^2)
    C = 8.0345e37   # % moment of inertia at the polar (kg·m^2)
    Chandler = 1.6490e-7    # % Chandler swing frequency (s^-1)
    k2 = 0.3055     # % tidal love number (geopential)
    h2 = 0.6149     # % tidal love number (displacement)

class PMConstant:
    # radius = 6371000 # Earth mean radius, unit is m
    radius = 6378136.3
    rho_water = 1025.0  #water density, unit is (kg/m^3)
    rho_earth = 5517  # unit[kg/m3]

    grav = 9.80665  # % Surface gravity (m/s^2)
    ggg = 6.67384e-11  # % Newton's constant (SI units)
    Mass = grav * radius ** 2 / ggg

    omega = 7.292115e-5 # Earth mean angular velocity, unit is (rad^-1)
    k2 = 0.295  #rotational Love number of degree 2
    k2_load = -0.301
    ks = 0.938  #secular (fluid limit) Love number
    Cm = 7.1237e37  #(3,3) component mantle tensor of inertia, unit is (kg m^2)
    Am = 7.0999e37  #(1,1) component mantle tensor of inertia, unit is (kg m^2)
    # Cm = 8.01736e37  #(3,3) component mantle tensor of inertia, unit is (kg m^2)
    # Am = 8.01014e37  #(1,1) component mantle tensor of inertia, unit is (kg m^2)
    LOD = 86400 # 24h=1440min=86400s
    pass


