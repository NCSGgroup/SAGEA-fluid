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