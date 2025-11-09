# SAGEA-fluid

## 1. Description

SAGEA-fluid is an open-source Python solver for evaluating the self-attraction and loading effect, geocenter motion, and Earth orientation parameters (polar motion and length-of-day variation) excited by Earth’s surface fluids. Building on the strengths of SAGEA (Liu et., al. 2025) with a great variety of data post-processing options, SAGEA-fluid enables the integration of multi-source surface fluid for geophysical effect estimation. This capability makes it possible for users who are not experts in geodesy to independently investigate surface fluid process. The below figure illustrates the overall workflow and interconnections within SAGEA-fluid. The innermost circle represents the specific fluid sources ingested by the toolbox, while adjacent light-grey ring provides a simple classification of the data types associated with these sources, including storage, pressure, and flux. The outermost ring depicts the main functional modules and processing mechanisms of SAGEA-fluid. The integrated design highlights the flexibility of SAGEA-fluid in accommodating multi-source datasets and generating harmonized geophysical diagnostics.

<img src="image/SAGEA-fluid structure.png" style="zoom: 15%;" />

SAGEA-fluid provides an accessible and extensible toolset for both geodesy specialists and non-specialists, supporting research on a wide range of Earth system processes. Potential applications include regional sea-level budget assessments under climate change, monitoring ice-sheet mass loss with satellite gravimetry, investigating atmosphere–ocean–hydrosphere–solid Earth interactions, and contributing to the establishment and maintenance of modern high-precision terrestrial reference frames.

When referencing this work, please cite: 

## 2. Contact

Weihang Zhang (zwh_cge@hust.edu.cn), Fan Yang ([fany@plan.aau.dk]())

## 3. Features

- SAGEA-fluid provides the open-source Python toolbox integrating self-attraction and loading (SAL) effects, GRACE geocenter motion (GCM), and Earth orientation parameters (EOPs) calculations in a unified framework.
- The toolbox enables flexible incorporation of multi-source fluid datasets (atmosphere, ocean, hydrology, cryosphere) and supports both GRACE-based and model-based inputs for interdisciplinary studies
- Advanced algorithms implemented include both spatial/pseudo-spectral methods for SAL calculations, GRACE low-degree terms both for geocenter motion and Earth's dynamic oblateness with SAL correction, and refined vertical integration for motion terms of EOPs.
- Provides flexible input/output options, allowing users to explore fluid-induced geophysical processes efficiently.

## 4. Installation

This program homepage is: [https://github.com/NCSGgroup/SAGEA-fluid](https://github.com/NCSGgroup/SAGEA-fluid)

Use this code to download this project.

`git clone https://github.com/NCSGgroup/SAGEA-fluid`

This project is developed based on Python 3.9 and the dependencies are listed in `requirememts.txt`.

Use these coded to download the dependencies:

`python -m pip install -r requirements.txt`

To ensure the successful running and validation of programs, please collect the corresponding auxiliary files and verification files at https://figshare.com/articles/dataset/Auxiliary_and_validation_data_for_SAGEA-fluid/30104062, and place folder  `data` in the project.

## 5. Quick Start

Several demo programs are under the direction `./demo/` allow users to quickly become familiar with estimating self-attraction and loading (SAL), geocenter motion (GCM), and Earth orientation parameters (EOPs).  Since SAGEA-fluid supports multiple fluid sources as inputs, demo for SAL are provided using both numerical models (i.e., ERA5 from Hersbach et., al. 2020 and ECCO from Pnote et., al. 2024) and GRACE gravity fields (CSR RL06, UTCSR, 2018). In addition to estimate GCM , a demo for estimation of GRACE C~20~ is also provided. For EOPs, which include polar motion (PM) and length-of-day (LOD) variation, each of these can be further separated into mass and motion terms. Thus, PM and LOD variation are divided into four demo programs for users using. 

### 5.1 Self-attraction and loading effect

1. `demo_NM()` in `./demo/gravitational_effect/demoSAL.py` shows the atmospheric surface pressure and ocean bottom (numerical models) as inputs to estimate SAL effect. 
2. `demo_GO()` in `./demo/gravitational_effect/demoSAL.py` exhibits the SAL effect when input is GRACE GSM.

### 5.2 Geocenter motion from GRACE

1. `demo_GCM()` in `./demo/geocenter_motion/demoGCM.py` is an example of GRACE GCM estimation, the results show the GSM-like (only hydrology and ice sheets) and full-geocenter (including atmosphere and ocean) performance in all axes.
2. `demo_J2()` in `./demo/geocenter_motion/demoGCM.py` is an example of GRACE C~20~ estimation, users can get C~20~ in Stokes coefficients during 2009.01 to 2009.12 when running this demo.

### 5.3 Earth orientation parameters

1. `demo_PM_mass_term()` in `./demo/earth_orientation/demoEOP.py` shows estimation of mass term of polar motion by SAGEA-fluid, taking the numerical models and calculation of AAM and OAM as an example.
2. `demo_PM_motion_term()` in `./demo/earth_orientation/demoEOP.py` shows estimation of motion term of polar motion by SAGEA-fluid, taking the numerical models and calculation of AAM and OAM as an example.
3. `demo_LOD_mass_term()` in `./demo/earth_orientation/demoEOP.py` shows estimation of mass term of length-of-day variation by SAGEA-fluid, taking   the GRACE GSM and calculation of HIAM and SLAM as an example.
4. `demo_LOD_motion_term()` in `./demo/earth_orientation/demoEOP.py` shows estimation of motion term of length-of-day variation by SAGEA-fluid, taking the numerical models and calculation of AAM and OAM as an example.



## 6. Detailed Tutorial

The tutorials below assume that the necessary Python libraries have already been imported. The import statements for the three core modules are as follows:

```python
`from pysrc.SAL.SeaLevelEquation import PseudoSpectralSLE,SpatialSLE`
```

```python
`from pysrc.GCM.GeocenterMotion import GeocenterMotion`
```

```python
`from pysrc.EOP.EarthOrientaition import EOP`
```



### 6.1 SAL

When estimating SAL using SAGEA-fluid, the procedure generally consists of three steps. The pseudo-spectral method is used here as an example.

- **Step 1: Import the input data.**

  `SAL = PseudoSpectralSLE(SH, lmax)`, where `SH` denotes the input spherical harmonic coefficients expressed in equivalent water height (EWH);

- **Step 2: Configure the model settings.**

  `SAL.setLatLon(lat, lon)`specifies the latitude and longitude associated with the input data;  

  `SAL.setLoveNumber(lmax, method = LoveNumber_type)` selects the Love numbers to be applied. If this option is not specified, the default Love numbers from Wang et al. (2012) consistent with the input degree are used.

- **Step 3: Run the sea level equation.**

  `Results = SAL.SLE(mask, rotation=True, isLand=True)`, where `mask` specifies the ocean function, for which SAGEA-fluid provides a default. The `rotation` parameter controls whether rotational feedback is included. The `isLand` parameter indicates whether water exchange between land and ocean is allowed.

After solving, users may obtain different output variables such as `Input`, `RSL_SH`, `Quasi_RSL_SH,` `RSL`, `GHC`, `VLM`, or `mask`. For example, the spatial distribution of relative sea-level change (EWH, in meters) can be extracted using `Results['RSL']`.

SAGEA-fluid also supports a spatial-domain method. In this case, step 1 is replaced by `SAL = SpatialSLE(grid, lat, lon)`, which does not require separately setting latitude and longitude.

### 6.2 GCM

The estimation of GRACE geocenter motion follows a similar three-step workflow.

- **Step 1: Import the input data.**

  `GCM = GeocenterMotion(GRACE, OceanSH, GAD, lmax)`, where `GRACE`, `OceanSH`, and `GAD` are all the mass coefficients of surface loads.

- **Step 2: Configure the spatial resolution.**

  `GCM.setResolution(resolution)`, which ensures consistency between the input data and the ocean function. The default resolution is 1$^{\circ}$.

- Step 3: Run the GRACE-OBP. 

  `Results = GCM.Low_Degree_Term(mask, buffer, GRD=True, rotation=True)`,  where GRD controls whether SAL effects are included, rotation determines whether rotational feedback is considered, and buffer specifies the spatial buffer width used for leakage correction (default 0 km).

Users may then retrieve the estimated low-degree terms as Stokes coefficients using `Results['Stokes']` or as mass coefficients using `Results['Mass']`. For convenience, geocenter motion can also be obtained directly via `Results = GCM.GSM_Like(mask, GRD, rotation, buffer)`, and individual geocenter components can be accessed through the X, Y or Z options (in meters).

### 6.3 EOPs

The estimation of Earth orientation parameters in SAGEA-fluid is performed separately for mass terms and motion terms. The relevant commands are listed below.

- **Mass term (spectral method).**
  `EOPs = EOP().PM_mass_term_SH(SH, isMas=False)`;
  `EOPs = EOP().LOD_mass_term_SH(SH, isMs=False)`, where `SH` represents surface fluid spherical harmonic coefficients (Stokes coefficients). If `isMas` is True, polar motion is output in milliarcseconds; if `isMs` is True, length-of-day variation is in milliseconds.
- **Mass term (spatial method).**
  `EOPs = EOP().PM_mass_term(Ps, lat, lon, isMas=False)`;
  `EOPs = EOP().LOD_mass_term(Ps, lat, lon, isMs=False)`, where `Ps` is the surface pressure field (Pa), and `[lat, lon]` correspond to the grid coordinates.
- **Motion term.**
  `EOPs = EOP().PM_motion_term(Us, Vs, lat, lon, levPres, type, isMas=False)`;
  `EOPs = EOP().LOD_motion_term(Us, lat, lon, levPres, type, isMs=False)`, where `[Us, Vs]` are the zonal and meridional wind or current components (m s$^{-1}$), and `levPres` denotes atmospheric pressure levels or ocean layer depths. The `type` parameter specifies whether atmospheric or oceanic angular momentum is used.

Users may extract specific EOP components such as $\chi_{1}$, $\chi_{2}$, or $\chi_{3}$ using statements like `EOPs['chi1']` or `EOPs['chi3']`. Note that the type parameter must be assigned using the enumeration class in SAGEA-fluid. Therefore, ensure that `EnumClass` has been imported beforehand, and after that, using `type=EnumClass.EAMtype.AAM` to set the desired source of excitation.

## Reference

Hersbach H, Bell B, Berrisford P, et al. The ERA5 global reanalysis. *Q J R Meteorol Soc*. 2020; 146: 1999–2049. https://doi.org/10.1002/qj.3803

Liu, S., Yang, F., & Forootan, E. (2025). SAGEA: A toolbox for comprehensive error assessment of GRACE and GRACE-FO based mass changes. Computers & Geosciences, 196, 105825. https://doi.org/10.1016/j.cageo.2024.105825

Ponte, R. M., Zhao, M., & Schindelegger, M. (2024). How well do we know the seasonal cycle in ocean bottom pressure? *Earth and Space Science*, 11, e2024EA003661. https://doi.org/10.1029/2024EA003661

University Of Texas Center For Space Research (UTCSR) 2018 GRACE STATIC FIELD GEOPOTENTIAL COEFFICIENTS CSR RELEASE 6.0. DOI: https://doi.org/10.5067/GRGSM-20C06

Wang, H., Xiang, L., Jia, L., Jiang, L., Wang, Z., Hu, B., & Gao, P. (2012). Load Love numbers and Green’s functions for elastic Earth models PREM, iasp91, ak135, and modified models with refined crustal structure from Crust 2.0. Computers Geosciences, 49 , 190-199. https://doi.org/10.1016/j.cageo.2012.06.022
