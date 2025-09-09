# SAGEA-fluid

## 1. Description

SAGEA-fluid is an open-source Python solver designed for the estimation of geophysical effects induced by surface fluid variations, including self-attraction and loading (SAL), geocenter motion (GCM), and Earth orientation parameters (EOPs) including polar motion (PM) and length-of-day (LOD) variations. Building upon the flexibility of the SAGEA (Liu et., al. 2005) framework in handling heterogeneous datasets, SAGEA-fluid enables the integration of multi-source atmospheric, oceanic, and hydrological data to compute corresponding geophysical excitations.

SAGEA-fluid provides an accessible and extensible toolset for both geodesy specialists and non-specialists, supporting research on a wide range of Earth system processes. Potential applications include regional sea-level budget assessments under climate change, monitoring ice-sheet mass loss with satellite gravimetry, investigating atmosphere–ocean–hydrosphere–solid Earth interactions, and contributing to the establishment and maintenance of modern high-precision terrestrial reference frames.

## 2. Contact

Weihang Zhang (zwh_cge@hust.edu.cn), Fan Yang ([fany@plan.aau.dk]())

## 3. Features

- SAGEA-fluid provides the open-source Python toolbox integrating self-attraction and loading (SAL) effects, GRACE geocenter motion (GCM), and Earth orientation parameters (EOPs) calculations in a unified framework.
- The toolbox enables flexible incorporation of multi-source fluid datasets (atmosphere, ocean, hydrology, cryosphere) and supports both GRACE-based and model-based inputs for interdisciplinary studies
- Advanced algorithms implemented include both spatial/pseudo-spectral methods for SAL calculations, GRACE low-degree terms both for geocenter motion and Earth's dynamic oblateness with SAL correction, and refined vertical integration for motion terms of EOPs.
- Provides flexible input/output options, allowing users to explore fluid-induced geophysical processes efficiently.

## 4. Installation

This program homepage is: https://github.com/NCSGgroup/SAGEA-fluid.git

Use this code to download this project.

`git clone https://github.com/NCSGgroup/SAGEA-fluid`

This project is developed based on Python 3.9 and the dependencies are listed in `requirememts.txt`.

Use these coded to download the dependencies:

`python -m pip install -r requirements.txt`

To ensure the successful running and validation of programs, please collect the corresponding auxiliary files and verification files 

## 5. Sample data

Since there are three geophysical effects in SAGEA-fluid, to estimate SAL, GRACE GCM, PM, and LOD variations, essential sample data for these four running demos are provided:

### **self-attraction and loading**

SAL induced from atmosphere (Suggested default path: /SAGEA-fluid/data/): monthly scale atmospheric surface pressure from ERA5. 

SAL induced from ocean (Suggested default path: /SAGEA-fluid/data/): monthly scale ocean bottom pressure from ECCO. 

SAL induced from hydrology and cryosphere (Suggested default path: /SAGEA-fluid/data/): monthly scale temporal gravity field model from CSR.

### Geocenter motion



### Earth orientation parameters

#### Polar motion



#### Length of day





## 6. Quick Start







Liu, S., Yang, F., & Forootan, E. (2025). SAGEA: A toolbox for comprehensive error assessment of GRACE and GRACE-FO based mass changes. Computers & Geosciences, 196, 105825. https://doi.org/10.1016/j.cageo.2024.105825
