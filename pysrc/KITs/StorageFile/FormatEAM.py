import os
from datetime import datetime
import numpy as np
from pysrc.KITs.Setting.EnumClasses import EAMtype


class FormatEAM:
    def __init__(self):
        self.__fileDir = None
        self.__filename = f"SAGEA-Fluid_EAM.txt"
        self.__MassTerm = None
        self.__MotionTerm = None
        self.dates = None
        self.epoch = None
        # self.EndDates = None
        self.EAMtype = EAMtype.AAM
        self.FileTitle = "Effective Atmospheric Angular Momentum Functions (AAM)"

    def configure(self,filedir,mass_term:dict,motion_term:dict,dates_range:list,EAMtype=EAMtype.AAM):
        self.__fileDir = filedir
        self.__MassTerm = mass_term
        self.__MotionTerm = motion_term
        self.EAMtype = EAMtype
        if self.EAMtype == EAMtype.AAM:
            self.FileTitle = "Effective Atmospheric Angular Momentum Functions (AAM)"
        elif self.EAMtype == EAMtype.OAM:
            self.FileTitle = "Effective Oceanic Angular Momentum Functions (OAM)"
        elif self.EAMtype == EAMtype.HAM:
            self.FileTitle = "Effective Hydrological Angular Momentum Functions (HAM)"
        elif self.EAMtype == EAMtype.SLAM:
            self.EAMtype = "Effective Sea-Level Angular Momentum Functions (SLAM)"

        self.__initFile()

    def __initFile(self):
        if not os.path.exists(self.__fileDir):
            os.makedirs(self.__fileDir)
        self.__fileFullPath = os.path.join(self.__fileDir, self.__filename)
        return self

    def __convenrt_date(self,date_str='2002-01-01'):
        """Converting str date like '2002-01-01' to 01-Jan-2002."""
        date_obj = datetime.strptime(date_str,"%Y-%m-%d")
        return date_obj.strftime("%d-%b-%Y")
    def __current_time(self):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%d-%b-%Y %H:%M:%S")
        return formatted_time

    def __convert_to_gnss(self,date_str):
        date_obj = datetime.strptime(date_str,"%Y-%m-%d")
        return date_obj.strftime("%Y%m%d.0000")

    def DegreeWrite(self):
        with open(self.__fileFullPath, 'w') as file:
            file.write(f"{self.FileTitle}\n")
            file.write(f'\n')
            file.write("TITLE:\n"
                       f"\tMonthly estimates of degree-1 (geocenter) gravity coefficients, generated from\n"
                       f"\tGRACE (04-2002 - 06/2017) and GRACE-FO (06/2018 onward) RL0603 solutions.\n"
                       f"\tLast reported data point: {self.__convenrt_date(date_str=self.EndDates[-1])}.\n")
            file.write(f'\n')
            file.write(f"UPDATE HISTORY:\n"
                       f"\tCreated/updated {self.__current_time()}\n")
            file.write(f'\n')
            file.write(f"AUTHOR:\n"
                       f"\tWeihang Zhang, Ohio State University (OSU)\n"
                       f"\t\tContact: zhang.17371@osu.edu\n"
                       f"\tFan Yang, Aalborg University (AAU)\n"
                       f"\t\tContact: fany@plan.aau.dk\n")
            file.write(f'\n')
            file.write(f"REFERENCES:\n")
            file.write(f"\tSun, Y., R. Riva, and P. Ditmar (2016), Optimizing estimates of annual variations and\n"
                       f"\ttrends in geocenter motion and J2 from a combination of GRACE data and geophysical\n"
                       f"\tmodels, J. Geophys. Res. Solid Earth, 121, doi:10.1002/2016JB013073.\n")
            file.write(f"\n")
            file.write(f"\tSwenson, S., D. Chambers, and J. Wahr 2008  Estimating geocenter variations from a\n"
                       f"\tcombination of GRACE and ocean model output, J. Geophys. Res.,  113, B08410,\n"
                       f"\tdoi:10.1029/2007JB005338.\n")
            file.write(f"\n")
            file.write(f"DESCRIPTION:\n"
                       f"\tThis file contains estimates of degree-1 gravity coefficients for the GRACE Release-06 data,\n"
                       f"\tbased on ocean and atmospheric models and GRACE coefficients for degrees 2 and higher.\n"
                       f"\tThe original method was developed by [Swenson et al., 2008], and subsequently expanded by\n"
                       f"\t[Sun et al., 2016] to include the effects of the barystatic sea level fingerprint (i.e.,\n"
                       f"\tsolving the sea level equation using land mass changes as measured by GRACE) in the ocean mass\n"
                       f"\tcontribution to degree-1.\n")
            file.write(f'\n')
            file.write(f"\tThe implementation for degree-1 coefficients below uses the optimal parameter values are\n"
                       f"\t{self.Institute} degree and order 60, {self.Filter} and leakage correction with buffer width {self.bufferwidth} km\n"
                       f"\t by SAGEA-Fluid [Zhang et al., 2026]. These coefficients represent the degree 1 gravity\n"
                       f"\tcoefficients that should be added to the GSM coefficients to correct for geocenter motion,\n"
                       f"\trelative to the modeled atmosphere and ocean degree 1 coefficients (e.g., GAC and GAD). The\n"
                       f"\tcoefficients have been normalized using the GRACE standards. Prior to the inversion for\n"
                       f"\tdegree-1, GRACE-C20 has been replaced with the SLR-C20 (TN-14), and GIA has been corrected\n"
                       f"\twith {self.GIA}.\n")
            file.write(f'\n')
            file.write(f"\tTo use for land applications or to compute global ocean mass, use these coefficients along\n"
                       f"\twith GRACE GSM coefficients for degrees 2 and higher, along with a Degree 1 Love number\n"
                       f"\t(k1) = 0.021.\n")
            file.write(f"\n")
            file.write(f"\tFor ocean applications, add the degree 1 values found in the ocean bottom pressure product\n"
                       f"\t(GAD) to these values. If you are interested in the full ocean, land, atmosphere geocenter,\n"
                       f"\tadd the degree 1 values of the atmosphere/ocean product (GAC) to these values.\n")
            file.write(f"\n")
            file.write(f"SPECIAL NOTES:\n"
                       f"\t1) The native GRACE & GRACE-FO C20 coefficient has been replaced with SLR-C20 (TN-14v3)\n"
                       f"\t2) GIA has been corrected for subtracting the ICE6G-D (Peltier et al., 2018) prior to the\n"
                       f"\tdegree-1 inversion\n"
                       f"\t3) From 06/2019 on, C30 in GRACE-FO has been replaced with SLR-C30 (TN-14v3)\n")
            file.write(f'\n')
            file.write(f"FORMAT:\n"
                       f"\tThe data format is similar to that for GSM, GAD, and GAC files distributed by PODAAC, see\n"
                       f"\tGRACE Level-2 handbook for more information.\n")
            file.write(f'\n')
            file.write(f"# EGM Coefficient Record 2\n"
                       f" variables:\n"
                       f"  record_key:\n"
                       f"    key_name              : GRCOF2\n"
                       f"    long_name             : Earth Gravity Spherical Harmonic Model Format Type\n"
                       f"    coverage_content_type : referenceInformation\n"
                       f"    data_type             : string\n"
                       f"    comment               : 1st column\n"
                       f"  degree_index:\n"
                       f"    long_name             : spherical harmonic degree l\n"
                       f"    coverage_content_type : referenceInformation\n"
                       f"    data_type             : int32\n"
                       f"    comment               : 2nd column\n"
                       f"  order_index:\n"
                       f"    long_name             : spherical harmonic order m\n"
                       f"    coverage_content_type : referenceInformation\n"
                       f"    data_type             : int32\n"
                       f"    comment               : 3rd column\n"
                       f"  clm:\n"
                       f"    long_name             : Clm coefficient; cosine coefficient for degree l and order m\n"
                       f"    data_type             : double precision\n"
                       f"    comment               : 4th column\n"
                       f"  slm:\n"
                       f"    long_name             : Slm coefficient; sine coefficient for degree l and order m\n"
                       f"    data_type             : double precision\n"
                       f"    comment               : 5th column\n"
                       f"  clm_std_dev:\n"
                       f"    long_name             : standard deviation of Clm\n"
                       f"    coverage_content_type : qualityInformation\n"
                       f"    data_type             : double precision\n"
                       f"    comment               : 6th column\n"
                       f"  slm_std_dev:\n"
                       f"    long_name             : standard deviation of Slm\n"
                       f"    coverage_content_type : qualityInformation\n"
                       f"    data_type             : double precision\n"
                       f"    comment               : 7th column\n"
                       f"  epoch_begin_time:\n"
                       f"    long_name             : epoch begin of Clm, Slm coefficients\n"
                       f"    time_format           : yyyymmdd.hhmm\n"
                       f"    coverage_content_type : referenceInformation\n"
                       f"    data_type             : string\n"
                       f"    comment               : 8th column\n"
                       f"  epoch_stop_time:\n"
                       f"    long_name             : epoch stop of Clm, Slm coefficients\n"
                       f"    time_format           : yyyymmdd.hhmm\n"
                       f"    coverage_content_type : referenceInformation\n"
                       f"    data_type             : string\n"
                       f"    comment               : 9th column\n"
                       f"  solution_flags:\n"
                       f"    long_name             : Comment when present\n"
                       f"    coverage_content_type : referenceInformation\n"
                       f"    data_type             : string\n"
                       f"    comment               : 10th column\n")
            file.write(f'\n')
            file.write(f"end of header ===============================================================================\n")

            self._mainContent(file=file)


    def _mainContent(self,file):
        C10 = self.__Degree1['C10']
        C11 = self.__Degree1['C11']
        S11 = self.__Degree1['S11']
        if self.__Degree1Uncert is None:
            for i in np.arange(len(C10)):
                begindate = self.__convert_to_gnss(date_str=self.BeginDates[i])
                enddate = self.__convert_to_gnss(date_str=self.EndDates[i])
                file.write(f"GRCOF %7i %6i %+15.10E %+15.10E %+14.4E %+13.4E  {begindate}  {enddate}\n"
                           % (1, 0, C10[i], 0, 0, 0))
                file.write(f"GRCOF %7i %6i %+15.10E %+15.10E %+14.4E %+13.4E  {begindate}  {enddate}\n"
                           % (1, 1, C11[i], S11[i], 0, 0))
        else:
            UC10 = self.__Degree1Uncert['C10']
            UC11 = self.__Degree1Uncert['C11']
            US11 = self.__Degree1Uncert['S11']
            for i in np.arange(len(C10)):
                begindate = self.__convert_to_gnss(date_str=self.BeginDates[i])
                enddate = self.__convert_to_gnss(date_str=self.EndDates[i])
                file.write(f"GRCOF %7i %6i %+15.10E %+15.10E %+14.4E %+13.4E  {begindate}  {enddate}\n"
                           % (1, 0, C10[i], 0, UC10[i], 0))
                file.write(f"GRCOF %7i %6i %+15.10E %+15.10E %+14.4E %+13.4E  {begindate}  {enddate}\n"
                           % (1, 1, C11[i], S11[i], UC11[i], US11[i]))


def demo():
    begin = ['2002-01-01','2002-02-01','2002-03-01']
    end = ['2002-01-31','2002-02-28','2002-03-31']
    AAM_mass_tems = {"chi1":np.array([0.1,0.2,0.45]),
                     "chi2":np.array([0.34,0.534,0.434]),
                     "chi3":np.array([0.343,0.457,0.999])}
    AAM_motion_tems = {"chi1": np.array([1332, 2231, 45]),
                       "chi2": np.array([3412334,53423534, 43434]),
                       "chi3": np.array([234343, 243457, 12999])}

    filedir = "D:/Data/L2_low_degrees/L2_low_degrees/"

    a = FormatEAM()
    a.setFilename(Institute='CSR',GIA='ICGE6D',Filter='DDK3',Buff_width=300)
    a.setUncertainty(Degree1Uncert=Degree1_Uncert)
    a.configure(filedir=filedir,Degree1=Degree1,BeginDates=begin,EndDates=end)
    a.DegreeWrite()

if __name__ == "__main__":
    demo()