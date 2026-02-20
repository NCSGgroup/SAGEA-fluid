import re

import numpy as np

from lib.SaGEA.auxiliary.aux_tool.FileTool import FileTool


def load_EAM(filepath):
    with open(filepath) as f:
        txt = f.read()
    mass_chi1,mass_chi2,mass_chi3 =[],[],[]
    motion_chi1,motion_chi2,motion_chi3 = [],[],[]
    date_range =[]
    pat_data = r"\s*^\d{4}.*"
    data = re.findall(pat_data,txt,re.M)
    for i in data:
        line = i.split()
        # print(line)
        date_time = line[0]+line[1]+line[2]+line[3]
        date_range.append(date_time)
        mass_chi1.append(np.float64(line[5]))
        mass_chi2.append(np.float64(line[6]))
        mass_chi3.append(np.float64(line[7]))

        motion_chi1.append(np.float64(line[8]))
        motion_chi2.append(np.float64(line[9]))
        motion_chi3.append(np.float64(line[10]))

    mass_chi1 = np.array(mass_chi1)
    mass_chi2 = np.array(mass_chi2)
    mass_chi3 = np.array(mass_chi3)


    motion_chi1 = np.array(motion_chi1)
    motion_chi2 = np.array(motion_chi2)
    motion_chi3 = np.array(motion_chi3)

    EAM = {"date_range":date_range,
           "mass_chi1":mass_chi1,"mass_chi2":mass_chi2,"mass_chi3":mass_chi3,
           "motion_chi1":motion_chi1,"motion_chi2":motion_chi2,"motion_chi3":motion_chi3}
    return EAM

def load_SLAM(filepath):
    with open(filepath) as f:
        txt = f.read()
    mass_chi1,mass_chi2,mass_chi3 =[],[],[]
    date_range =[]
    pat_data = r"\s*^\d{4}.*"
    data = re.findall(pat_data,txt,re.M)
    for i in data:
        line = i.split()
        # print(line)
        date_time = line[0]+line[1]+line[2]+line[3]
        date_range.append(date_time)
        mass_chi1.append(np.float64(line[5]))
        mass_chi2.append(np.float64(line[6]))
        mass_chi3.append(np.float64(line[7]))



    mass_chi1 = np.array(mass_chi1)
    mass_chi2 = np.array(mass_chi2)
    mass_chi3 = np.array(mass_chi3)



    EAM = {"date_range":date_range,
           "mass_chi1":mass_chi1,"mass_chi2":mass_chi2,"mass_chi3":mass_chi3}
    return EAM





def demo():
    filepath = FileTool.get_project_dir('I:\GFZ\EAM\AAM/ESMGFZ_AAM_v1.0_03h_2010.asc')
    a = load_EAM(filepath)
    print(a['mass_chi2'])

if __name__ == '__main__':
    demo()