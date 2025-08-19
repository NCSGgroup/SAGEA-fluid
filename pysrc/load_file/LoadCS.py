import numpy as np
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from pysrc.load_file.DataClass import SHC
import re
import datetime

def match_dates_from_filename(filename):
    match_flag = False
    this_date_begin, this_date_end = None, None

    '''date format: yyyymmdd-yyyymmdd or yyyy-mm-dd-yyyy-mm-dd'''
    if not match_flag:
        date_begin_end_pattern = r"(\d{4})-?(\d{2})-?(\d{2})-(\d{4})-?(\d{2})-?(\d{2})"
        date_begin_end_searched = re.search(date_begin_end_pattern, filename)

        if date_begin_end_searched is not None:
            date_begin_end = date_begin_end_searched.groups()
            this_date_begin = datetime.date(*list(map(int, date_begin_end[:3])))
            this_date_end = datetime.date(*list(map(int, date_begin_end[3:])))

            match_flag = True

    '''date format: yyyyddd-yyyyddd'''
    if not match_flag:
        date_begin_end_pattern = r"(\d{4})(\d{3})-(\d{4})(\d{3})"
        date_begin_end_searched = re.search(date_begin_end_pattern, filename)

        if date_begin_end_searched is not None:
            date_begin_end = date_begin_end_searched.groups()
            this_date_begin = datetime.date(int(date_begin_end[0]), 1, 1) + datetime.timedelta(
                days=int(date_begin_end[1]) - 1)
            this_date_end = datetime.date(int(date_begin_end[2]), 1, 1) + datetime.timedelta(
                days=int(date_begin_end[3]) - 1)

            match_flag = True

    '''date format: yyyymm'''
    if not match_flag:
        date_begin_end_pattern = r"_(\d{4})(\d{2})_"
        date_begin_end_searched = re.search(date_begin_end_pattern, filename)

        if date_begin_end_searched is not None:
            year_month = date_begin_end_searched.groups()
            year = int(year_month[0])
            month = int(year_month[1])

            this_date_begin = datetime.date(int(year), month, 1)
            if month < 12:
                this_date_end = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
            elif month == 12:
                this_date_end = datetime.date(year + 1, 1, 1) + datetime.timedelta(days=1)
            else:
                assert False

            match_flag = True

    assert match_flag, f"illegal date format in filename: {filename}"

    return this_date_begin, this_date_end

class LoadCS:
    def __init__(self):
        self.lmax = 60
        pass
    def setMaxDegree(self,lmax):
        self.lmax = lmax
        return self

    def get_flag(self,file,lmcs_in_queue:np.ndarray):
        index = 0
        with open(file,"r") as f:
            for i in f.readlines():
                if i is not None:
                    a = i.split()
                    if len(a) != 0 and len(a) != 1:
                        if a[lmcs_in_queue[0]] == '0' and a[lmcs_in_queue[1]] == '0':
                            # print(index)
                            break
                        else:
                            index +=1
                    else:
                        index+=1
                else:
                    index+=1
        return index

    def get_CS(self,*filepath,begin_date=None,end_date=None,lmcs_in_queue:np.ndarray,get_dates=False):

        C, S = np.zeros((self.lmax + 1, self.lmax + 1)), np.zeros((self.lmax + 1, self.lmax + 1))

        if len(filepath) == 1:
            assert filepath[0].exists(), f"{filepath[0]} does not exist!"
            if filepath[0].is_file():
                with open(filepath[0], "r") as f:
                    for i in f.readlines()[self.get_flag(file=filepath[0],lmcs_in_queue=lmcs_in_queue):]:
                        a = i.split()
                        C[int(a[lmcs_in_queue[0]]), int(a[lmcs_in_queue[1]])] = a[lmcs_in_queue[2]]
                        S[int(a[lmcs_in_queue[0]]), int(a[lmcs_in_queue[1]])] = a[lmcs_in_queue[3]]

                if get_dates:
                    this_date_begin, this_date_end = match_dates_from_filename(filepath[0].name)
                    SH = SHC(c=C, s=S)
                    return SH,[this_date_begin],[this_date_end]
                else:
                    SH = SHC(c=C, s=S)
                    return SH

            elif filepath[0].is_dir():
                file_list = FileTool.get_files_in_dir(filepath[0], sub=True)
                file_list.sort()

                files_to_load = []
                for i in range(len(file_list)):
                    this_begin_date, this_end_date = match_dates_from_filename(file_list[i].name)
                    if this_begin_date >= begin_date and this_end_date <= end_date:
                        files_to_load.append(file_list[i])

                return self.get_CS(*files_to_load,begin_date=begin_date,end_date=end_date,lmcs_in_queue=lmcs_in_queue,get_dates=get_dates)
        else:
            shc = None
            dates_begin,dates_end = [],[]

            for i in np.arange(len(filepath)):
                load = self.get_CS(filepath[i],begin_date=begin_date,end_date=end_date,lmcs_in_queue=lmcs_in_queue,get_dates=get_dates)
                if type(load) is tuple:
                    assert len(load) in (1, 3)
                    load_shc = load[0]
                else:
                    load_shc = load

                if shc is None:
                    shc = load_shc
                else:
                    shc.append(load_shc)
                if get_dates:
                    assert len(load)==3
                    d_begin,d_end = load[1],load[2]
                    dates_begin.append(d_begin[0])
                    dates_end.append(d_end[0])
            if get_dates:
                return shc,dates_begin,dates_end
            else:
                return shc

def demo():
    from datetime import date
    # filepath = FileTool.get_project_dir(
    #     "D:\Cheung\PyZWH\data/ref_sealevel\SLFsh_coefficients\GFZOP\CM\WOUTrotation/SLF-2_2003001-2003031_GRAC_GFZOP_BA01_0600")
    filepath = FileTool.get_project_dir(
        "I:\GFZ\GSM\RL060\BA01/")
    begin_date, end_date = date(2002, 4, 1), date(2016, 8, 31)
    a = LoadCS().get_CS(filepath,begin_date=begin_date, end_date=end_date,lmcs_in_queue=np.array([1,2,3,4]))

    print(a.value.shape)

if __name__ == '__main__':
    demo()