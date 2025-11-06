import numpy as np
import scipy.io as scio

from lib.SaGEA.auxiliary.aux_tool.FileTool import FileTool

from lib.SaGEA.auxiliary.preference.EnumClasses import LLN_Data, LLN_variable, Frame


class LoveNumber:
    """
    Notice: all love numbers are given with starting index of 0.
    Ref1: Load Love numbers and Green's functions for elastic Earth models PREM, iasp91, ak135, and modified models with refined crustal structure from Crust 2.0
    Ref2: On computing the geoelastic response to a disk load
    """

    def __init__(self):

        self.__path = FileTool.get_project_dir() / 'data'/'LLN'
        self.lmax = None
        self.method = None
        self.frame = Frame.CE
        self.LLN = None
        self.config().get_Love_number()
        pass

    def config(self, lmax=1000, method=LLN_Data.Wang):
        self.lmax = lmax
        self.method = method
        return self

    def get_Love_number(self):
        """
        number_type: 'h' for vertical displacement, 'l' for horizontal displacement, 'k' for gravitational potential
        """

        if self.method == LLN_Data.PREM:
            func = self.__PREM
        elif self.method == LLN_Data.Wang:
            func = self.__Wang
        elif self.method == LLN_Data.REF:
            func = self.__REF
        elif self.method == LLN_Data.iasp91:
            func = self.__iasp91
        elif self.method == LLN_Data.ak135:
            func = self.__ak135
        elif self.method == LLN_Data.ak135hard:
            func = self.__ak135hard
        elif self.method == LLN_Data.iasp91hard:
            func = self.__iasp91hard
        elif self.method == LLN_Data.PREMhard:
            func = self.__PREMhard
        elif self.method == LLN_Data.PREMsoft:
            func = self.__PREMsoft
        else:
            print(self.method)
            raise Exception

        return func()

    def __Wang(self):
        """
        This is a more accurate way to extract Love number, which is recommended.
        Starting from degree 0
        :return:
        """

        assert self.lmax <= 13000

        path = self.__path / 'Wang.dat'

        data = np.loadtxt(path, usecols=(1, 2, 3), max_rows=13000)

        h, l, k = data[:self.lmax, 0], data[:self.lmax, 1], data[:self.lmax, 2]

        h = np.append(-0.13205, h)
        l = np.append(0, l)
        k = np.append(0, k)

        LLN = {LLN_variable.k: k, LLN_variable.h: h, LLN_variable.l: l}

        self.frame = Frame.CE
        self.LLN = LLN

        return self

    def __REF(self):
        """
        This is a more accurate way to extract Love number, which is recommended.
        Starting from degree 0
        :return:
        """

        assert self.lmax <= 40000

        path = self.__path / 'REF_6371_loading_love_numbers.mat'

        love = scio.loadmat(path)

        k = love['k_love'][:self.lmax+1]
        h = love['h_love'][:self.lmax+1]
        l = love['l_love'][:self.lmax+1]

        '''A modification: h(degee-0)= -0.132 '''
        # h[0] = -0.13205

        LLN = {LLN_variable.k: np.squeeze(k), LLN_variable.h: np.squeeze(h), LLN_variable.l: np.squeeze(l)}

        self.frame = Frame.CE
        self.LLN = LLN
        return self

    def __PREM(self):

        assert self.lmax <= 46000

        path = self.__path / 'PREM-LLNs-complete.dat'

        data = np.loadtxt(path, usecols=(1, 2, 3), max_rows=self.lmax, skiprows=1)

        h, l, k = data[:self.lmax, 0], data[:self.lmax, 1], data[:self.lmax, 2]

        # h = np.append(0, h)
        h = np.append(-0.13205, h)
        l = np.append(0, l)
        k = np.append(0, k)

        LLN = {LLN_variable.k: k, LLN_variable.h: h, LLN_variable.l: l}

        self.frame = Frame.CE
        self.LLN = LLN

        return self

    def __iasp91(self):
        assert self.lmax <= 46000

        path = self.__path / 'iasp91-LLNs-complete.dat'

        data = np.loadtxt(path, usecols=(1, 2, 3), max_rows=self.lmax, skiprows=1)

        h, l, k = data[:self.lmax, 0], data[:self.lmax, 1], data[:self.lmax, 2]

        h = np.append(-0.13205, h)
        l = np.append(0, l)
        k = np.append(0, k)

        LLN = {LLN_variable.k: k, LLN_variable.h: h, LLN_variable.l: l}

        self.frame = Frame.CE
        self.LLN = LLN

        return self

    def __ak135(self):

        assert self.lmax <= 46000

        path = self.__path / 'ak135-LLNs-complete.dat'

        data = np.loadtxt(path, usecols=(1, 2, 3), max_rows=self.lmax, skiprows=1)

        h, l, k = data[:self.lmax, 0], data[:self.lmax, 1], data[:self.lmax, 2]

        h = np.append(-0.13205, h)
        l = np.append(0, l)
        k = np.append(0, k)

        LLN = {LLN_variable.k: k, LLN_variable.h: h, LLN_variable.l: l}

        self.frame = Frame.CE
        self.LLN = LLN

        return self

    def __ak135hard(self):

        assert self.lmax <= 46000

        path = self.__path / 'ak135hard-LLNs-complete.dat'

        data = np.loadtxt(path, usecols=(1, 2, 3), max_rows=self.lmax, skiprows=1)

        h, l, k = data[:self.lmax, 0], data[:self.lmax, 1], data[:self.lmax, 2]

        h = np.append(-0.13205, h)
        l = np.append(0, l)
        k = np.append(0, k)

        LLN = {LLN_variable.k: k, LLN_variable.h: h, LLN_variable.l: l}

        self.frame = Frame.CE
        self.LLN = LLN

        return self

    def __iasp91hard(self):

        assert self.lmax <= 46000

        path = self.__path / 'iasp91hard-LLNs-complete.dat'

        data = np.loadtxt(path, usecols=(1, 2, 3), max_rows=self.lmax, skiprows=1)

        h, l, k = data[:self.lmax, 0], data[:self.lmax, 1], data[:self.lmax, 2]

        h = np.append(-0.13205, h)
        l = np.append(0, l)
        k = np.append(0, k)

        LLN = {LLN_variable.k: k, LLN_variable.h: h, LLN_variable.l: l}

        self.frame = Frame.CE
        self.LLN = LLN

        return self

    def __PREMhard(self):

        assert self.lmax <= 46000

        path = self.__path / 'PREMhard-LLNs-complete.dat'

        data = np.loadtxt(path, usecols=(1, 2, 3), max_rows=self.lmax, skiprows=1)

        h, l, k = data[:self.lmax, 0], data[:self.lmax, 1], data[:self.lmax, 2]

        h = np.append(-0.13205, h)
        l = np.append(0, l)
        k = np.append(0, k)

        LLN = {LLN_variable.k: k, LLN_variable.h: h, LLN_variable.l: l}

        self.frame = Frame.CE
        self.LLN = LLN

        return self

    def __PREMsoft(self):

        assert self.lmax <= 46000

        path = self.__path / 'PREMsoft-LLNs-complete.dat'

        data = np.loadtxt(path, usecols=(1, 2, 3), max_rows=self.lmax, skiprows=1)

        h, l, k = data[:self.lmax, 0], data[:self.lmax, 1], data[:self.lmax, 2]

        h = np.append(-0.13205, h)
        l = np.append(0, l)
        k = np.append(0, k)

        LLN = {LLN_variable.k: k, LLN_variable.h: h, LLN_variable.l: l}

        self.frame = Frame.CE
        self.LLN = LLN

        return self

    def convert(self, target=Frame.CM):
        if self.frame == target:
            return self
        else:
            self.frame = target
            pass

        if target == Frame.CM:
            # PREM
            self.LLN[LLN_variable.k][1] = -1.0
            self.LLN[LLN_variable.h][1] = -1.2843084
            self.LLN[LLN_variable.l][1] = -0.89695547

        if target == Frame.CF:
            # PREM
            # self.LLN[LLN_variable.k][1] = 0.02607313
            self.LLN[LLN_variable.k][1] = 0.021
            self.LLN[LLN_variable.h][1] = -0.25823532
            self.LLN[LLN_variable.l][1] = 0.12911766

        return self


def demo():

    lln = LoveNumber().config(lmax=300, method=LLN_Data.PREMsoft).get_Love_number()
    print(lln.LLN[LLN_variable.h])

def demo1():
    ln = LoveNumber().config(lmax=60,method=LLN_Data.Wang).get_Love_number()
    ln.convert(target=Frame.CF)
    print(ln.LLN[LLN_variable.k])


if __name__ == '__main__':
    demo1()
