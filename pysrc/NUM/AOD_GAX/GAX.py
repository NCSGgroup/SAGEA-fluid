import numpy as np
from tqdm import tqdm
from pysrc.NUM.LoadFile.LoadSH import AOD_GFZ
from pysrc.NUM.MathKit.GeoMathKit import GeoMathKit
from pysrc.NUM.Setting.Setting import AODtype
from pysrc.NUM.MathKit.FormatGAX import FormatWrite


class Generate_GAX:
    def __init__(self):
        self.InputPath = None
        self.OutputPath = None
        self.Nmax = 180
        self.daylist = None
        self.begindate = None
        self.enddate = None
        self.daylist = None
        self.TimeEpoch = ["00:00:00", "03:00:00", "06:00:00",
                          "09:00:00", "12:00:00", "15:00:00",
                          "18:00:00", "21:00:00"]
    def setDuration(self,BeginDate='2022-01-01',EndDate='2022-12-31'):
        self.begindate = BeginDate
        self.enddate = EndDate
        self.daylist = GeoMathKit.dayListByDay(begin=BeginDate,end=EndDate)
        return self.daylist
    def setTimeEpoch(self,interval=3):
        TimeEpoch = []
        for i in np.arange(0, 24, interval):
            time = '{}:00:00'.format(str(i).rjust(2, '0'))
            TimeEpoch.append(time)
        self.TimeEpoch = TimeEpoch
        return self

    def setPath(self,InputPath,OutPath):
        self.InputPath = InputPath
        self.OutputPath = OutPath
        return self

    def GAX(self,GAX='GAD'):
        CS_esmeble = []
        for day in tqdm(self.daylist):
            date = day.strftime("%Y-%m-%d")
            for time in self.TimeEpoch:
                CS = np.array(AOD_GFZ().load(self.InputPath).setType(AODtype.OBA).setTime(date, time).getCS(self.Nmax))
                CS_esmeble.append(CS)
        CS_esmeble = np.array(CS_esmeble)
        # print(CS_esmeble.shape)
        GAX_CS = np.mean(CS_esmeble,axis=0)
        GAX_CS_sigma = np.zeros_like(GAX_CS)
        # print(GAX_C.shape)
        dates = [self.begindate,self.enddate]
        fm = FormatWrite()
        fm.setGAXconfigure(filedir=self.OutputPath,GAX=GAX,date=dates,degree=self.Nmax,
                           c=GAX_CS[0],s=GAX_CS[1],sigmaC=GAX_CS_sigma[0],sigmaS=GAX_CS_sigma[1])
        fm.GAXstyle()



if __name__ == "__main__":
    a = Generate_GAX()
    a.setDuration(BeginDate='2026-09-01',EndDate='2026-09-30')
    a.setPath(InputPath=f"D:/Documents/Datasets/GRACE/GFZ/AOD1B07/",
              OutPath="D:\Documents\Datasets\GRACE\CSR\GAX\GAD_PyHawk/")
    a.GAX()



