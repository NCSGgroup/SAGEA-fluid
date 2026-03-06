import datetime
import time as ti
import json
import os
import requests
import shutil
import numpy as np

class Download():
    def __init__(self):
        self.save_path = None
        self.source_path = None
        self.destin_path = None
        self.param = None
        self.daylist = None
        self.timeepoch = ['00:00','03:00','06:00','09:00',
                          '12:00','15:00','18:00','21:00']

        pass
    def setPath(self,path='D:/PyProject/Data/model level 0.25/'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_path = path
        return self
    def setMovePath(self,source_path,destin_path):
        self.source_path = source_path
        self.destin_path = destin_path
        return self

    def setTime(self,begin='2020-01-01',end='2020-01-05'):
        daylist = []
        begin_date = datetime.datetime.strptime(begin,'%Y-%m-%d')
        end_date = datetime.datetime.strptime(end,'%Y-%m-%d')

        while begin_date <= end_date:
            date_str = begin_date
            daylist.append(date_str)
            begin_date += datetime.timedelta(days=1)
        self.daylist = daylist
        return self
    def GetCRA40(self,InputPath):
        with open(InputPath, 'r') as file:
            lines = file.readlines()

        # 遍历每一行并下载文件
        for url in lines:
            url = url.strip()  # 去掉换行符或空白符
            if url:  # 检查url是否为空
                # 从URL中提取文件名
                filename = url.split('/')[-1].split('?')[0]
                # 下载文件
                response = requests.get(url)
                if response.status_code == 200:
                    # 将文件保存到指定路径
                    for date in self.daylist:
                        date = date.strftime('%Y-%m-%d')
                        ss = date.split('-')
                        sstr = ss[0]+ss[1]+ss[2]
                        root_path = os.path.join(self.save_path,sstr)
                        if not os.path.exists(root_path):
                            os.makedirs(root_path)
                        if sstr in filename:
                            file_path = os.path.join(root_path, filename)
                            with open(file_path, 'wb') as f:
                                f.write(response.content)
                            print(f"Downloaded: {filename} in {root_path}")
                else:
                    print(f"Failed to download: {filename}, status code: {response.status_code}")

    def GetRightPath(self):
        if not os.path.exists(self.destin_path):
            os.makedirs(self.destin_path)
        for date in self.daylist:
            date = date.strftime('%Y-%m-%d')
            ss = date.split('-')
            day = ss[0]+ss[1]+ss[2]
            year = ss[0]
            destination_directory = self.destin_path+'/'+year+'/'+day+'/'
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            for root,dirs,files in os.walk(self.source_path):
                for file in files:
                    if day in file:
                        source_file_path = os.path.join(root, file)
                        # 构造目标文件路径
                        destination_file_path = os.path.join(destination_directory, file)

                        # 移动文件到目标目录
                        shutil.move(source_file_path, destination_file_path)
                        print(f"Moved: {source_file_path} -> {destination_file_path}")



    @staticmethod
    def JsonWrite(isWrite=True):

        config = {'GeoHeight':'model level',
                  'SavePath':'H:/ERA5/model level/',
                  'Parameter':'TEM',
                  'BeginDate':'2010-01-01',
                  'EndDate':'2010-01-02',
                  'grid':0.25,
                  'Time_interval':3,
                  'Time_BeginEpoch':0,
                  'Time_EndEpoch':24}

        if isWrite:
            with open('DownloadCRA40.json','w') as f:
                f.write(json.dumps(config,indent=9))
        return config
def demo():
    a = Download()
    a.setPath('H:/CRA/2024/')
    a.setTime(begin='2024-01-01',end='2022-02-01')
    a.GetCRA40(InputPath='H:/CRA/202401.txt')
    # a.setTime(begin='2021-12-01', end='2021-12-31')
    # a.GetCRA40(InputPath='D:/CRA/202112.txt')

def demo1():
    a = Download()
    a.setMovePath(source_path='D:/CRA/2021/202101/', destin_path='H:/CRA\data/CRA.grib2/')
    a.setTime(begin='2020-12-31', end='2020-12-31')
    a.GetRightPath()



if __name__ == '__main__':
    demo()
