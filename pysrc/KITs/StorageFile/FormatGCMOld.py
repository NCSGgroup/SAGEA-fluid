import os
import numpy as np


class FormatGCM:
    def __init__(self, GCM: dict, BeginDate: list, EndDate: list):
        """
        初始化GRACE文件生成器
        :param GCM: 时间序列字典，必须包含 'C10'/'C11'/'S11' 三个键，值为等长列表
        :param BeginDate: 开始时间列表，长度与GCM内序列一致
        :param EndDate: 结束时间列表，长度与GCM内序列一致
        """
        # 外部传入的核心数据
        self.GCM = GCM
        self.BeginDate = BeginDate
        self.EndDate = EndDate

        # 自动校验数据合法性（避免运行报错）
        self._validate_data()

    def _validate_data(self):
        """私有方法：校验输入数据是否符合要求"""
        required_keys = {'C10', 'C11', 'S11'}
        # 检查GCM是否包含必需的键
        if not required_keys.issubset(self.GCM.keys()):
            raise ValueError(f"GCM字典必须包含 {required_keys} 三个键！")

        # 检查所有序列长度是否一致
        data_length = len(self.GCM['C10'])
        if (len(self.GCM['C11']) != data_length or
                len(self.GCM['S11']) != data_length or
                len(self.BeginDate) != data_length or
                len(self.EndDate) != data_length):
            raise ValueError("GCM内序列、BeginDate、EndDate长度必须完全一致！")

    def GCMstyle(self, output_file: str):
        """
        生成符合格式的.txt文件
        :param output_file: 输出文件路径（如 'grace_output.txt'）
        :param custom_name: 第一行自定义名称，默认 CUSTOM_DATA
        """
        with open(output_file, "w", encoding="utf-8") as f:

            f.write(f"GRACE Degree-1 Coefficients\n")
            f.write(f"GIA ICE6GD, DDK3, Buffer300\n")

            f.write("  variables:\n")
            f.write("    record_key:\n")
            f.write("      key_name              : GRCOF2\n")
            f.write("      long_name             : Earth Gravity Spherical Harmonic Model Format Type\n")
            f.write("      coverage_content_type : referenceInformation\n")
            f.write("      data_type             : string\n")
            f.write("      comment               : 1st column\n")
            f.write("    degree_index:\n")
            f.write("      long_name             : spherical harmonic degree l\n")
            f.write("      coverage_content_type : referenceInformation\n")
            f.write("      data_type             : int32\n")
            f.write("      comment               : 2nd column\n")
            f.write("    order_index:\n")
            f.write("      long_name             : spherical harmonic degree m\n")
            f.write("      coverage_content_type : referenceInformation\n")
            f.write("      data_type             : int32\n")
            f.write("      comment               : 3rd column\n")
            f.write("    clm:\n")
            f.write("      long_name             : Clm coefficient; cosine coefficient for degree l and order m\n")
            f.write("      data_type             : double precision\n")
            f.write("      comment               : 4th column\n")
            f.write("    slm:\n")
            f.write("      long_name             : Slm coefficient; sine coefficient for degree l and order m\n")
            f.write("      data_type             : double precision\n")
            f.write("      comment               : 5th column\n")
            f.write("    epoch_begin_time: \n")
            f.write("      long_name             : epoch begin of Clm, Slm coefficients\n")
            f.write("      time_format           : yyyymmdd.hhmm\n")
            f.write("      coverage_content_type : referenceInformation\n")
            f.write("      data_type             : string\n")
            f.write("      comment               : 6th column\n")
            f.write("    epoch_stop_time: \n")
            f.write("      long_name             : epoch stop of Clm, Slm coefficients\n")
            f.write("      time_format           : yyyymmdd.hhmm\n")
            f.write("      coverage_content_type : referenceInformation\n")
            f.write("      data_type             : string\n")
            f.write("      comment               : 7th column\n")
            f.write("end of header ==================================================\n")

            # 循环遍历所有时间节点
            n_times = len(self.GCM['C10'])
            for i in range(n_times):
                # 写入 C10 行
                f.write(f"GRCOF2    1   0  {self.GCM['C10'][i]}  0 {self.BeginDate[i]}  {self.EndDate[i]}\n")
                # 写入 C11/S11 行
                f.write(f"GRCOF2    1   1  {self.GCM['C11'][i]}  {self.GCM['S11'][i]}  {self.BeginDate[i]}  {self.EndDate[i]}\n")

        print(f"✅ 文件已成功生成：{output_file}")

def demo():
    from datetime import date
    from pysrc.KITs.LoadFile.LoadICGEM import load_SHC
    from lib.SaGEA.auxiliary.aux_tool.FileTool import FileTool
    load_path_csr = "D:/Data/GRACE/CSR/GSM/Consistent/"
    res, lmax, buffer_width, DDK = 0.5, 90, 300, 3
    gsm_dir, gsm_key = FileTool.get_project_dir(load_path_csr), 'gfc'
    begin_date, end_date = date(2002, 1, 1), date(2024, 4, 30)
    shc, dates_begin, dates_end = load_SHC(gsm_dir, key=gsm_key, lmax=lmax, begin_date=begin_date, end_date=end_date,
                                           get_dates=True, )
    print(len(dates_begin),len(dates_end))
    gcm_path = "D:/PyCode/SAGEA-fluid/result/GCM_Consistency/V3/DATA/Degree1_Ellipsoid/"

    gcm_data = np.load(f"{gcm_path}/CSR_ICE6GD_DDK3_BUF300.npz")
    print(len(gcm_data['C10']))
    print(type(gcm_data))

    GCM_Storage = FormatGCM(GCM=gcm_data,BeginDate=dates_begin,EndDate=dates_end)
    GCM_Storage.GCMstyle(output_file="D:/PyCode/SAGEA-fluid/result/GCM_Consistency/GeocenterMotion/gcm.txt")

if __name__ == "__main__":
    demo()




