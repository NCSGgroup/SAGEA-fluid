import sys
import shutil
import os
from pathlib import Path
import time


class FileMover:
    def __init__(self):
        self.transferred_files = []
        self.failed_files = []

    def move_file(self, source_path, destination_dir, new_name=None, overwrite=False):
        """
        转移文件到指定目录

        Parameters:
        source_path: 源文件路径
        destination_dir: 目标目录
        new_name: 新文件名（可选）
        overwrite: 是否覆盖已存在的文件
        """
        try:
            # 检查源文件是否存在
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"源文件不存在: {source_path}")

            if not os.path.isfile(source_path):
                raise ValueError(f"路径不是文件: {source_path}")

            # 创建目标目录（如果不存在）
            Path(destination_dir).mkdir(parents=True, exist_ok=True)

            # 确定目标文件名
            if new_name:
                destination_path = os.path.join(destination_dir, new_name)
            else:
                destination_path = os.path.join(destination_dir, os.path.basename(source_path))

            # 检查目标文件是否已存在
            if os.path.exists(destination_path) and not overwrite:
                raise FileExistsError(f"目标文件已存在: {destination_path}")

            # 执行文件转移
            shutil.move(source_path, destination_path)

            # 记录成功转移的文件
            self.transferred_files.append({
                'source': source_path,
                'destination': destination_path,
                'timestamp': time.time()
            })

            print(f"✓ 成功转移: {os.path.basename(source_path)} -> {destination_path}")
            return True

        except Exception as e:
            error_info = {
                'source': source_path,
                'destination': destination_dir,
                'error': str(e),
                'timestamp': time.time()
            }
            self.failed_files.append(error_info)
            print(f"✗ 转移失败: {os.path.basename(source_path)} - {e}")
            return False

    def move_multiple_files(self, file_list, destination_dir, overwrite=False):
        """
        批量转移多个文件
        """
        results = []
        for file_info in file_list:
            if isinstance(file_info, dict):
                # 支持带新文件名的转移
                source = file_info['source']
                new_name = file_info.get('new_name')
                result = self.move_file(source, destination_dir, new_name, overwrite)
            else:
                # 简单文件路径
                result = self.move_file(file_info, destination_dir, None, overwrite)
            results.append(result)
        return results

    def get_summary(self):
        """获取转移摘要"""
        return {
            'total_transferred': len(self.transferred_files),
            'total_failed': len(self.failed_files),
            'transferred_files': self.transferred_files,
            'failed_files': self.failed_files
        }


# 使用示例
def main():
    mover = FileMover()

    # 单个文件转移
    mover.move_file(
        source_path="I:\ERA5\HD_PL/2010/20100101/sp-2010010100.nc",
        destination_dir="I:\ERA5\HD_SL/2010/20100101/",
        new_name=None,  # 可选：重命名文件
        overwrite=True  # 可选：覆盖已存在的文件
    )

    # # 批量文件转移
    # files_to_move = [
    #     {"source": "C:/Users/username/Documents/report.pdf", "new_name": "monthly_report.pdf"},
    #     {"source": "C:/Users/username/Downloads/data.xlsx", "new_name": "sales_data.xlsx"},
    #     "C:/Users/username/Downloads/presentation.pptx",  # 不重命名
    # ]
    #
    # mover.move_multiple_files(
    #     file_list=files_to_move,
    #     destination_dir="D:/Work/Backup",
    #     overwrite=False
    # )

    # 显示摘要
    summary = mover.get_summary()
    print(f"\n转移摘要:")
    print(f"成功: {summary['total_transferred']} 个文件")
    print(f"失败: {summary['total_failed']} 个文件")

def demo1():
    import pandas as pd
    import numpy as np
    mover = FileMover()
    root_path = "I:\ERA5\MAD_PL_1/"
    destination_dir = "I:\ERA5\MAD_PL/"

    date_range = pd.date_range(start='2000-01-01',end='2005-12-31',freq='MS')
    # path_last = date_range.strftime("%Y%m%d").tolist()
    file_names = date_range.strftime("%Y%m").tolist()
    year_labels = date_range.strftime("%Y").tolist()

    for i in np.arange(len(file_names)):
        mover.move_file(
            source_path=f"{root_path}/{year_labels[i]}/v_wind-{file_names[i]}.nc",
            destination_dir=f"{destination_dir}/{year_labels[i]}/",
            new_name=None,  # 可选：重命名文件
            overwrite=True  # 可选：覆盖已存在的文件
        )
    summary = mover.get_summary()
    print(f"\n转移摘要:")
    print(f"成功: {summary['total_transferred']} 个文件")
    print(f"失败: {summary['total_failed']} 个文件")


def demo2():
    import pandas as pd
    import numpy as np

    year = 2019
    mover = FileMover()
    root_path = f"I:\CRA40\DAD_PL/{year}/"
    destination_dir = f"I:\CRA40\DAD_PL/{year}/"

    date_range = pd.date_range(start=f'{year}-01-01',end=f'{year}-12-31',freq='D')
    # path_last = date_range.strftime("%Y%m%d").tolist()
    last_layer = date_range.strftime("%Y%m").tolist()
    # year_labels = date_range.strftime("%Y").tolist()
    notations = date_range.strftime("%Y%m%d").tolist()

    for i in np.arange(len(notations)):
        print(notations[i])
        mover.move_file(
            source_path=f"{root_path}/CRA40_WIV_{notations[i]}_GLB_0P50_DAY_V1_0_0.grib2",
            destination_dir=f"{destination_dir}/{last_layer[i]}/",
            new_name=None,  # 可选：重命名文件
            overwrite=True  # 可选：覆盖已存在的文件
        )
    summary = mover.get_summary()
    print(f"\n转移摘要:")
    print(f"成功: {summary['total_transferred']} 个文件")
    print(f"失败: {summary['total_failed']} 个文件")


if __name__ == "__main__":
    demo2()