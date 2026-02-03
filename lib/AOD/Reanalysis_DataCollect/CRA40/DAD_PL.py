import os

import numpy as np
import requests
from pathlib import Path
import time


def download_grib_files(url_file_path, download_dir, max_retries=3, delay=1):
    """
    从包含URL列表的文件下载GRIB2数据

    参数:
    url_file_path: 包含URL列表的文本文件路径
    download_dir: 下载文件保存的目录
    max_retries: 最大重试次数
    delay: 重试之间的延迟(秒)
    """

    # 创建下载目录（如果不存在）
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # 读取URL文件
    try:
        with open(url_file_path, 'r', encoding='utf-8') as file:
            urls = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"错误: 文件 {url_file_path} 不存在")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    print(f"找到 {len(urls)} 个URL需要下载")

    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # 创建会话
    session = requests.Session()

    # 下载每个文件
    for i, url in enumerate(urls, 1):
        # 从URL提取文件名
        filename = url.split('/')[-1].split('?')[0]  # 移除查询参数
        file_path = os.path.join(download_dir, filename)

        print(f"正在下载文件 {i}/{len(urls)}: {filename}")

        # 如果文件已存在，跳过下载
        if os.path.exists(file_path):
            print(f"文件已存在，跳过: {filename}")
            continue

        # 尝试下载（带重试机制）
        for attempt in range(max_retries):
            try:
                response = session.get(url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()

                # 保存文件
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                print(f"成功下载: {filename} ({os.path.getsize(file_path)} bytes)")
                break  # 成功下载，跳出重试循环

            except requests.exceptions.RequestException as e:
                print(f"下载尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {delay} 秒后重试...")
                    time.sleep(delay)
                else:
                    print(f"下载失败: {filename}")

        # 文件间短暂延迟，避免请求过于频繁
        time.sleep(0.5)
    print("下载完成！")

def Data_classification():
    import pandas as pd
    import numpy as np
    from pysrc.File_Processing.FileMove import FileMover

    year = 2015
    mover = FileMover()
    root_path = f"I:\CRA40\DAD_PL/{year}/"
    destination_dir = f"I:\CRA40\DAD_PL/{year}/"
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    last_layer = date_range.strftime("%Y%m").tolist()
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

def main():
    # url_file_path = f"I:/CRA40/2022S.txt"  # 替换为您的URL文件路径
    # download_dir = f"I:/CRA40/DAD_SL/2022/"  # 替换为您想要的下载目录
    # download_grib_files(url_file_path, download_dir)

    # for year in np.arange(2006, 2011):
    #     url_file_path = f"I:/CRA40/{year}vD.txt"  # 替换为您的URL文件路径
    #     download_dir = f"I:/CRA40/DAD_PL/{year}/"  # 替换为您想要的下载目录
    #     download_grib_files(url_file_path, download_dir)
    #
    # url_file_path = f"I:/CRA40/2005vD_12.txt"  # 替换为您的URL文件路径
    # download_dir = f"I:/CRA40/DAD_PL/2005/"  # 替换为您想要的下载目录
    # download_grib_files(url_file_path, download_dir)

    url_file_path = f"I:/CRA40/2011vD_12.txt"  # 替换为您的URL文件路径
    download_dir = f"I:/CRA40/DAD_PL/2011/"  # 替换为您想要的下载目录
    download_grib_files(url_file_path, download_dir)

    # for year in np.arange(2022,2025):
    #     url_file_path = f"I:/CRA40/{year}SG.txt"  # 替换为您的URL文件路径
    #     download_dir = f"I:/CRA40/DAD_SL/{year}/"  # 替换为您想要的下载目录
    #     download_grib_files(url_file_path, download_dir)


if __name__ == "__main__":
    # main()
    Data_classification()