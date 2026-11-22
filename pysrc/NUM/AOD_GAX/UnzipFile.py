import os
import zipfile
import tarfile
import gzip
import shutil


def extract_all_compressed_files(target_dir):
    """
    解压指定目录下所有.zip、.tar.gz、.gz、.asc.gz格式的压缩文件，解压后删除原压缩文件

    参数:
        target_dir (str): 目标目录路径
    """
    # 验证目标目录是否存在
    if not os.path.isdir(target_dir):
        print(f"错误：目录 '{target_dir}' 不存在！")
        return

    # 遍历目标目录下的所有文件
    for filename in os.listdir(target_dir):
        # 拼接完整文件路径
        file_path = os.path.join(target_dir, filename)

        # 跳过目录（只处理文件）
        if os.path.isdir(file_path):
            continue

        # 处理.zip格式文件
        if filename.lower().endswith('.zip'):
            try:
                print(f"正在解压: {filename}")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                os.remove(file_path)
                print(f"解压完成，已删除原文件: {filename}")
            except Exception as e:
                print(f"解压 {filename} 失败: {str(e)}")

        # 处理.tar.gz（tar打包+gz压缩）
        elif filename.lower().endswith('.tar.gz'):
            try:
                print(f"正在解压: {filename}")
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(target_dir)
                os.remove(file_path)
                print(f"解压完成，已删除原文件: {filename}")
            except tarfile.ReadError:
                print(f"错误：{filename} 不是有效的tar.gz格式文件")
            except Exception as e:
                print(f"解压 {filename} 失败: {str(e)}")

        # 处理纯gz压缩文件（.gz/.asc.gz，单个文件压缩）
        elif filename.lower().endswith(('.gz', '.asc.gz')):
            try:
                print(f"正在解压: {filename}")
                # 解压后的文件名：去掉.gz后缀（如AOD1B_2009-01-01_X_07.asc.gz → AOD1B_2009-01-01_X_07.asc）
                extract_filename = filename[:-3] if filename.lower().endswith('.gz') else filename[:-7]
                extract_path = os.path.join(target_dir, extract_filename)

                # 用gzip解压单个文件
                with gzip.open(file_path, 'rb') as f_in, open(extract_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

                # 解压成功后删除原压缩文件
                os.remove(file_path)
                print(f"解压完成，已删除原文件: {filename}")
            except Exception as e:
                print(f"解压 {filename} 失败: {str(e)}")

def demo(year=2020):
    # 请在这里修改为你要处理的目录路径（绝对路径/相对路径都可以）
    # 示例：Windows路径 r"C:\Users\YourName\Desktop\test"
    # 示例：Linux/Mac路径 "/home/yourname/Desktop/test"
    # target_directory = f"D:/PyCode/AOD-ERA5/data/GFZ/{year}/"
    target_directory = f"D:\Documents\Datasets\GRACE\GFZ\AOD1B07/{year}/"

    # 调用解压函数
    extract_all_compressed_files(target_directory)
    print("所有压缩文件处理完成！")

# 主程序入口
if __name__ == "__main__":
    demo(year=2026)
    # demo(year=2008)
    # demo(year=2007)
    # demo(year=2005)