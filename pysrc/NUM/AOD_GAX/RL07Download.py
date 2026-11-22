import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

# ========== 在这里改你要的年份 ==========
# TARGET_YEARS = ["2020", "2021", "2022"]
TARGET_YEARS = ["2026"]
# ======================================

BASE_URL = "https://isdc-data.gfz.de/grace/Level-1B/GFZ/AOD/RL07/"
# SAVE_ROOT = "D:/PyCode/AOD-ERA5/data/GFZ/"
SAVE_ROOT = "D:\Documents\Datasets\GRACE\GFZ\AOD1B07/"

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"已存在：{save_path}")
        return True
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(1024*1024):
                f.write(chunk)
        print(f"下载完成：{save_path}")
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f"失败：{url} | {e}")
        return False

def get_files_in_dir(dir_url):
    resp = requests.get(dir_url, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")
    files = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if href and not href.endswith("/") and not href.startswith("?"):
            files.append(urljoin(dir_url, href))
    return files

if __name__ == "__main__":
    os.makedirs(SAVE_ROOT, exist_ok=True)
    for year in TARGET_YEARS:
        year_dir = os.path.join(SAVE_ROOT, year)
        os.makedirs(year_dir, exist_ok=True)
        year_url = BASE_URL + year + "/"
        print(f"\n===== 正在下载：{year} =====")
        file_urls = get_files_in_dir(year_url)
        for url in file_urls:
            fname = url.split("/")[-1]
            download_file(url, os.path.join(year_dir, fname))