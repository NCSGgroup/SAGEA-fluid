import pygrib

# 定义输入文件和输出文件
date = '2008-12-18'
sstr = date.split('-')
year = sstr[0]
day = sstr[0]+sstr[1]+sstr[2]
file1 = f'H:/CRA40/{year}/{day}/CRA40_TEM_{day}06_GLB_0P50_HOUR_V1_0_0.grib2'
# file2 = f'H:/CRA40/2008/20081217/CRA40_TEM_2008121718_GLB_0P50_HOUR_V1_0_0.grib2'
file2 = f'H:/CRA40/{year}/{day}/CRA40_TEM_{day}12_GLB_0P50_HOUR_V1_0_0.grib2'
output_file = f'H:/CRA40/{year}/{day}/CRA40_TEM_{day}00_GLB_0P50_HOUR_V1_0_0.grib2'

# 读取两个 .grib2 文件
grbs1 = pygrib.open(file1)
grbs2 = pygrib.open(file2)

# 创建一个空的grib消息列表，用于保存平均后的grib记录
output_gribs = []

for grb1, grb2 in zip(grbs1, grbs2):
    if grb1.shortName != grb2.shortName:
        raise ValueError(f"Variable mismatch: {grb1.shortName} and {grb2.shortName}")

    # 对两个grib记录的数据求平均
    data_avg = (grb1.values + grb2.values) / 2.0

    # 克隆第一个grib记录并替换其中的数据为平均值
    grb1.values = data_avg
    output_gribs.append(grb1.tostring())

# 将平均后的grib记录写入一个新的.grib2文件
with open(output_file, 'wb') as out:
    for grib_message in output_gribs:
        out.write(grib_message)

# 关闭grib文件
grbs1.close()
grbs2.close()

print(f"平均后的GRIB2文件已保存为 {output_file}")