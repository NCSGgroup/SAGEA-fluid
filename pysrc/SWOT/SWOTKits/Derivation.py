# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : Derivation.py
# @Create  : 9/29/2026 6:35 PM
# @Project : PySWOT
# @Software: PyCharm
"""
Description:
"""

import numpy as np

def derivative_center(data, axis, spacing):
    """只在两侧邻点都有效时使用中心差分，不跨越缺测点。"""
    result = np.full(data.shape, np.nan, dtype=np.float64)

    center = [slice(None)] * data.ndim
    before = [slice(None)] * data.ndim
    after  = [slice(None)] * data.ndim

    center[axis] = slice(1, -1)
    before[axis] = slice(None, -2)
    after[axis]  = slice(2, None)

    center = tuple(center)
    before = tuple(before)
    after = tuple(after)

    usable = (
        np.isfinite(data[center])
        & np.isfinite(data[before])
        & np.isfinite(data[after])
    )

    values = (data[after] - data[before]) / (2 * spacing)
    result[center] = np.where(usable, values, np.nan)

    return result

def derivative_adjacent(data,spacing,pNo=1):
    dA_along = np.full(data.shape, np.nan)
    dA_cross = np.full(data.shape, np.nan)
    # 竖直方向：沿 axis=0，相邻行差分
    # dA_along[:-1, :] = (data[1:, :] - data[:-1, :]) / spacing
    # # 最后一行：使用后向差分
    # dA_along[-1, :] = (data[-1, :] - data[-2, :]) / spacing
    # # 水平方向：沿 axis=1，相邻列差分
    # dA_cross[:, :-1] = (data[:, 1:] - data[:, :-1]) / spacing
    # # 最后一列：使用后向差分
    # dA_cross[:, -1] = (data[:, -1] - data[:, -2]) / spacing
    if pNo % 2 == 1:
        dA_along[1:,:] = (data[1:,:]-data[:-1,:])/spacing
        dA_cross[:,1:] = (data[:,1:]-data[:,:-1])/spacing
    else:
        dA_along[1:,:] = (data[:-1,:]-data[1:,:])/spacing
        dA_cross[:,1:] = (data[:,:-1]-data[:,1:])/spacing
    return dA_along,dA_cross

def derivative_2D(data,spacing,pNo=1):
    dfdy = np.full(data.shape, np.nan)
    dfdx = np.full(data.shape, np.nan)

    if pNo % 2 == 1:
        dfdy[1:,:] = (data[1:,:]-data[:-1,:])/spacing   # axis=0, its row calculation
        dfdx[:,1:] = (data[:,1:]-data[:,:-1])/spacing   # axis=1, its column calculation
    else:
        dfdy[1:,:] = (data[:-1,:]-data[1:,:])/spacing   # axis=0, its row calculation
        dfdx[:,1:] = (data[:,:-1]-data[:,1:])/spacing   # axis=1, its column calculation
    return dfdy,dfdx
def derivative_1D(data,spacing,pNo=1):
    df = np.full(data.shape,np.nan)
    # print(f"data shape and df shape:{data.shape,df.shape}")
    if pNo % 2 == 1:
        df[1:] = (data[1:]-data[:-1])/spacing
    else:
        df[1:] = (data[:-1]-data[1:])/spacing
    return df
    return df


def demo():
    res = 2
    fx = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
    df = derivative_1D(data=fx,spacing=2)
    print(df)

if __name__ == "__main__":
    # Track_graident()
    demo()
