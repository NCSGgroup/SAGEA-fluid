# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : Statistics.py
# @Create  : 10/19/26 17:02
# @Project : PySWOT
# @Software: PyCharm
"""
Description:
"""
from decimal import Decimal
import numpy as np

def count_decimal_number(num):
    s = str(num)
    if "." in s:
        return len(s.split(".")[-1])
    else:
        return 0

def count_decimal_digit(value,effective:bool=False):
    """
    This function is to count the number of decimal digit from a number.
    :param value: both list and np.ndarray are accepted!
    :param effective: False retains the 0 value in the end of a digit, True removes the 0 in the end of the digit
    :return: the decimal number of the elements of the input
    """
    is_np = isinstance(value,np.ndarray)
    out = np.asarray(value,dtype=object)
    for i, v in np.ndenumerate(out):
        s = str(v).strip()
        try:
            d = Decimal(s)
            if effective: d=d.normalize()
            out[i] = max(0,-d.as_tuple().exponent)
        except:
            out[i] = None
    return out if is_np else out.tolist()

def unique_statistic(data):
    """
    This function is to count the number of unique elements in a numpy array.
    :param data:
    :return:
    """
    if isinstance(data,np.ndarray):
        seen = []
        out = []
        for v in data.flat:
            if v not in seen:
                seen.append(v)
                out.append(v)
        return np.array(out,dtype=object)
    else:
        return list(dict.fromkeys(data))

def round_away_zero(num):
    """
    This function is to get the number which is away from 0,
    :param num: the num is the number we need to round away zero.
    :return: if num=45.34213, then return 46;
             if num=-45.3123, then return -46
    """
    x = np.asarray(num,dtype=float)
    return np.where(x>=0,np.ceil(x),np.floor(x)).astype(int)

def round_remove_decimal(num):
    """
    This function is to get the inter part of the decimal number.
    :param num: the decimal number
    :return: the inter part of the decimal number,
             for example, num=2.34244, return=2
    """
    x = np.floor(num)
    return x

def demo1():
    print(count_decimal_number(10.232))
    print(count_decimal_number(100))

def demo2():
    lst = [1.342,None,3,4,'ewdas',6,7,8,9,10,11,12,13,14,15]
    res_lst = count_decimal_digit(lst)
    print(res_lst)

def demo3():
    num_a = round_away_zero(num=-1.31232)
    print(num_a)
    num_b = round_remove_decimal(num=0.23432)
    print(num_b)

if __name__ == "__main__":
    # demo2()
    demo3()