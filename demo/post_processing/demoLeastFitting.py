#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2025/6/6 15:46
# @File    : de_aliasing.py
import matplotlib.pyplot as plt
import numpy as np

from SaGEA.auxiliary.aux_tool.MathTool import MathTool


def fit_function(x, a, b, c, d):
    return a + b * x + c * np.sin(2 * np.pi * x) + d * np.cos(2 * np.pi * x)


def fit_function_linear(x, a, b):
    return a + b * x


def get_fit_signal(func, x, y):
    z = MathTool.curve_fit(func, x, y)
    # coef = z[0][:]
    # sigma = z[1][1,1]

    pass

    return func(x, *z[0][0])


def demo():
    x = np.arange(2005, 2015, 1 / 12)

    a0, b0, c0, d0, sigma = 1.2, 2.1, 4.4, 0.7, 1.8
    y0 = fit_function(x, a0, b0, c0, d0)
    y0 -= np.mean(y0)

    y_with_noise = y0 + np.random.normal(size=len(x), scale=sigma)
    y_fit = get_fit_signal(fit_function, x, y0)

    y_residual = y_with_noise - y_fit

    plt.plot(x, y_with_noise, marker=".", label="original")
    plt.plot(x, y_fit, label="fitted")
    plt.plot(x, y_residual, label="residual")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    demo()
