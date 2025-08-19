import pickle

import numpy as np
from scipy.special import assoc_laguerre,factorial

def normalized_assoc_legendre(l,m,alpha):
    if abs(m) > 1:
        return 0.0
    Plm = assoc_laguerre(l,m)[0]
    fact_ratio = factorial(l-np.abs(m),exact=True)/factorial(l+np.abs(m),exact=True)

    normalization = np.sqrt((2*l+1)/(4*np.pi)*fact_ratio)

    return normalization*Plm


def Legendre_polynomial(alpha, lmax: int):
    """
    get legendre polynomial up to degree/order lmax in Lat. This is traditional unnormalized Legendre function.
    :param alpha: ndarray, relative angle between two spherical nodes, unit[degree]
    :param lmax: int, max degree
    :return: 3d-ndarray, indexes stand for (len(alpha), degree l)
    """
    z = np.cos(np.deg2rad(alpha))

    P0 = np.ones_like(z)
    P1 = np.ones_like(z) * z
    P = [P0, P1]
    for l in range(2, lmax + 1):
        Pl = (2 * l - 1) * z * P[-1] / l - (l - 1) * P[-2] / l
        P.append(Pl)

    return np.array(P).T


def Legendre_4pi(lat, lmax: int, option=0):
    """
    get associated legendre up to degree/order lmax in Lat. This is 4pi normalization.
    :param lat: ndarray, co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
    :param lmax: int, max degree
    :param option:
    :return: 3d-ndarray, indexes stand for (co-lat[rad], degree l, order m)
    """

    if option != 0:
        lat = (90. - lat) / 180. * np.pi

    if type(lat) is np.ndarray:
        lsize = np.size(lat)
    else:
        lsize = 1

    pilm = np.zeros((lsize, lmax + 1, lmax + 1))
    pilm[:, 0, 0] = 1.0
    pilm[:, 1, 1] = np.sqrt(3) * np.sin(lat)

    '''For the diagonal element'''
    for n in range(2, lmax + 1):
        pilm[:, n, n] = np.sqrt((2 * n + 1) / (2 * n)) * np.sin(lat) * pilm[:, n - 1, n - 1]

    for n in range(1, lmax + 1):
        pilm[:, n, n - 1] = np.sqrt(2 * n + 1) * np.cos(lat) * pilm[:, n - 1, n - 1]

    for n in range(2, lmax + 1):
        for m in range(n - 2, -1, -1):
            pilm[:, n, m] = \
                np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (2 * n - 1)) \
                * np.cos(lat) * pilm[:, n - 1, m] \
                - np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (n - m - 1) * (n + m - 1) / (2 * n - 3)) \
                * pilm[:, n - 2, m]

    return pilm


def demo():
    print(np.ones(5000).shape)
    P = Legendre_polynomial(alpha=np.ones(5000)*60, lmax=10000)
    print(P.shape)
    pass

if __name__ == '__main__':
    demo()