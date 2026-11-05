# -*- coding: utf-8 -*-
# @Author  : Weihang Zhang
# @Email   : zhang.17371@osu.edu
# @File    : Adjustment.py
# @Create  : 10/24/26 23:54
# @Project : PySWOT
# @Software: PyCharm
"""
Description:
"""

import numpy as np

class Adjustment:
    def __init__(self):
        self.L_obs = None
        self.P_wgt = None
        self.A_dgn = None
        self.dof = None

        pass
    def __initial_P_Matrix(self):
        single_line_L = self.L_obs.flatten()
        self.P_wgt = np.diag(np.ones_like(single_line_L))
        return self.P_wgt

    def __initial_dof(self):
        n_num = self.A_dgn.shape[0]     # number of the observation equations (row numbers)
        r_num = self.A_dgn.shape[1]     # number of the unknown/estimation/parameters (column numbers)
        self.dof = n_num - r_num
        print(f"> > degree of order (dof) is {self.dof}")
        return self.dof

    def least_squares(self,L,A,P=None):
        """
        This is the least-squares solution to find the model parameters
        :param L: observation values
        :param A: Design matrix
        :param P: Weight matrix
        :return:
        """
        self.L_obs = L.reshape(-1, 1)  # make sure the vector L is (n,1), not only (n,)
        if P is None:
            self.P_wgt = self.__initial_P_Matrix()
        else:
            self.P_wgt = P
        self.A_dgn = A
        """Normal equation"""
        N = self.A_dgn.T @ self.P_wgt @ self.A_dgn  # ATPA
        c = self.A_dgn.T @ self.P_wgt @ self.L_obs  # ATPL
        para = np.linalg.inv(N) @ c                 # [(ATPA)-1](ATPL)
        return para

    def residual(self,L,A,P=None):
        self.L_obs = L.reshape(-1, 1)
        para = self.least_squares(L,A,P)
        v = self.L_obs - self.A_dgn * para
        return v

    def posterior(self,L,A,P=None):
        self.L_obs = L.reshape(-1, 1)
        v = self.residual(self.L_obs,A,P)
        dof = self.__initial_dof()
        sigma = np.sqrt((v.T @ P @ v)/dof)


    def goodness_fit(self):
        iter = 0
        pass




def demo():
    t = np.linspace(0,10,5)
    L_obs = np.array([3,4,5,2,34,5])
    # observation equation is: y=a*t

    A = Adjustment()
    # x_model = A.least_squares(L_obs,A_dgn)
    # print(x_model)
    # L_obs = L_obs.reshape(-1,1)
    # L_inv = np.linalg.inv(L_obs)
    # L_t = L_obs.T

    # P_wgt = np.diag(np.ones_like(L_obs.flatten()))
    # print(f"L_obs shape is:{L_obs.shape},{L_t.shape}")
    # print(f"P_wgt shape is:{P_wgt.shape}")
    # a = Adjustment()

    pass

if __name__ == "__main__":
    demo()
