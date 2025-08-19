from sympy.physics.wigner import wigner_3j
import numpy as np
from SaGEA.auxiliary.aux_tool.FileTool import FileTool
from SaGEA.auxiliary.load_file.LoadL2SH import load_SHC
from datetime import date
from tqdm import tqdm


def Ocean_function(loadfile="data/basin_mask/SH/Ocean_maskSH.dat",lmax=60):
    OceanFuction_SH = FileTool.get_project_dir(loadfile)
    shc_OceanFunction = load_SHC(OceanFuction_SH, key='', lmax=lmax)  # load basin mask (in SHC)
    grid_basin = shc_OceanFunction.to_grid(grid_space=1)
    grid_basin.limiter(threshold=0.5)
    ocean_function = grid_basin.value[0]
    ocean = {
        "GRID":ocean_function,
        "SH":shc_OceanFunction,
    }
    return ocean
def compute_wigner_3j(l1, l2, l, m1, m2, m):
    return float(wigner_3j(l1, l2, l, m1, m2, m))

def compute_gaunt(l1, l2, l, m1, m2, m):
    if m1 + m2 != m:
        return 0.0
    cg_l0 = compute_wigner_3j(l1, l2, l, 0, 0, 0)
    cg_m = compute_wigner_3j(l1, l2, l, m1, m2, -m)
    prefactor = np.sqrt((2*l1 + 1) * (2*l2 + 1) / (4 * np.pi * (2*l + 1)))
    return prefactor * cg_l0 * cg_m * (-1)**m

def spectral_mask(clm_g_C, clm_g_S, clm_mask_C, clm_mask_S, lmax_g=60, lmax_mask=360):
    lmax_out = lmax_g
    clm_ocean_C = np.zeros((lmax_out + 1, 2 * lmax_out + 1))
    clm_ocean_S = np.zeros((lmax_out + 1, 2 * lmax_out + 1))
    for l in tqdm(range(lmax_out + 1)):
        for m in range(-l, l + 1):
            sum_C,sum_S = 0.0,0.0
            for l1 in range(lmax_g + 1):
                for m1 in range(0, l1 + 1):
                    for l2 in range(lmax_mask + 1):
                        for m2 in range(0, l2 + 1):
                            for s1 in [-1, 1]:
                                for s2 in [-1, 1]:
                                    m1_s = m1 * s1
                                    m2_s = m2 * s2
                                    m_s = m1_s + m2_s
                                    if abs(m_s) > l:
                                        continue
                                    gaunt = compute_gaunt(l1, l2, l, m1_s, m2_s, m_s)
                                    # print(f"gaunt is {gaunt}")
                                    C1 = clm_g_C[l1, m1] if s1 == 1 else clm_g_C[l1, m1]
                                    S1 = clm_g_S[l1, m1] if s1 == 1 else -clm_g_S[l1, m1]
                                    C2 = clm_mask_C[l2, m2] if s2 == 1 else clm_mask_C[l2, m2]
                                    S2 = clm_mask_S[l2, m2] if s2 == 1 else -clm_mask_S[l2, m2]

                                    sum_C += (C1 * C2 - S1 * S2) * gaunt
                                    sum_S += (C1 * S2 + S1 * C2) * gaunt
            clm_ocean_C[l, m] = sum_C
            clm_ocean_S[l, m] = sum_S if m != 0 else 0.0  # S_{l0} = 0
    return clm_mask_C,clm_mask_S

lmax=60
begin_date, end_date = date(2003, 1, 1), date(2003, 1, 31)
print(f'----Load GRACE--------\n')
gsm_dir = FileTool.get_project_dir("I:/GFZ/GSM/SLE_GSM/BA01/")
shc = load_SHC(gsm_dir, key='GRCOF2', lmax=lmax, begin_date=begin_date, end_date=end_date,
               get_dates=False,)
ocean_sh = Ocean_function()['SH']
print(shc.value.shape,ocean_sh.value.shape)

shc_C,shc_S = shc.get_cs2d()
ocean_C,ocean_S = ocean_sh.get_cs2d()

print(shc_C.shape,shc_S.shape,ocean_S.shape)

clm_ocean_C, clm_ocean_S = spectral_mask(clm_g_C=shc_C[0], clm_g_S=shc_S[0], clm_mask_C=ocean_C[0], clm_mask_S=ocean_S[0],lmax_g=60,lmax_mask=60)

# 保存结果
np.savez("ocean_masked.npz", C=clm_ocean_C, S=clm_ocean_S)
# gaunt1 = compute_gaunt(4, 22, 31, 3, 4, 3)
# print(gaunt1)
