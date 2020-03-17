"""
wrap chi_1, so that it is within +-pi of chi_2
"""
import numpy as np

def wrap(chi_c, chi):
    while chi_c - chi > np.pi:
        chi_c = chi_c - 2.0 * np.pi
    while chi_c - chi < -np.pi:
        chi_c = chi_c + 2.0 * np.pi
    return chi_c
