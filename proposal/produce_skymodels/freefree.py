import numpy as np
import sys

def afac(Te,x):
    return 0.366 * x**0.1 * Te**-0.15 * (np.log(4.995e-2/x) + 1.5*np.log(Te))


def R2Tb( Te, x):
    """
    Converts Rayleigh to free-free brightness temperature in uK from Dickinson 2003

    Inputs:
    Te - electron temperature
    x  - frequency in GHz - returns in microK
    """
    T4 = Te/1e4
    return 8.396e3 * x**-2.1 * T4**0.667 * 10**(0.029/T4) * (1+0.08) * afac(Te,x)

if __name__ == "__main__":

    x = float(sys.argv[1])

    print(R2Tb(7500,x))
