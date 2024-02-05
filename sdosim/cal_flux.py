import numpy as np


source_3C48 = [2.465,-0.004,-0.1251]
source_3C147 = [2.806,-0.140,-0.1031]
source_3C161 = [1.250,+0.726,-0.2286]


def cal_model(nu,source):
    ''' Input nu in GHz '''

    nu = nu*1000. # Convert to MHz

    a,b,c = source

    log_flux = a+b*np.log10(nu)+c*(np.log10(nu))**2

    flux = (10**(log_flux)) # in Jy

    return flux


source_now = source_3C161

for frequency in [4.85,13.7]:
    print(f'{cal_model(nu=frequency,source=source_now):.1f} at {frequency} GHz')
