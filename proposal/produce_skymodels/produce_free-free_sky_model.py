import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')


# Options

frequency = 30 # GHz

if frequency == 13.7:
    factor = 32.5104218770366
elif frequency == 4.85:
    factor = 294.0749598539121
elif frequency == 22.8:
    factor = 10.96849636947083
elif frequency == 12.9:
    factor = 36.9517039567314
elif frequency == 30:
    factor = 6.095933830526541

# Open H-alpha map

halpha_map = hp.read_map('/scratch/nas_falcon/scratch/rca/projects/gbt/data/ancillary_data/Halpha_fwhm06_1024.fits')

#
# # Conversion formula ## off by factor of 1000 since it returns to mk not microk? chekc with stuarts code
#
# def halpha2freefree(nu, Te):
#     ''' nu in GHz, Te in K '''
#
#     frequencies = [0.4, 1.4, 2.3, 10, 30, 44, 70, 100]
#     gaunt_7500 = [6.15, 5.49, 5.23, 4.46, 3.88, 3.68, 3.43, 3.25]
#
#     from scipy.interpolate import interp1d
#     interpolate = interp1d(frequencies, gaunt_7500, kind='quadratic')
#
#     #plt.plot(frequencies, gaunt_7500)
#     #plt.show()
#
#     gaunt = interpolate(nu)
#
#     factor = 8.396*1e3 * gaunt * np.power(nu,-2.1) * np.power(Te/1e4,0.667) * np.power(10, 0.029/(Te/1e4)) * 1.08
#
#     return factor
#
# factor = halpha2freefree(nu=frequency, Te=7500)

print(f'conversion factor is {factor} at 7500 K')



# Create free-free map

freefree_map = halpha_map*factor*1e-3 # in mK, since the factors above computed usng stuarts code are to microk


# Plot and save map

hp.write_map(f'free-free_model_{frequency}GHz_mK.fits', freefree_map, overwrite=True)

hp.gnomview(freefree_map,rot=[-164.3, -11.6],reso=3.4, title=f'Free-Free {frequency}GHz from H-Alpha (mK)')
plt.show()
