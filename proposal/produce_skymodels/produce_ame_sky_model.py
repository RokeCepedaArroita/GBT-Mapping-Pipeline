import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')


# Options

frequency = 30 # GHz


# Factors

if frequency==13.7:
    factor = 108.2 # uk/(MJR sr-1)
elif frequency==4.85:
    factor = 64 # uk/(MJR sr-1)
elif frequency==22.8:
    factor = 50.95 # uk/(MJR sr-1)
elif frequency==12.9:
    factor = 114.43189807 # uk/(MJR sr-1)
elif frequency==30:
    factor = 27.775264 # uk/(MJR sr-1)


# Open Planck 2018 857 GHz map

simulated_ame_map = hp.read_map('/scratch/nas_falcon/scratch/rca/projects/gbt/data/ancillary_data/HFI_SkyMap_857-field-Int_2048_R3.00_full.fits')


# Scale down by factor

simulated_ame_map = simulated_ame_map*factor # uK


# Convert map to mk

simulated_ame_map = simulated_ame_map*1e-3


# Save map in milikelvin

hp.write_map(f'ame_model_{frequency}GHz_mK.fits', simulated_ame_map)


# Plot to see

phi = [-164.3]
theta = [-11.6]
reson = 3.4
hp.gnomview(simulated_ame_map, rot=[phi, theta],reso=reson, title=f'Simulated AME {frequency} GHz (mK)')
plt.show()
