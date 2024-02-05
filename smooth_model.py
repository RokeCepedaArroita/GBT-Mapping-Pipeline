import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


model = hp.read_map('/scratch/nas_falcon/scratch/rca/projects/gbt/skymodels/skymodel_30GHz_mK.fits')

# Smooth the map
def smooth_map(m, fwhm_initial, fwhm_final):
    ''' All inputs in degrees '''
    # Transfer FWHM in degrees
    fwhm = np.sqrt(fwhm_final**2-fwhm_initial**2)
    # Smoothing
    m2 = np.copy(m)
    m[np.isnan(m2)] = np.nanmedian(m)
    ms = hp.smoothing(m, fwhm=fwhm*np.pi/180.,verbose=False)
    # Masking
    ms[np.isnan(m2)] = np.nan
    return ms


model_smooth = smooth_map(model, fwhm_initial=6/60, fwhm_final=10/60)

hp.write_map('/scratch/nas_falcon/scratch/rca/projects/gbt/skymodels/skymodel_30GHz_mK_10arcmin.fits', model_smooth)
