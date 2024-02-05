# open maps

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt



ame = hp.read_map('../proposal/others/sky_models/ame_model_4.85GHz_mK.fits')
freefree = hp.read_map('../proposal/others/sky_models/free-free_model_4.85GHz_mK.fits')

ame = hp.read_map('../proposal/others/sky_models/ame_model_13.7GHz_mK.fits')
freefree = hp.read_map('../proposal/others/sky_models/free-free_model_13.7GHz_mK.fits')


ame = hp.read_map('../proposal/others/sky_models/ame_model_22.8GHz_mK.fits')
freefree = hp.read_map('../proposal/others/sky_models/free-free_model_22.8GHz_mK.fits')

ame = hp.read_map('../proposal/others/sky_models/ame_model_30GHz_mK.fits')
freefree = hp.read_map('../proposal/others/sky_models/free-free_model_30GHz_mK.fits')



ame       = hp.ud_grade(ame,nside_out=1024)
freefree  = hp.ud_grade(freefree,nside_out=1024)

# smooth to 1 deg scales maybe

ame_ratio = np.divide(ame,np.add(ame,freefree))
freefree_ratio = np.divide(freefree,np.add(ame,freefree))

# hp.mollview(ame_ratio,min=0,max=1)
hp.mollview(freefree,min=0,max=1)
hp.projtext(192.41, -11.51, 'x', lonlat=True, coord='G')
plt.show()
