import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


ame = hp.read_map('/scratch/nas_falcon/scratch/rca/projects/gbt/proposal/others/sky_models/ame_model_30GHz_mK.fits')

ff = hp.read_map('/scratch/nas_falcon/scratch/rca/projects/gbt/proposal/others/sky_models/free-free_model_30GHz_mK.fits')

print(hp.get_nside(ame))
print(hp.get_nside(ff))

ame = hp.ud_grade(ame, nside_out=hp.get_nside(ff) )

print(hp.get_nside(ame))
print(hp.get_nside(ff))


total = ame + ff

hp.write_map('/scratch/nas_falcon/scratch/rca/projects/gbt/skymodels/skymodel_30GHz_mK.fits',total)

hp.mollview(total)
plt.show()
