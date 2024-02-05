import numpy as np
import matplotlib.pyplot as plt
import datetime
from tools import *
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')

''' Script to plot simulated map resulting from Stuart's simulation code '''

# Quick Settings

frequency = 13.7 # to decide which contour to plot 13.7, 4.85
sim_map = f'daisy_repeats000_{frequency}GHz_beampix.fits'

# Long-term settings
central_coords = [192.41, -11.51]

if frequency == 4.85:
    zoom = 1 # arbitrary zoom factor for figure
elif frequency == 13.7:
    zoom = 1 # arbitrary zoom factor for figure


base_dir = '/scratch/nas_falcon/scratch/rca/projects/gbt/sdosim/gbt_outputs'
contour_data = f'/scratch/nas_falcon/scratch/rca/projects/gbt/skymodels/skymodel_{frequency}GHz_mK.fits'
# contour_data = '/scratch/nas_falcon/scratch/rca/data/ancillary_data/CMBon/HFI_SkyMap_857_512_smth.fits'

# Custom LDN Marks

molecular_clouds = {'locs': [[192.1572,-11.1062], [192.3958,-11.3262], [192.1106,-11.7302], [192.3917,-12.2417], [192.5388,-11.5275], [192.3249,-12.0295], [192.3156,-10.7656], [191.3483,-11.0403], [192.2851,-09.0423], [192.7880,-09.0563], [191.9490, -11.5458]],
                   'names': ['LDN 1582B','LDN 1582','LDN 1577','LDN 1583','LDN 1584','LDN 1581','LDN 1580','LDN 1573','LDN 1579','LDN 1585', 'B30']}


# Open data
image, nhits, rms, w = open_simulation(base_dir, sim_map)

# Reproject contour data
contour_array, contour_footprint = healpix2wcs(data_location=contour_data, target_wcs=w, shape=np.shape(image))

# locs = np.where(contour_array==np.nanmax(contour_array))
# here1, here2 = w.wcs_pix2world(locs[0], locs[1], 0)
# print(here1, here2)
# asdas
# If 4.85 GHz, smooth contour data by 2 pixels
if frequency == 4.85:
    from scipy.ndimage.filters import gaussian_filter
    contour_array=gaussian_filter(contour_array, sigma=2)


# Plot map
plot_map(image=image, nhits=nhits, rms=rms, wcs=w, zoom=zoom, title=sim_map[:-5], contourdata=contour_array, dark_clouds=molecular_clouds, central_coords=central_coords, frequency=frequency)
plt.show()
