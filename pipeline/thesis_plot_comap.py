import numpy as np
import matplotlib.pyplot as plt


# Config

subtract_average = False

band = 'average'

perform_median_subtraction = True

central_coords = [192.41, -11.51]

final_fwhm_arcmin = 10


original_fwhm_arcmin = 4.5 # original resolution
size_of_pixel_arcmin = 1.0 # pixel sizes, comap is 1 arcmin
minmax_mK = [-5,5]
minmax_mK_smooth = [-3.5,3.5]

subtract_min=False


radius = 1.5
zoomout = 1.05 # final desired resolution


figsize = np.multiply([3.6,3.0],1.6)
delta_arrow = [-15, 4]
radii_arrow = [0.7, 0.9]


text_position = [221-25,190+125]


# Stable config
base_dir = '../COMAP'
# contour_dir = '/scratch/nas_falcon/scratch/rca/projects/gbt/skymodels/skymodel_13.7GHz_mK.fits'
# contour_dir = '/scratch/nas_falcon/scratch/rca/data/ancillary_data/contours/HFI_SkyMap_857-field-Int_2048_R2.02_full_20_arcminutes.fits'
# contour_dir = '/scratch/nas_falcon/scratch/rca/data/ancillary_data/HFI_SkyMap_857_512_smth_RJ.fits'
# contour_dir = '/scratch/nas_falcon/scratch/rca/data/ancillary_data/contours/HFI_SkyMap_857-field-Int_2048_R2.02_full_10_arcminutes.fits'
contour_dir = '/scratch/nas_falcon/scratch/rca/projects/gbt/skymodels/skymodel_30GHz_mK_10arcmin.fits'

#contour_dir = '/scratch/nas_falcon/scratch/rca/projects/gbt/skymodels/skymodel_4.85GHz_mK.fits'

def read_wcs(map_dir):
    '''Open WCS '''

    from astropy.wcs import WCS
    from astropy.io import fits

    # Open original WCS with parameters from the fits file

    hdu = fits.open(map_dir)[0]
    w = WCS(hdu.header)

    return w


def open_map(base_dir, sim_map):
    ''' Opens Spitzer Data '''

    def open_map_fits(map_dir, hdu_number=0, verbose=True):
        '''Opens fits file'''

        from astropy.io import fits

        # Open fits file
        hdul = fits.open(map_dir)
        data = hdul[hdu_number].data

        # Prints info
        if verbose:
            print(hdul.info())
            print(hdul[0].header)
            print(np.shape(data))

        return data

    # Read map
    image = np.array(open_map_fits(f'{base_dir}/{sim_map}', hdu_number=0, verbose=False))

    # Set all zeros to NaN
    image[image==0] = np.nan

    # Read and create WCS object
    wcs = read_wcs(f'{base_dir}/{sim_map}')

    return image, wcs


def healpix2wcs(data_location, target_wcs, shape):
    ''' Reproject Healpy map into WCS format '''

    # Open data
    from reproject import reproject_from_healpix, reproject_to_healpix
    from astropy.io import fits
    hdulist = fits.open(data_location)

    # Manually specify coordinates
    hdulist[1].header['COORDSYS'] = 'GALACTIC' # add coordinate frame to header

    # Perform reprojection
    array, footprint = reproject_from_healpix(input_data=hdulist[1], output_projection=target_wcs, shape_out=(shape[0],shape[1]))

    return array, footprint


def smoothimage(image, final_fwhm_arcmin, original_fwhm_arcmin, size_of_pixel_arcmin):

    # Calculate parameters
    transfer_fwhm_arcmin = np.sqrt(final_fwhm_arcmin**2-original_fwhm_arcmin**2)
    transfer_std_arcmin = transfer_fwhm_arcmin/2.355 # transfer sigma
    transfer_std_pixels = transfer_std_arcmin/size_of_pixel_arcmin

    # Do actual convolution
    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve
    kernel = Gaussian2DKernel(x_stddev=transfer_std_pixels)
    smoothed_image = convolve(image, kernel, boundary='fill', fill_value=np.nan)
    smoothed_image[np.isnan(image)] = np.nan
    return smoothed_image


# Actual script starts below



if band == 'average':
    # Open and average all bands
    for band_temp in np.arange(8):
        name = f'fg9_c2_Feeds1-2-3-5-6-9-10-12-13-14-15-16-17-18-19_Band{band_temp}-RCACROP.fits'
        image_temp, wcs = open_map(base_dir=base_dir, sim_map=name)
        if band_temp == 0:
            image_sum = image_temp
        else:
            image_sum = np.add(image_sum,image_temp)
    image = np.divide(image_sum,8)
    frequency = 30

else:
    name = f'fg9_c2_Feeds1-2-3-5-6-9-10-12-13-14-15-16-17-18-19_Band{band}-RCACROP.fits'
    image, wcs = open_map(base_dir=base_dir, sim_map=name)
    frequency = band+26.5



if subtract_average:
    # Open and average all bands
    for band_temp in np.arange(8):
        name = f'fg9_c2_Feeds1-2-3-5-6-9-10-12-13-14-15-16-17-18-19_Band{band_temp}-RCACROP.fits'
        image_temp, wcs = open_map(base_dir=base_dir, sim_map=name)
        if band_temp == 0:
            image_sum = image_temp
        else:
            image_sum = np.add(image_sum,image_temp)
    image_average = np.divide(image_sum,8)

    # Open specific band and subtract the average
    name = f'fg9_c2_Feeds1-2-3-5-6-9-10-12-13-14-15-16-17-18-19_Band{band}-RCACROP.fits'
    image, wcs = open_map(base_dir=base_dir, sim_map=name)
    frequency = band+26.5
    image = np.subtract(image, image_average)


# Subtract median of comap
if perform_median_subtraction:
    image = np.subtract(image, np.nanmedian(image))


# Smooth image
smoothed_image = smoothimage(image=image, final_fwhm_arcmin=final_fwhm_arcmin, \
    original_fwhm_arcmin=original_fwhm_arcmin, size_of_pixel_arcmin=size_of_pixel_arcmin)

if subtract_min:
    smoothed_image = smoothed_image - np.nanmin(smoothed_image)

plt.figure(1,figsize=(figsize[0],figsize[1]))
ax = plt.subplot(projection=wcs)

image = image*1000 # conver to mK
plt.imshow(image, vmin=minmax_mK[0], vmax=minmax_mK[1])
cbar = plt.colorbar()
cbar.set_label(r'$\mathrm{mK}$', labelpad=0.8)
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title(rf'COMAP ${frequency}\,$GHz: FWHM$={original_fwhm_arcmin}^\prime$')

def MAD(d,axis=0):
    ''' Return Median Absolute Deviation for array along one axis '''
    med_d = np.nanmedian(d,axis=axis)
    rms = np.sqrt(np.nanmedian((d-med_d)**2,axis=axis))*1.48
    print(f'MAD={rms},std={np.nanstd(d)}')
    return rms

std = MAD(image[~np.isnan(image)])




# Mark centre of observation
pix_x1, pix_y1 = wcs.wcs_world2pix(central_coords[0], central_coords[1], 0)
ax.plot(pix_x1, pix_y1, 'x', color='k')




def circle_in_worldcoords2pix(wcs, center, radius, npoints):

    # Theta from 0 to 2*pi
    theta = np.linspace(0, 2*np.pi, npoints)

    # Compute x and y
    dx = radius*np.cos(theta)
    dy = radius*np.sin(theta)
    x = np.add(dx, center[0])
    y = np.add(dy, center[1])

    # Convert to world pixel coordinates
    pix_x, pix_y = wcs.wcs_world2pix(x, y, 0)

    return pix_x, pix_y



# Plot circle around Ku observations
circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs, center=central_coords, radius=0.4, npoints=360)
ax.plot(circle_x, circle_y, 'k--', linewidth=0.9,alpha=0.3)

# Plot circle around C observations
circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs, center=central_coords, radius=1.5, npoints=360)
ax.plot(circle_x, circle_y, 'k--', linewidth=0.9,alpha=0.3)


# set limits
corner1x = central_coords[0]-radius*zoomout
corner2x = central_coords[0]+radius*zoomout
corner1y = central_coords[1]-radius*zoomout
corner2y = central_coords[1]+radius*zoomout
pix_x1, pix_y1 = wcs.wcs_world2pix(corner1x, corner1y, 0)
pix_x2, pix_y2 = wcs.wcs_world2pix(corner2x, corner2y, 0)
plt.xlim(np.sort([pix_x1,pix_x2]))
plt.ylim(np.sort([pix_y1,pix_y2]))
ax.set_facecolor(('#a9a9a9'))


# Reproject contour data
contour_data = contour_dir
array, footprint = healpix2wcs(data_location=contour_data, target_wcs=wcs, shape=np.shape(image))

# If 4.85 GHz, smooth contour data by 2 pixels
if band == 'C':
    from scipy.ndimage.filters import gaussian_filter
    array=gaussian_filter(array, sigma=1)

# Plot contours
ax.contour(array, colors='black', alpha=0.8, linewidths=0.5)




# Plot locations of clouds
molecular_clouds = {'locs': [[192.1572,-11.1062], [192.3958,-11.3262], [192.1106,-11.7302], [192.3917,-12.2417], [192.5388,-11.5275], [192.3249,-12.0295], [192.3156,-10.7656], [191.3483,-11.0403], [192.2851,-09.0423], [192.7880,-09.0563], [191.9490, -11.5458]],
                   'names': ['LDN 1582B','LDN 1582','LDN 1577','LDN 1583','LDN 1584','LDN 1581','LDN 1580','LDN 1573','LDN 1579','LDN 1585', 'B30']}
from astropy import units as u
from astropy.coordinates import SkyCoord
for cloud_loc in molecular_clouds['locs']:
    pix_x, pix_y = wcs.wcs_world2pix(cloud_loc[0], cloud_loc[1], 0)
    ax.plot(pix_x, pix_y, '^', color='k',alpha=0.3)


# Mark centre of observation
c = SkyCoord(l=central_coords[0]*u.degree, b=central_coords[1]*u.degree, frame='galactic')
pix_x1, pix_y1 = wcs.wcs_world2pix(c.icrs.ra.value, c.icrs.dec.value, 0)
ax.plot(pix_x1, pix_y1, 'x', color='k')


def straight_line_radius_calc_radec(centre,destination,radius,wcs):
    ''' input in galactic '''

    def galactic2radec(galpoint):
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        # Convert from galactic to icrs
        c = SkyCoord(l=galpoint[0]*u.degree, b=galpoint[1]*u.degree, frame='galactic')
        # Convert to ra-dec
        radecpoint = [c.icrs.ra.value, c.icrs.dec.value]
        return radecpoint

    centre = galactic2radec(centre)
    destination = galactic2radec(destination)


    delta_lon = destination[0] - centre[0]
    delta_lat = destination[1] - centre[1]

    angle = np.arctan(delta_lat/delta_lon) # radians

    delta_lon_rad = radius*np.cos(angle)
    delta_lat_rad = radius*np.sin(angle)

    pointx = centre[0] + delta_lon_rad
    pointy = centre[1] + delta_lat_rad

    point_at_radius = [pointx, pointy]

    pix_xlamori, pix_ylamori = wcs.wcs_world2pix(point_at_radius[0], point_at_radius[1], 0)

    final_point_wcs = [pix_xlamori, pix_ylamori]

    return final_point_wcs


def straight_line_radius_calc(centre,destination,radius, wcs):

    delta_lon = destination[0] - centre[0]
    delta_lat = destination[1] - centre[1]

    angle = np.arctan(delta_lat/delta_lon) # radians

    delta_lon_rad = radius*np.cos(angle)
    delta_lat_rad = radius*np.sin(angle)

    pointx = centre[0] + delta_lon_rad
    pointy = centre[1] + delta_lat_rad

    point_at_radius = [pointx, pointy]

    pix_xlamori, pix_ylamori = wcs.wcs_world2pix(point_at_radius[0], point_at_radius[1], 0)

    final_point_wcs = [pix_xlamori, pix_ylamori]

    return final_point_wcs


# Lambda orionis position
point_at_radius = straight_line_radius_calc(central_coords,[195.05189264, -11.99506278],radius=radius*radii_arrow[0], wcs=wcs)
point_at_radius2 = straight_line_radius_calc(central_coords,[195.05189264, -11.99506278],radius=radius*radii_arrow[1], wcs=wcs)
delta_point_radius = np.subtract(point_at_radius2, point_at_radius)
ax.text(point_at_radius2[0]+delta_arrow[0], point_at_radius2[1]+delta_arrow[1], r'$\lambda$ Ori', color='k', rotation=np.rad2deg(np.arctan(delta_point_radius[1]/delta_point_radius[0])), fontsize=9)
plt.arrow(point_at_radius[0], point_at_radius[1], delta_point_radius[0], delta_point_radius[1], head_width=1.5, head_length=3, fc='k', ec='k', linewidth=1)



ax.text(text_position[0], text_position[1], rf'FWHM$={original_fwhm_arcmin}^\prime$'+'\n'+rf'(COMAP'+'\n'+fr'${frequency}\,$GHz)', color='k',fontsize=12)


plt.savefig(f'./figures/maps/{name[:-5]}_thesis_COMAP.png')
plt.savefig(f'./figures/maps/{name[:-5]}_thesis_COMAP.pdf')

#plt.show()
#plt.close()



plt.figure(3,figsize=(8,6.5))
ax = plt.subplot(projection=wcs)

smoothed_image = smoothed_image*1000
plt.imshow(smoothed_image, vmin=minmax_mK_smooth[0], vmax=minmax_mK_smooth[1])


def MAD(d,axis=0):
    ''' Return Median Absolute Deviation for array along one axis '''
    med_d = np.nanmedian(d,axis=axis)
    rms = np.sqrt(np.nanmedian((d-med_d)**2,axis=axis))*1.48
    print(f'MAD={rms},std={np.nanstd(d)}')
    return rms

std = MAD(smoothed_image[~np.isnan(smoothed_image)])


cbar = plt.colorbar()
cbar.set_label(r'$\mathrm{mK}$', labelpad=0.8)
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
#plt.title(rf'COMAP ${frequency}\,$GHz: FWHM$={final_fwhm_arcmin}^\prime$')

# Set limits
corner1x = central_coords[0]-radius*zoomout
corner2x = central_coords[0]+radius*zoomout
corner1y = central_coords[1]-radius*zoomout
corner2y = central_coords[1]+radius*zoomout
pix_x1, pix_y1 = wcs.wcs_world2pix(corner1x, corner1y, 0)
pix_x2, pix_y2 = wcs.wcs_world2pix(corner2x, corner2y, 0)
plt.xlim(np.sort([pix_x1,pix_x2]))
plt.ylim(np.sort([pix_y1,pix_y2]))
ax.set_facecolor(('#a9a9a9'))


# Reproject contour data
contour_data = contour_dir
array, footprint = healpix2wcs(data_location=contour_data, target_wcs=wcs, shape=np.shape(smoothed_image))

# If 4.85 GHz, smooth contour data by 2 pixels
if band == 'C':
    from scipy.ndimage.filters import gaussian_filter
    array=gaussian_filter(array, sigma=1)

# Plot contours
ax.contour(array, colors='black', alpha=0.8, linewidths=0.5)


# Plot locations of clouds
molecular_clouds = {'locs': [[192.1572,-11.1062], [192.3958,-11.3262], [192.1106,-11.7302], [192.3917,-12.2417], [192.5388,-11.5275], [192.3249,-12.0295], [192.3156,-10.7656], [191.3483,-11.0403], [192.2851,-09.0423], [192.7880,-09.0563], [191.9490, -11.5458]],
                   'names': ['LDN 1582B','LDN 1582','LDN 1577','LDN 1583','LDN 1584','LDN 1581','LDN 1580','LDN 1573','LDN 1579','LDN 1585', 'B30']}
from astropy import units as u
from astropy.coordinates import SkyCoord
for cloud_loc in molecular_clouds['locs']:
    pix_x, pix_y = wcs.wcs_world2pix(cloud_loc[0], cloud_loc[1], 0)
    ax.plot(pix_x, pix_y, '^', color='k',alpha=0.3)

central_coords = [192.41, -11.51]

# Mark centre of observation
c = SkyCoord(l=central_coords[0]*u.degree, b=central_coords[1]*u.degree, frame='galactic')
pix_x1, pix_y1 = wcs.wcs_world2pix(c.icrs.ra.value, c.icrs.dec.value, 0)
ax.plot(pix_x1, pix_y1, 'x', color='k')

# Plot circle around Ku observations
circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs, center=central_coords, radius=0.4, npoints=360)
ax.plot(circle_x, circle_y, 'k--', linewidth=0.9,alpha=0.3)

# Plot circle around C observations
circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs, center=central_coords, radius=1.5, npoints=360)
ax.plot(circle_x, circle_y, 'k--', linewidth=0.9,alpha=0.3)




# Lambda orionis position
point_at_radius = straight_line_radius_calc(central_coords,[195.05189264, -11.99506278],radius=radius*radii_arrow[0], wcs=wcs)
point_at_radius2 = straight_line_radius_calc(central_coords,[195.05189264, -11.99506278],radius=radius*radii_arrow[1], wcs=wcs)
delta_point_radius = np.subtract(point_at_radius2, point_at_radius)
ax.text(point_at_radius2[0]+delta_arrow[0], point_at_radius2[1]+delta_arrow[1], r'$\lambda$ Ori', color='k', rotation=np.rad2deg(np.arctan(delta_point_radius[1]/delta_point_radius[0])), fontsize=9)
plt.arrow(point_at_radius[0], point_at_radius[1], delta_point_radius[0], delta_point_radius[1], head_width=1.5, head_length=3, fc='k', ec='k', linewidth=1)

ax.text(text_position[0], text_position[1], rf'FWHM$={final_fwhm_arcmin}^\prime$'+'\n'+r'(COMAP'+'\n'+fr'${frequency}\,$GHz)', color='k',fontsize=12)


plt.savefig(f'COMAP_band_average.png')
plt.savefig(f'COMAP_band_average.pdf')


plt.savefig(f'./figures/maps/{name[:-5]}_COMAP_thesis_smooth.png')
plt.savefig(f'./figures/maps/{name[:-5]}_COMAP_thesis_smooth.pdf')

plt.show()
plt.close()


































#image = smoothed_image

# PLOT NOISE PROPERTIES
plt.figure(1)
def MAD(d,axis=0):
    ''' Return Median Absolute Deviation for array along one axis '''
    med_d = np.nanmedian(d,axis=axis)
    rms = np.sqrt(np.nanmedian((d-med_d)**2,axis=axis))*1.48
    print(f'MAD={rms},std={np.nanstd(d)}')
    return rms

std = MAD(image[~np.isnan(image)])

counts = plt.hist(image[~np.isnan(image)],400, histtype='step', label='Pixel values')
counts = counts[0]
plt.xlim([np.nanmedian(image)-5*std,np.nanmedian(image)+5*std])
plt.plot([0,0],[0,np.nanmax(counts)*1.1],'k--',alpha=0.1)
plt.plot([np.nanmedian(image),np.nanmedian(image)],[0,np.nanmax(counts)*1.1],'g--')
plt.plot([np.nanmedian(image)+std,np.nanmedian(image)+std],[0,np.nanmax(counts)*1.1],'g--')
plt.plot([np.nanmedian(image)-std,np.nanmedian(image)-std],[0,np.nanmax(counts)*1.1],'g--', label=f'MAD = {std:.1f} mK')
plt.ylim([0,np.nanmax(counts)*1.1])
plt.xlabel('Pixel Values (mK)')
plt.title(f'{name}')
plt.legend()
plt.savefig(f'./figures/maps/{name[:-5]}_dist.png')
plt.savefig(f'./figures/maps/{name[:-5]}_dist.pdf')
plt.show()
