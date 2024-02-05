import numpy as np
import matplotlib.pyplot as plt
import datetime
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')


def openimage(map_dir, hdu_number=0, verbose=True):
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


def read_wcs(map_dir):
    '''Open WCS '''

    from astropy.wcs import WCS
    from astropy.io import fits

    # Open original WCS with parameters from the fits file

    hdu = fits.open(map_dir)[0]
    w = WCS(hdu.header)

    return w


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



def reproject_edges(data_location, target_wcs, shape):
    ''' Reproject Healpy map into WCS format '''

    # Open data
    from reproject import reproject_from_healpix, reproject_to_healpix, reproject_exact
    from astropy.io import fits
    hdul = fits.open(data_location)
    data = hdul[0].data

    # Manually specify coordinates
    # hdulist[1].header['COORDSYS'] = 'GALACTIC' # add coordinate frame to header

    # Perform reprojection
    array, footprint = reproject_exact(input_data=hdul, output_projection=target_wcs, shape_out=(shape[0],shape[1]), hdu_in=0, parallel=True)

    array_edges = np.copy(array)
    array_edges[~np.isnan(array_edges)] = 1
    array_edges[np.isnan(array_edges)] = 0

    return array_edges, footprint


def open_spitzer(base_dir, sim_map):
    ''' Opens Spitzer Data '''

    def open_spitzer_fits(map_dir, hdu_number=0, verbose=True):
        '''Opens fits file'''

        from astropy.io import fits

        # Open fits file
        hdul = fits.open(map_dir)
        data = hdul[hdu_number].data

        units = hdul[0].header['BUNIT']
        instrument = hdul[0].header['INSTRUME']

        print(f'Units are {units} for this {instrument} image.')

        # Prints info
        if verbose:
            print(hdul.info())
            print(hdul[0].header)
            print(np.shape(data))

        return data

    # Read map
    image = np.array(open_spitzer_fits(f'{base_dir}/{sim_map}', hdu_number=0, verbose=False))

    # Set all zeros to NaN
    image[image==0] = np.nan

    # Read and create WCS object
    wcs = read_wcs(f'{base_dir}/{sim_map}')

    return image, wcs


def open_simulation(base_dir, sim_map):
    ''' Opens Stuart's Simulated Data '''

    # Read map
    image = np.array(openimage(f'{base_dir}/{sim_map}', hdu_number=0, verbose=False))
    nhits = np.array(openimage(f'{base_dir}/{sim_map}', hdu_number=1, verbose=False))
    rms = np.array(openimage(f'{base_dir}/{sim_map}', hdu_number=2, verbose=False))

    # Set all zeros to NaN
    image[image==0] = np.nan
    nhits[nhits==0] = np.nan
    rms[rms==0] = np.nan
    image[np.isinf(image)] = np.nan
    nhits[np.isinf(nhits)] = np.nan
    rms[np.isinf(rms)] = np.nan

    # Read and create WCS object
    wcs = read_wcs(f'{base_dir}/{sim_map}')

    return image, nhits, rms, wcs



def open_comap(mapdir):
    ''' Opens Stuart's Simulated Data '''

    # Read map
    image = np.array(openimage(f'{mapdir}', hdu_number=0, verbose=False))

    # Set all zeros to NaN
    image[image==0] = np.nan
    image[np.isinf(image)] = np.nan

    # Read and create WCS object
    wcs = read_wcs(f'{mapdir}')

    return image, wcs




def get_coords(wcs, data):
    ''' Returns coordinates from WCS in the default coordinate system,
    as Skycoord object, in this case Galactic coordinates in degrees '''

    from astropy.wcs import utils

    # Pixel numbers on each axis
    xcoords = np.arange(np.shape(data)[0])
    ycoords = np.arange(np.shape(data)[1])

    # All combinations of pixels, both for x and y
    xcoords_all = np.tile(xcoords,np.shape(data)[0])
    ycoords_all = np.transpose(np.tile(ycoords, (np.shape(data)[1],1))).flatten()

    # Return coordinates of all pixels
    coords = utils.pixel_to_skycoord(xcoords_all, ycoords_all, wcs, origin=0, mode='all')

    return coords



def plot_square_fromradec(lon_bounds, lat_bounds, wcs):

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    # Convert from icrs to galactic
    c_lower = SkyCoord(ra=lon_bounds[0]*u.degree, dec=lat_bounds[0]*u.degree, frame='icrs')
    c_upper = SkyCoord(ra=lon_bounds[1]*u.degree, dec=lat_bounds[1]*u.degree, frame='icrs')

    lon_bounds = [c_lower.galactic.l.value, c_upper.galactic.l.value]
    lat_bounds = [c_lower.galactic.b.value, c_upper.galactic.b.value]

    square_x = [lon_bounds[0], lon_bounds[0], lon_bounds[1], lon_bounds[1], lon_bounds[0]]
    square_y = [lat_bounds[0], lat_bounds[1], lat_bounds[1], lat_bounds[0], lat_bounds[0]]

    square_x_wcs, square_y_wcs = wcs.wcs_world2pix(square_x, square_y, 0)

    return square_x_wcs, square_y_wcs



def reproject_entire_image(spitzer_base, mapname, target_wcs, target_image):

    from reproject.mosaicking import coadd
    array, footprint = coadd(f'{spitzer_base}/{mapname}', output_projection=target_wcs, shape_out=np.shape(target_image), input_weights=None, hdu_in=None, reproject_function=None, hdu_weights=None, combine_function='mean', match_background=False, background_reference=None)


    return array


def circle_in_radecworldcoords2pix(wcs, center, radius, npoints):

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    # Theta from 0 to 2*pi
    theta = np.linspace(0, 2*np.pi, npoints)

    # Compute x and y
    dx = radius*np.cos(theta)
    dy = radius*np.sin(theta)
    x = np.add(dx, center[0])
    y = np.add(dy, center[1])

    # Convert from galactic to icrs
    c = SkyCoord(l=x*u.degree, b=y*u.degree, frame='galactic')

    # Convert to world pixel coordinates
    pix_x, pix_y = wcs.wcs_world2pix(c.icrs.ra.value, c.icrs.dec.value, 0)

    return pix_x, pix_y


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



def plot_map(image, nhits, rms, wcs, zoom, title, contourdata=None, dark_clouds=None, central_coords=None, frequency=None):
    ''' Visualise simulation results '''

    figsize = np.multiply([5,4],0.8)
    plot_xlabel = True # True False

    # Plot image
    plt.figure(1, figsize=(figsize[0],figsize[1]))
    ax = plt.subplot(projection=wcs)
    ax.set_facecolor(('#a9a9a9'))
    plt.imshow(image, vmin=0, vmax=np.nanmax(contourdata)+np.nanmedian(rms)/2., origin='lower')

    # Apply zoom
    plt.xlim(image.shape[1]/zoom, image.shape[1]*(zoom-1)/zoom)
    plt.ylim(image.shape[1]/zoom, image.shape[0]*(zoom-1)/zoom)

    # Colorbar
    cbar = plt.colorbar()
    # cbar.set_label('mK', rotation=270, labelpad=20)

    if frequency ==13.7:
        band = 'Ku'
        spacing=15
    elif frequency == 4.85:
        band = 'C'
        spacing=60

    # Grid
    plt.grid(color='white', ls='solid', linewidth=0.5)


    from astropy import units as u
    lon = ax.coords[0]
    lat = ax.coords[1]
    lat.set_ticks(spacing=spacing*u.arcmin)
    lon.set_ticks(spacing=spacing*u.arcmin)

    # Labels
    if plot_xlabel:
        plt.xlabel('Galactic Longitude')
    else:
        lon = ax.coords[0]
        lon.set_axislabel(r'$\,$')


    plt.ylabel('Galactic Latitude')
    plt.title(f'{band}-Band Signal [mK]')

    # Plot contours
    ax.contour(contourdata, colors='black', alpha=0.8, linewidths=0.5)


    # Plot locations of clouds
    if not dark_clouds is None:
        for cloud_loc in dark_clouds['locs']:
            pix_x, pix_y = wcs.wcs_world2pix(cloud_loc[0], cloud_loc[1], 0)
            ax.plot(pix_x, pix_y, '^', color='k')


    # Mark centre of observation
    pix_x1, pix_y1 = wcs.wcs_world2pix(central_coords[0], central_coords[1], 0)
    ax.plot(pix_x1, pix_y1, 'x', color='k')

    # Plot circle around Ku observations
    circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs, center=central_coords, radius=0.4, npoints=360)
    ax.plot(circle_x, circle_y, 'k--', linewidth=0.9)

    # Save figure
    plt.savefig(f'./simresults/{title}_map.png', bbox_inches='tight', pad_inches=0, dpi=600)
    plt.savefig(f'./simresults/{title}_map.pdf', bbox_inches='tight', pad_inches=0, dpi=600)


    # plot_single_map(gs, gs_number=0, data=image, wcs=wcs, units='mK', contourdata=contourdata, zoom=zoom, yinfo=True)
    # plot_single_map(gs, gs_number=1, data=rms, wcs=wcs, units='mK', contourdata=contourdata, zoom=zoom, yinfo=False)
    # plot_single_map(gs, gs_number=2, data=nhits, wcs=wcs, units='#', contourdata=contourdata, zoom=zoom, yinfo=False)

    # Plot hits
    plt.figure(2, figsize=(figsize[0],figsize[1]))
    ax = plt.subplot(projection=wcs)
    ax.set_facecolor(('#a9a9a9'))
    plt.imshow(np.log10(nhits), vmin=np.nanmin(np.log10(nhits)), vmax=np.nanmax(np.log10(nhits)), origin='lower')

    print(f'Total exposure time over the map = {np.nansum(nhits/3600):.2f} hours (remember that for Ku the requested time is half this)')

    # Apply zoom
    plt.xlim(image.shape[1]/zoom, image.shape[1]*(zoom-1)/zoom)
    plt.ylim(image.shape[1]/zoom, image.shape[0]*(zoom-1)/zoom)

    # Colorbar
    cbar = plt.colorbar()
    # cbar.set_label(r'$\mathrm{log}_{10}(\Delta T/\mathrm{s})$', rotation=270, labelpad=20)

    # Grid
    plt.grid(color='white', ls='solid', linewidth=0.5)

    from astropy import units as u
    lon = ax.coords[0]
    lat = ax.coords[1]
    lat.set_ticks(spacing=spacing*u.arcmin)
    lon.set_ticks(spacing=spacing*u.arcmin)

    # Labels
    if plot_xlabel:
        plt.xlabel('Galactic Longitude')
    else:
        lon = ax.coords[0]
        lon.set_axislabel(r'$\,$')

    plt.ylabel('Galactic Latitude')
    plt.title(f'{band}-Band Tot. Exposure [' + r'$\mathrm{log}_{10}(\Delta T/\mathrm{sec})$' + ']')

    # Plot contours
    ax.contour(contourdata, colors='black', alpha=0.8, linewidths=0.5)

    print(f'Median exposure is {np.nanmedian(nhits):.2f} sec')
    print(f'Mean exposure is {np.nanmean(nhits):.2f} sec')

    # Plot locations of clouds
    if not dark_clouds is None:
        for cloud_loc in dark_clouds['locs']:
            pix_x, pix_y = wcs.wcs_world2pix(cloud_loc[0], cloud_loc[1], 0)
            ax.plot(pix_x, pix_y, '^', color='k')

    # Mark centre of observation
    pix_x1, pix_y1 = wcs.wcs_world2pix(central_coords[0], central_coords[1], 0)
    ax.plot(pix_x1, pix_y1, 'x', color='k')

    # Plot circle around Ku observations
    circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs, center=central_coords, radius=0.4, npoints=360)
    ax.plot(circle_x, circle_y, 'k--', linewidth=0.9)

    # Save figure
    plt.savefig(f'./simresults/{title}_exp.png', bbox_inches='tight', pad_inches=0, dpi=600)
    plt.savefig(f'./simresults/{title}_exp.pdf', bbox_inches='tight', pad_inches=0, dpi=600)


    # Plot rms
    plt.figure(3, figsize=(figsize[0],figsize[1]))
    ax = plt.subplot(projection=wcs)
    ax.set_facecolor(('#a9a9a9'))

    plt.imshow(rms, vmin=0, vmax=np.nanmedian(rms)*2, origin='lower')

    # Apply zoom
    plt.xlim(image.shape[1]/zoom, image.shape[1]*(zoom-1)/zoom)
    plt.ylim(image.shape[1]/zoom, image.shape[0]*(zoom-1)/zoom)

    # Colorbar
    cbar = plt.colorbar()
    # cbar.set_label('mK', rotation=270, labelpad=20)

    # Grid
    plt.grid(color='white', ls='solid', linewidth=0.5)

    from astropy import units as u
    lon = ax.coords[0]
    lat = ax.coords[1]
    lat.set_ticks(spacing=spacing*u.arcmin)
    lon.set_ticks(spacing=spacing*u.arcmin)

    # Labels
    if plot_xlabel:
        plt.xlabel('Galactic Longitude')
    else:
        lon = ax.coords[0]
        lon.set_axislabel(r'$\,$')


    lat = ax.coords[1]
    lat.set_axislabel(r'$\,$')
    lat.set_ticklabel_visible(False)
    #plt.ylabel('Galactic Latitude')

    plt.title(f'{band}-Band Noise [mK]')

    print(f'Median rms is {np.nanmedian(rms):.2f} mK')
    print(f'Mean rms is {np.nanmean(rms):.2f} mK')

    # Plot contours
    ax.contour(contourdata, colors='black', alpha=0.8, linewidths=0.5)

    # Plot locations of clouds
    if not dark_clouds is None:
        for cloud_loc in dark_clouds['locs']:
            pix_x, pix_y = wcs.wcs_world2pix(cloud_loc[0], cloud_loc[1], 0)
            ax.plot(pix_x, pix_y, '^', color='k')


    # Mark centre of observation
    pix_x1, pix_y1 = wcs.wcs_world2pix(central_coords[0], central_coords[1], 0)
    ax.plot(pix_x1, pix_y1, 'x', color='k')

    # Plot circle around Ku observations
    circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs, center=central_coords, radius=0.4, npoints=360)
    ax.plot(circle_x, circle_y, 'k--', linewidth=0.9)

    # Save figure
    plt.savefig(f'./simresults/{title}_rms.png', bbox_inches='tight', pad_inches=0, dpi=600)
    plt.savefig(f'./simresults/{title}_rms.pdf', bbox_inches='tight', pad_inches=0, dpi=600)



    # Plot signal to noise
    plt.figure(4, figsize=(figsize[0],figsize[1]))
    ax = plt.subplot(projection=wcs)
    ax.set_facecolor(('#a9a9a9'))

    min_contours = np.nanmin(contourdata[~np.isnan(image)])

    print(f'Subtracting {min_contours} mK from contourdata to create signal to noise')

    signal_to_noise = np.divide( np.subtract(image,min_contours) , rms)

    plt.imshow(signal_to_noise, vmin=0, vmax=np.nanmedian(signal_to_noise)*4, origin='lower')

    print(f'Median signal to noise is {np.nanmedian(signal_to_noise):.2f} sigma')
    print(f'Mean signal to noise is {np.nanmean(signal_to_noise):.2f} sigma')

    # Apply zoom
    plt.xlim(image.shape[1]/zoom, image.shape[1]*(zoom-1)/zoom)
    plt.ylim(image.shape[1]/zoom, image.shape[0]*(zoom-1)/zoom)

    # Colorbar
    cbar = plt.colorbar()
    # cbar.set_label('mK', rotation=270, labelpad=20)

    # Grid
    plt.grid(color='white', ls='solid', linewidth=0.5)

    from astropy import units as u
    lon = ax.coords[0]
    lat = ax.coords[1]
    lat.set_ticks(spacing=spacing*u.arcmin)
    lon.set_ticks(spacing=spacing*u.arcmin)

    # Labels
    if plot_xlabel:
        plt.xlabel('Galactic Longitude')
    else:
        lon = ax.coords[0]
        lon.set_axislabel(r'$\,$')


    lat = ax.coords[1]
    lat.set_axislabel(r'$\,$')
    lat.set_ticklabel_visible(False)
    #plt.ylabel('Galactic Latitude')


    plt.title(f'{band}-Band Beam S/N ['+r'$\sigma$]')

    # Plot contours
    ax.contour(contourdata, colors='black', alpha=0.8, linewidths=0.5)

    # Plot locations of clouds
    if not dark_clouds is None:
        for cloud_loc in dark_clouds['locs']:
            pix_x, pix_y = wcs.wcs_world2pix(cloud_loc[0], cloud_loc[1], 0)
            ax.plot(pix_x, pix_y, '^', color='k')


    # Mark centre of observation
    pix_x1, pix_y1 = wcs.wcs_world2pix(central_coords[0], central_coords[1], 0)
    ax.plot(pix_x1, pix_y1, 'x', color='k')

    # Plot circle around Ku observations
    circle_x, circle_y = circle_in_worldcoords2pix(wcs=wcs, center=central_coords, radius=0.4, npoints=360)
    ax.plot(circle_x, circle_y, 'k--', linewidth=0.9)

    # Save figure
    plt.savefig(f'./simresults/{title}_s2n.png', bbox_inches='tight', pad_inches=0, dpi=600)
    plt.savefig(f'./simresults/{title}_s2n.pdf', bbox_inches='tight', pad_inches=0, dpi=600)

    return



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
