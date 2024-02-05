import numpy as np

def returnfiledirs(datadir,session,bank):
    ''' Returns the data directories '''
    fitdir   = datadir + f'/AGBT20B_336_0{session}.raw.vegas/AGBT20B_336_0{session}.raw.vegas.{bank}.fits'
    indexdir = datadir + f'/AGBT20B_336_0{session}.raw.vegas/AGBT20B_336_0{session}.raw.vegas.{bank}.index'
    flagdir  = datadir + f'/AGBT20B_336_0{session}.raw.vegas/AGBT20B_336_0{session}.raw.vegas.{bank}.flag'
    session_full_name = f'AGBT20B_336_0{session}_BANK_{bank}'
    return fitdir, indexdir, flagdir, session_full_name



def openfits(fitdir, hdu_number, verbose=True):
    '''Opens fits file'''

    from astropy.io import fits

    # Open fits file
    hdul = fits.open(fitdir)
    data = hdul[hdu_number].data

    # Prints info
    if verbose:
        print(hdul.info())
        print(hdul[hdu_number].header)

    return data


def getdata(datadir, session, bank):
    ''' Simple way to read in the data '''

    # Get directory addresses
    fitdir, indexdir, flagdir, session_full_name = returnfiledirs(datadir,session,bank)

    # Open data
    data = openfits(fitdir,hdu_number='SINGLE DISH', verbose=False)

    return data


def AltAz2Galactic(time, alt, az):
    ''' Converts Horizon to Galactic coordinates:
        all inputs and outputs are in degrees '''

    from astropy.coordinates import AltAz
    from astropy.coordinates import Galactic
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy import units as u

    # Location of the telescope
    GBT = EarthLocation(lat=38.43312*u.deg, lon=-79.83983*u.deg, height=8.245950E+02*u.m)

    # Convert times to astropy object
    def custom_date_format_to_isot(timestamps):
        timestamps = timestamps.replace('_', '-', 2)
        timestamps = timestamps.replace('_', 'T', 1)
        return timestamps

    timestamps_isot = custom_date_format_to_isot(time)
    observing_time = Time(timestamps_isot, format='isot', scale='utc')

    # Initialize astropy coordinates
    horizon_coordinates = SkyCoord(alt=alt*u.deg, az=az*u.deg, location=GBT, obstime=observing_time, frame='altaz')
    galactic_coordinates = horizon_coordinates.transform_to(Galactic())

    # Extract l,b
    l = galactic_coordinates.l.value
    b = galactic_coordinates.b.value

    return l,b


def load_galactic_coordinates(datadir, session, data):
    ''' Creates Galactic coordinates and saves them to a fileself.
    If the file already exists, it just reads them in '''

    import os.path
    gal_coords_file = f'{datadir}/AGBT20B_336_0{session}.raw.vegas/AGBT20B_336_0{session}_galactic_coordinates.txt'
    if os.path.isfile(gal_coords_file): # If file with l & b exists, then skip conversion step and just load l and b
        print(f'Galactic coordinates found for session {session}, reading them in.')
        l, b = np.loadtxt(gal_coords_file)

    else: # If no l & b are found, then create them and save them
        print(f'No file for Galactic coordinates found for session {session}, creating them.')
        l, b = AltAz2Galactic(time=data.field('DATE-OBS'), alt=data.field('ELEVATIO'), az=data.field('AZIMUTH'))
        np.savetxt(gal_coords_file, [l, b])

    return l, b


def plot_data_simple(x_axis_name, y_axis_name, xlabel, ylabel, title, savename=None, show=False):
    ''' Function to plot a column of data against another '''

    plt.plot(data.field(f'{x_axis_name}'), data.field(f'{y_axis_name}'),',')
    plt.title(title + f' [{session_full_name}]')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if savename is not None:
        plt.savefig(savename+'.png')
        plt.savefig(savename+'.pdf')

    if show:
        plt.show()
    return


def read_galactic_coordinates(datadir, band, object_name='daisy_center'):
    ''' Returns full l, b on a given source if the files already exist
    for galactic coordinates. Bands can be 'C' and 'Ku' '''

    sessions = {'C': [4],
               'Ku': [1,2,3]}

    # Concatenate coordinates
    import os.path
    l = []
    b = []
    for session in sessions[band]:
        data = getdata(datadir, session, bank='A') # we are only interested in coordinates, so I set bank to A
        gal_coords_file = f'{datadir}/AGBT20B_336_0{session}.raw.vegas/AGBT20B_336_0{session}_galactic_coordinates.txt'
        if os.path.isfile(gal_coords_file): # If file with l & b exists, then skip conversion step and just load l and b
            print(f'Galactic coordinates found for session {session}, reading them in.')
            l_temp, b_temp = np.loadtxt(gal_coords_file)
        l.append(l_temp[data.field('OBJECT')==object_name])
        b.append(b_temp[data.field('OBJECT')==object_name])

    return l[0],b[0]


def calculate_cdelt(band, daisy_radius, nyquist):
    ''' Returns Nyquist pixel size in armin '''

    # Pixelization: nyquist is pixels per FWHM beamwidt
    x_extent = daisy_radius*2 # deg
    y_extent = daisy_radius*2 # deg

    # Define write modes
    if band == 'C':
        beam_FWHM_arcmin = 2.54
    elif band == 'Ku':
        beam_FWHM_arcmin = 0.90

    # Get Nyquist spacing
    nxpix = np.round(x_extent/(beam_FWHM_arcmin/60.)*nyquist) # nyquist is the number of pix/beam
    nypix = np.round(y_extent/(beam_FWHM_arcmin/60.)*nyquist) # nyquist is the number of pix/beam
    cdelt = beam_FWHM_arcmin/nyquist # Nyquist pixel size in arcmin

    return cdelt, nxpix



def plot_heatmap(array1, array2, label1, label2, res=50, log10=False, title=None, savename=None, xlim=None, ylim=None):
    ''' Plots a heatmap of two variables '''

    import matplotlib.pyplot as plt

    # Set resolution
    res = np.round(res)
    res = np.int_(res)

    def discard_common_nans(array1, array2):
        array1nonans = array1[(~np.isnan(array1) & ~np.isnan(array2))]
        array2nonans = array2[(~np.isnan(array1) & ~np.isnan(array2))]
        return array1nonans, array2nonans

    def discard_outside_limit(array1, array2):
        array1withinlimit = array1[(array1>xlim[0]) & (array1<xlim[1]) & (array2>ylim[0]) & (array2<ylim[1])]
        array2withinlimit = array2[(array1>xlim[0]) & (array1<xlim[1]) & (array2>ylim[0]) & (array2<ylim[1])]
        return array1withinlimit, array2withinlimit

    # Discard NaNs and discard anything outside the x and y limits
    if xlim is not None and ylim is not None:
        array1, array2 = discard_outside_limit(array1, array2)
    array1nonans, array2nonans = discard_common_nans(array1, array2)

    heatmap, xedges, yedges = np.histogram2d(array1nonans, array2nonans, bins=res)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


    if log10:
        plt.imshow(np.log10(heatmap.T), extent=extent, origin='lower', zorder = 4)
    else:
        plt.imshow(heatmap.T, extent=extent, origin='lower', zorder = 4)

    plt.xlabel(label1)
    plt.ylabel(label2)
    if title is not None:
        plt.title(title+' [log10]')
    plt.colorbar()

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if savename is not None:
        plt.savefig(f'{savename}', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.axis('square')

    plt.show()
    plt.close()
    return



def plot_hits(l_Ku, b_Ku, l_C, b_C, pix_per_beam):
    ''' Make a hit map '''

    # Basic parameters
    central_coords = [192.41, -11.51]
    rad_Ku = 0.4*1.1 # to set x and y limits
    rad_C = 1.5*1.1 # to set x and y limits

    # Calculate optimum gridding
    cdelt_C, npix_C = calculate_cdelt(band='C', daisy_radius=1.5, nyquist=pix_per_beam)
    cdelt_Ku, npix_Ku = calculate_cdelt(band='Ku', daisy_radius=0.4, nyquist=pix_per_beam)
    print(f'Using {cdelt_C:.2f} arcmin pixels for C ({npix_C:.0f} pix), and {cdelt_Ku:.2f} arcmin pixels for Ku ({npix_Ku:.0f} pix).')

    # Plot the heatmaps
    plot_heatmap(array1=l_C, array2=b_C, label1='Galactic Longitude (deg)', label2='Galactic Latitude (deg)', title=f'C-Band Hits: {pix_per_beam} pix/beam', res=npix_C, log10=True,     savename=f'./figures/cband_hits_{pix_per_beam}.png', xlim=[central_coords[0]-rad_C, central_coords[0]+rad_C], ylim=[central_coords[1]-rad_C, central_coords[1]+rad_C])
    plot_heatmap(array1=l_Ku, array2=b_Ku, label1='Galactic Longitude (deg)', label2='Galactic Latitude (deg)', title=f'Ku-Band Hits: {pix_per_beam} pix/beam', res=npix_Ku, log10=True, savename=f'./figures/kuband_hits_{pix_per_beam}.png', xlim=[central_coords[0]-rad_Ku, central_coords[0]+rad_Ku], ylim=[central_coords[1]-rad_Ku, central_coords[1]+rad_Ku])

    return


def lst_calculator(lon, utc_time):
    ''' lon in degrees, utc_time  as 'MMDDYY HHMM' '''

    #Only valid for dates between 1901 and 2099. Accurate to within 1.1s.
    Long = lon      #Longitude of location in question (BMX LAT = 40.869 [40deg 52' 8"], BMX LONG = -72.866 [-72deg 51' 57"], Custer LONG = -72.435)

    #Calculate longitude in DegHHMM format for edification of user:
    hemisphere = 'W'
    if Long > 0:        #if the number is positive it's in the Eastern hemisphere
        hemisphere = 'E'
    LongDeg = int(Long)
    LongMin = (Long - int(Long))*60
    LongSec = (LongMin - int(LongMin))*60
    LongMin = int(LongMin)
    LongSec = int(LongSec)

    #TD = raw_input('\nEnter the UTC time and date as MMDDYY HHMM. (UTC = EST+5, EDT+4):\n')
    TD=utc_time

    #split TD into individual variables for month, day, etc. and convert to floats:
    MM = float(TD[0:2])
    DD = float(TD[2:4])
    YY = float(TD[4:6])
    YY = YY+2000
    hh = float(TD[7:9])
    mm = float(TD[9:11])

    # Convert mm to fractional time:
    mm = mm/60.

    # Reformat UTC time as fractional hours:
    UT = hh+mm

    # Calculate the Julian date:
    JD = (367*YY) - int((7*(YY+int((MM+9)/12)))/4) + int((275*MM)/9) + DD + 1721013.5 + (UT/24)

    #calculate the Greenwhich mean sidereal time:
    GMST = 18.697374558 + 24.06570982441908*(JD - 2451545)
    GMST = GMST % 24    #use modulo operator to convert to 24 hours
    GMSTmm = (GMST - int(GMST))*60          #convert fraction hours to minutes
    GMSTss = (GMSTmm - int(GMSTmm))*60      #convert fractional minutes to seconds
    GMSThh = int(GMST)
    GMSTmm = int(GMSTmm)
    GMSTss = int(GMSTss)
    #print('\nGreenwhich Mean Sidereal Time: %s:%s:%s' %(GMSThh, GMSTmm, GMSTss))

    #Convert to the local sidereal time by adding the longitude (in hours) from the GMST.
    #(Hours = Degrees/15, Degrees = Hours*15)
    Long = Long/15.      # Convert longitude to hours
    LST = GMST+Long     #Fraction LST. If negative we want to add 24...
    if LST < 0:
        LST = LST +24
    LSTmm = (LST - int(LST))*60          #convert fraction hours to minutes
    LSTss = (LSTmm - int(LSTmm))*60      #convert fractional minutes to seconds
    LSThh = int(LST)
    LSTmm = int(LSTmm)
    LSTss = int(LSTss)

    LST_deg = LSThh+LSTmm/60.+LSTss/3600.

    return LST_deg


def schedule(datadir,session,saveplot):
    ''' Plots the schedule of a session '''

    # Calculate sidereal time of object
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy.time import Time
    from astropy import units as u
    import matplotlib.pyplot as plt

    # Create date range
    def datetime_linspace(start, end):
        ''' Create a range of dates separated by one minute '''
        import pandas as pd
        start = start[:-5]+'00.00' # floor start to the nearest minute
        start = pd.Timestamp(start) # timestamps are in nanoseconds
        end = pd.Timestamp(end)
        nminutes = np.floor((end.value-start.value)/(60*1e9))+1 # number of one minute steps
        dates = np.arange(nminutes)*60*1e9 + start.value
        dates = pd.to_datetime(dates)
        return dates

    # Get date range
    data = getdata(datadir, session, bank='A') # bank set to A since we only care about coordinates
    dates = datetime_linspace(start=data.field('DATE-OBS')[0], end=data.field('DATE-OBS')[-1])

    # Calculate LST
    def LST_wrapper(gbt_lon, dates):
        ''' Calls the LST function in the right format '''
        LST = []
        for date in dates:
            YY = str(int(str(date).split('-')[0])-2000)
            MM = str(date).split('-')[1]
            DD = str(date).split('-')[2].split(' ')[0]
            hh = str(date).split(' ')[1].split(':')[0]
            mm = str(date).split(' ')[1].split(':')[1]
            LST.append(lst_calculator(lon=gbt_lon, utc_time=f'{MM}{DD}{YY} {hh}{mm}'))
        return LST

    gbt_lon = -79.83983
    LST = LST_wrapper(gbt_lon, dates)

    # Calculate elevation of objects of interest (target, 3C147 and 0531+...)
    def elevation_calculator(dates, source_name):
        ''' Get elevation of the source at a given date as seen from the GBT '''
        # Calibrator Coordinates
        catalogue_galactic = {'daisy_center': [192.41, -11.51],
                              '3C147':        [161.68636686748, +10.29833172063],
                              '0530+1331':    [191.36766147, -11.01194840]}
        calibrator_coords = catalogue_galactic[source_name]
        # Set location
        observing_location = EarthLocation(lat=38.43312*u.deg, lon=-79.83983*u.deg, height=8.245950E+02*u.m)
        # Get elevation of object
        coords = SkyCoord(l=calibrator_coords[0]*u.degree,b=calibrator_coords[1]*u.degree, frame='galactic')
        # Get elevation
        times = Time(dates, scale='utc')
        coordsaltaz = coords.transform_to(AltAz(obstime=times,location=observing_location))
        elevation = coordsaltaz.alt.degree
        return elevation

    # Plot elevations in a loop
    fig, ax = plt.subplots(figsize=(8*1.3,3.2*1.3))
    plt.title(f'Session {session} Schedule')
    plt.ylim([10,90])
    plt.xlim([np.nanmin(LST), np.nanmax(LST)])
    plt.xlabel('LST (h)')
    plt.ylabel('Elevation (deg)')
    colours = {'daisy_center': 'b',
               '3C147':        'r',
               '0530+1331':    'g'}

    for source in ['daisy_center', '3C147', '0530+1331']:

        # Find the boundaries of each observation on the source
        def find_boundaries(source):
            instances = np.where(data.field('OBJECT')==source)[0]
            boundaries_start = instances[np.where(np.r_[True, instances[1:] > instances[:-1]+1])[0]]  # find changes
            # Now that we know where new names start, find the previous instance and mark it as an end
            boundaries_end = instances[np.where(np.r_[True, instances[1:] > instances[:-1]+1])[0]-1]
            # Sort the arrays
            boundaries_start = np.sort(boundaries_start)
            boundaries_end = np.sort(boundaries_end)
            # Add the first and last instances here
            boundaries_start = np.insert(boundaries_start, 0, instances[0]) # also concatenate the first instance
            boundaries_end = np.insert(boundaries_end, -1, instances[-1]) # also concatenate the last instance
            # Remove any repeated items
            boundaries_start = np.unique(boundaries_start)
            boundaries_end = np.unique(boundaries_end)
            return boundaries_start, boundaries_end

        boundaries_start, boundaries_end = find_boundaries(source)

        # Find the LSTs of those boundaries
        def LST_wrapper2(gbt_lon, dates):
            ''' Calls the LST function in the right format '''
            LST = []
            for date in dates:
                YY = str(int(str(date).split('-')[0])-2000)
                MM = str(date).split('-')[1]
                DD = str(date).split('-')[2].split('T')[0]
                hh = str(date).split('T')[1].split(':')[0]
                mm = str(date).split('T')[1].split(':')[1]
                LST.append(lst_calculator(lon=gbt_lon, utc_time=f'{MM}{DD}{YY} {hh}{mm}'))
            return LST

        boundary_LSTs_start = LST_wrapper2(gbt_lon, data.field('DATE-OBS')[boundaries_start])
        boundary_LSTs_end = LST_wrapper2(gbt_lon, data.field('DATE-OBS')[boundaries_end])

        # Plot transparent boxes
        for start, end in zip(boundary_LSTs_start, boundary_LSTs_end):
            ax.axvspan(start, end, -90, 90, alpha=0.3, color=colours[source], lw=0, zorder=0)

        # Plot elevation profile
        elevation = elevation_calculator(dates, source_name=source)
        plt.plot(LST, elevation, colours[source], linewidth=1.5, label=source, zorder=1)

    plt.legend(loc='upper right')

    if saveplot:
        plt.savefig(f'./figures/schedule_session{session}.png')
        plt.savefig(f'./figures/schedule_session{session}.pdf')

    plt.show()

    return



# # Tried but failed to modify the fits file
# def add2fits(fitdir, field, data, hdu_number=0):
#     ''' Adds new data to the fits file '''
#
#     from astropy.io import fits
#     hdul = fits.open(fitdir)
#     hdul[hdu_number].header[field] = data
#     print('Overwritting data to add new field...')
#     hdul.writeto(fitdir, overwrite=True)
#     print(f'Field {field} with value {data} has been added to hdu number {hdu_number}!')
#
#     print(hdul[hdu_number].header)
#
#
#     return
#
# # add2fits(fitdir, field='ROKETEST', data='IT WORKS', hdu_number=2)
#
#
# # Now open and print
# data = openfits(fitdir,hdu_number=2, verbose=True)
# print(header.field('ROKECE'))
# asdasdasd
