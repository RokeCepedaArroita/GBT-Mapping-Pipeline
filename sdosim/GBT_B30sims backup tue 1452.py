import numpy as np
from matplotlib import pyplot
import h5py
import healpy as hp
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm


from concurrent import futures
from multiprocessing import Pool, Process, RawArray, shared_memory, managers, Queue, Lock
import time


from sdosim.Scheduler import Scheduler
from sdosim.Observations.Observations import *
from sdosim.Observations.Observatory  import *
from sdosim.ReceiverModels.ReceiverModes  import *
from sdosim.Tools import Coordinates
from sdosim.WriteModes.Mapping import Mapper, MapperHPX
from sdosim.Tools.SharedMemoryTools import *

def SetupSimulation():
    """
    Setup the telescope, observing modes, mapper, etc...
    """
    print('Building Telescope')

    # Generate telescope feed positions
    frequencies = [4.85] # GHz
    tsys = 31 # K
    sample_rate = 50 # Hz
    band_width  = 1.25e9 # Hz

    # Central coords
    central_coords_rise = [192.4, -11.5]
    central_coords_set = [192.4, -11.5]

    source_lha0_set = 1. # start of set obs
    source_lha0_rise = -2. # start of rise obs

    # Do not know what these are
    cdelt = 0.5 # arcmin
    crval = [192.4, -11.5]

    # Scan
    daisy_radius = 0.5
    daisy_infinity_time = 10. # in seconds - cannot be faster than 60!!!
    observation_length_sec = 1200. # in seconds
    nDays = 4
    nRepointings = 1
    minElevation = -10000

    # Pixelization
    x_extent = daisy_radius*2 # deg
    y_extent = daisy_radius*2 # deg

    # Define write modes
    c_beam_FWHM_arcmin = 2.54
    ku_beam_FWHM_arcmin = 0.90

    if frequencies[0] == 4.85:
        nxpix = 600#np.round(x_extent/(c_beam_FWHM_arcmin/60.)*3) # 3 pix/beam
        nypix = 600#np.round(y_extent/(c_beam_FWHM_arcmin/60.)*3) # 3 pix/beam
    elif frequencies[0] == 13.7:
        nxpix = np.round(x_extent/(ku_beam_FWHM_arcmin/60.)*3) # 3 pix/beam
        nypix = np.round(y_extent/(ku_beam_FWHM_arcmin/60.)*3) # 3 pix/beam


    if frequencies[0] == 4.85:
        scale_factor = 7.0 # this is where the degradation factors come
    elif frequencies[0] == 13.7:
        scale_factor = 8.2 # this is where the degradation factors come


    # Define observation strategy and schedules:
    start_time = Time('2020-02-01T20:00:00.000',format='isot', scale='utc')
    observation_length = TimeDelta(observation_length_sec, format='sec')


    # Probably will not change below

    telescope_longitude = -(79+50/60.+23.406/3600.) # GBT
    telescope_latitude  = 38+25/60+59.236/3600. # GBT
    telescope = Telescope(tsys=tsys, sample_rate=sample_rate,lons=[telescope_longitude],lats=[telescope_latitude],feedpositions=[[0,0]], frequencies=frequencies)


    # Observing strategy
    observingMode = DaisyRaDec(r0=daisy_radius, tau=daisy_infinity_time, phi1=0, phi2=0) # tau is number of seconds for infinity sign, max used 30 for 1.5 deg scans, check

    print(start_time, flush=True)

    # Actual observations
    skycoord   = SkyCoord(central_coords_set[0]*u.deg,central_coords_set[1]*u.deg,frame='galactic').icrs
    scheduler_setting = Scheduler.Scheduler(start_time = start_time,
                                            skycoord=skycoord,
                                            longitude=telescope.lons[0],
                                            latitude =telescope.lats[0],
                                            observingMode = observingMode,
                                            nRepointings=nRepointings,
                                            nDays = nDays,
                                            source_lha0=central_coords_set,
                                            minElevation=minElevation,
                                            length=observation_length)



    skycoord = SkyCoord(central_coords_rise[0]*u.deg,central_coords_rise[1]*u.deg,frame='galactic').icrs
    scheduler_rising = Scheduler.Scheduler(start_time = scheduler_setting.end_time,
                                           skycoord=skycoord,
                                           longitude=telescope.lons[0],
                                           latitude =telescope.lats[0],
                                           observingMode = observingMode,
                                           nRepointings=nRepointings,
                                           nDays = nDays,
                                           source_lha0=source_lha0_rise,
                                           minElevation=minElevation,
                                           length=observation_length)

    # Add schedulers together
    scheduler = scheduler_setting + scheduler_rising
    print('only doing setting one schedule (the setting one)')


    # Define receiver modes
    mapmodes  = [RokeGBTModel(filename=f'skymaps/skymodel_{frequencies[0]}GHz_mK.fits')]
    rcvrmodes = [WhiteNoise(tsys=telescope.tsys, sample_rate=sample_rate, band_width=telescope.band_width,k=scale_factor)] + mapmodes
    writemodes = [Mapper(crval=crval,cdelt=[cdelt,cdelt],crpix=[nxpix//2,nypix//2],ctype=['GLON-TAN','GLAT-TAN'])]

    # Create the noiseless version of the maps:
    try:
        xpix, ypix = np.meshgrid(np.arange(nxpix), np.arange(nypix))
        pix_phi, pix_theta = writemodes[0].wcs.all_pix2world(xpix.flatten(),ypix.flatten(),0)
        pix_phi, pix_theta = pix_phi.reshape(xpix.shape), pix_theta.reshape(ypix.shape)
        noisefree_map = np.zeros([len(frequencies),ypix.shape[0],ypix.shape[1]])
        for mapmode in mapmodes:
            noisefree_map += mapmode.get_map(pix_phi*np.pi/180.,(90-pix_theta)*np.pi/180.,frequencies)
    except:
        noisefree_map = np.zeros([len(frequencies),12*writemodes[0].nside**2])


    return telescope, rcvrmodes, writemodes, noisefree_map, scheduler


def proc_func(feeds, i0, i1, lock=None):
    """
    Function for parallelised looping over feeds
    """

    for i in range(i0,i1):
        # First load the shared memory needed for the input data maps
        existing_shms = []

        for rcvrmode in feeds[i].rcvrmodes:
            if hasattr(rcvrmode,'data'):
                X_shape = var_dict['shape_{}'.format(rcvrmode.__name__)]
                X_type  = var_dict[ 'type_{}'.format(rcvrmode.__name__)]
                existing_shm  = shared_memory.SharedMemory(name=shm_data['shm_list'][rcvrmode.__name__].name)
                rcvrmode.data = np.ndarray(X_shape, dtype=X_type, buffer=existing_shm.buf)
                existing_shms += [existing_shm]

        # Second load the sahred memory needed for the output write modes
        for writemode in feeds[i].writemodes:
            if hasattr(writemode,'data'):
                X_shape = var_dict['shape_{}'.format(writemode.__name__)]
                X_type  = var_dict[ 'type_{}'.format(writemode.__name__)]
                existing_shm  = shared_memory.SharedMemory(name=shm_data['shm_list'][writemode.__name__].name)
                writemode.data = np.ndarray(X_shape, dtype=X_type, buffer=existing_shm.buf)
                existing_shms += [existing_shm]


        feeds[i](lock)
        feeds[i].clear()

        for existing_shm in existing_shms:
            existing_shm.close()

    return feeds


def run_feeds(observations, lock=None ,   nproc = 2):


    # Clear out the shared map data for each run
    clear_shared_memory('NaiveMapper')

    print('Parallel Loop')
    feeds = observations()
    t0 = time.time()
    processes = []
    select = np.sort(np.mod(np.arange(len(feeds)),nproc))
    begin_end = [np.where((select == f))[0][[0,-1]] for  f in range(nproc)]
    step = len(feeds)//nproc

    t0 = time.time()

    for i in range(nproc):
        _process = Process(target=proc_func,args=(feeds,begin_end[i][0],begin_end[i][1],shm_lock,))
        processes.append(_process)
        _process.start()

    for process in processes:
        process.join()

    print('Run Time: {}'.format(time.time()-t0))

    # return the mapper
    return feeds[0].writemodes[0]


if __name__ =="__main__":



    # Setup observations
    telescope, rcvrmodes, writemodes, noise_free, scheduler = SetupSimulation()
    observation = Observations(telescope(),
                               scheduler.schedules,
                               rcvrmodes,
                               writemodes)

    # We allocate the shared memory after initially creating receiver datasets
    create_shared_memory(rcvrmodes)
    create_shared_memory(writemodes)



    # Generate time-ordered data/maps:
    nRepeats = 1
    h = []
    s = []
    for i in range(nRepeats):

        # Creates the data (will clear maps when called)
        mapper = run_feeds(observation, shm_lock)

        # gets a copy of the current maps for output
        mapdata = get_shared_memory(mapper.__name__)
        if hasattr(mapper,'wcs'):
            wcs = mapper.wcs
            signal = np.reshape(mapdata[0,:],(mapper.nypix,mapper.nxpix))
            hits   = np.reshape(mapdata[1,:],(mapper.nypix,mapper.nxpix))
        else:
            signal = mapdata[0,:]
            hits   = mapdata[1,:]

        # combine hits and weighted signal
        h += [hits]
        s += [signal]
        hall = np.array(h)
        sall = np.array(s)

        rms = np.sqrt(2)*telescope.tsys/np.sqrt(np.sum(hall,axis=0)/telescope.sample_rate*telescope.band_width)

        if mapper.__name__ == 'NaiveMapper':
            from astropy.io import fits
            hdu = fits.PrimaryHDU((np.sum(sall,axis=0)/np.sum(hall,axis=0)-telescope.tsys)*1e3, header=wcs.to_header()) # Convert map to mK
            hdu_hits = fits.ImageHDU(np.sum(hall,axis=0)/telescope.sample_rate, header=wcs.to_header())
            hdu_rms = fits.ImageHDU(rms*1e3, header=wcs.to_header()) # Convert rms to mK
            hdu_noise_free = fits.ImageHDU(noise_free[0,:], header=wcs.to_header()) # Convert free rms to mK
            hdul = fits.HDUList([hdu,hdu_hits,hdu_rms,hdu_noise_free])

            # Write fits
            hdul.writeto(f'gbt_outputs/daisy_repeats{i:03d}_{telescope.frequencies[0]}GHz.fits',overwrite=True)
            print(f'Saved gbt_outputs/daisy_repeats{i:03d}_{telescope.frequencies[0]}GHz.fits!')

    for k,shm in shm_data['shm_list'].items():
        print('unlink',k)
        shm.close()
        shm.unlink() # free memory
