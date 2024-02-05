import numpy as np
from matplotlib import pyplot
import Coordinates
import h5py
import healpy as hp
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import astropy.units as u
from Observations import *
from Observatory import * 
from ReceiverModes import *
from Mapping import Mapper, MapperHPX
from tqdm import tqdm
import Scheduler


tsys = 1400
sample_rate = 50

from concurrent import futures
from multiprocessing import Pool, Process, RawArray, shared_memory, managers, Queue

var_dict = {}
shm_data = {'shm_list':{}}

def create_shared_memory(rcvrmodes):
    
    # Save the big datasets to global variables, then delete them from the objects
    for rcvrmode in rcvrmodes:
        if hasattr(rcvrmode,'data'):
            X_shape  = rcvrmode.data.shape
            X_type   = rcvrmode.data.dtype
            var_dict['shape_{}'.format(rcvrmode.__name__)] = X_shape
            var_dict[ 'type_{}'.format(rcvrmode.__name__)] = X_type

            shm = shared_memory.SharedMemory(create=True,size=rcvrmode.data.nbytes,name=rcvrmode.__name__)
            np_array = np.ndarray(rcvrmode.data.shape, dtype=rcvrmode.data.dtype, buffer=shm.buf)
            np_array[:] = rcvrmode.data[:]
            var_dict['data_{}'.format(rcvrmode.__name__)] = shm.name
            rcvrmode.data = None
            shm_data['shm_list'][rcvrmode.__name__] = shm
    #return  shm_list

def clear_shared_memory(mode):
    
    X_shape = var_dict['shape_{}'.format(mode)]
    X_type  = var_dict[ 'type_{}'.format(mode)]
    shm  = shared_memory.SharedMemory(name=mode)
    data = np.ndarray(X_shape, dtype=X_type, buffer=shm.buf) 
    data *= 0
    shm.close()

def get_shared_memory(name):

    X_shape = var_dict['shape_{}'.format(name)]
    X_type  = var_dict[ 'type_{}'.format(name)]
    existing_shm  = shared_memory.SharedMemory(name=shm_data['shm_list'][name].name)
    mdata = copy.copy(np.ndarray(X_shape, dtype=X_type, buffer=existing_shm.buf))
    existing_shm.close()
    return mdata


def RunSimulation():

    # Generate telescope feed positions
    feed_positions = np.loadtxt('../../COMAP/runcomapreduce/COMAP_FEEDS.dat',usecols=[1,2])*0.1853/60.
    feed_pos_rot = feed_positions[:19,::-1]
    
    print('Building Telescope')
    frequencies = [27,29,31,33]
    telescope = Telescope(sample_rate=sample_rate,lons=[-118.28222222],lats=[37.23388861],feedpositions=feed_pos_rot, frequencies=frequencies)

    # Define receiver modes
    mapmodes = [FFHIPASS(filename='SkyMaps/HIPASS_ZOA_FF_1_1024.fits'),\
                AMEPlanck857(filename='SkyMaps/HFI_SkyMap_857-field-Int_2048_R3.00_full_5arcmin.fits'),\
                CGPShpx(filename='SkyMaps/CGPS_VGPS_CONT_1_2048_partial_5arcmin.fits')]
    rcvrmodes = [WhiteNoise(tsys=tsys, sample_rate=1./sample_rate, band_width=2e9,k=np.sqrt(2))] + mapmodes


    # Define write modes
    nxpix = 400 #1024
    nypix = 100 #512
    cdelt = 5 # arcmin
    crval = [36.25,0]
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



    return telescope, rcvrmodes, writemodes, noisefree_map

def define_observations(schedules, telescope, rcvrmodes, writemodes):

    # Now we may have a list of targets 
    print('Assigning Schedules to Receivers')
    observations = Observations(telescope(), 
                                schedules,
                                rcvrmodes,
                                writemodes)

    return observations

# def run_feeds(observations, mapper):

#     # This loop would be parallelised
#     rot = hp.rotator.Rotator(coord=['C','G'])
#     for i, feed in enumerate(tqdm(observations())):
#         feed()
#         theta, phi = rot((90-feed.dec)*np.pi/180., feed.ra*np.pi/180.)
#         mapper.accumulateHits(phi*180./np.pi, (np.pi/2.-theta)*180./np.pi)
#         mapper.accumulateSignal(phi*180./np.pi, (np.pi/2.-theta)*180./np.pi, feed.tod[0,:])

#     return mapper.wcs, np.reshape(mapper.signal,(mapper.nypix,mapper.nxpix)), np.reshape(mapper.hits,(mapper.nypix,mapper.nxpix))


def proc_func(feeds, i0, i1):

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

                
        feeds[i]()
        feeds[i].clear()

        for existing_shm in existing_shms:
            existing_shm.close()

    return feeds

import time
def run_feeds(observations):

    
    # Clear out the shared map data for each run
    clear_shared_memory('NaiveMapper')

    print('Parallel Loop')
    feeds = observations()
    t0 = time.time()
    nproc = 1
    processes = []
    select = np.sort(np.mod(np.arange(len(feeds)),nproc))
    begin_end = [np.where((select == f))[0][[0,-1]] for  f in range(nproc)]
    print(begin_end)
    step = len(feeds)//nproc
    
    for i in range(nproc):
        _process = Process(target=proc_func,args=(feeds,begin_end[i][0],begin_end[i][1],))#i*step,(i+1)*step,))
        processes.append(_process)
        _process.start()

    for process in processes:
        process.join()


    # return the mapper 
    return feeds[0].writemodes[0]
    

if __name__ =="__main__":
    

    
    
    #SchedulePositions()

    telescope, rcvrmodes, writemodes, noise_free = RunSimulation()
    obs = []
    start = Time('2020-02-01T00:00:00.000',format='isot')
    longitudes = [22.5 + i*48/60. for i in range(6)]
    longitudes = longitudes[0:10]
#range(34)] #[20,21.25] #22.5,23.75,25,26.25,27.5,28.75,30,31.25,32.5,33.75,35,36.25,37.5,38.75,40]
    observingMode = AzElLissajous(az_radius=48/60., el_radius=48/60,az_slew_speed=0.45, el_slew_speed=0.45)
    schedules = []
    for icen, gl in enumerate(longitudes):

        scheduler_setting = Scheduler.Scheduler(Time('2019-01-01T00:00:00.000',format='isot'),
                                                SkyCoord(gl*u.deg,0*u.deg,frame='galactic').icrs,
                                                telescope.lons[0], telescope.lats[0],
                                                observingMode = observingMode,
                                                nRepointings=4,
                                                nDays = 1,
                                                source_lha0=3,
                                                minElevation=0,
                                                span_ha = 3)
        
        schedules += scheduler_setting.schedules

    observation = define_observations(schedules, telescope, rcvrmodes, writemodes)
    
    # We allocate the shared memory after initially creating receiver datasets
    create_shared_memory(rcvrmodes)
    create_shared_memory(writemodes)

    
    h = []
    s = []

    # Loop over the full survey 60 times
    nRepeats = 1
    for i in range(nRepeats):
        mapper = run_feeds(observation)

        mapdata = get_shared_memory(mapper.__name__)
        if hasattr(mapper,'wcs'):
            wcs = mapper.wcs
            signal = np.reshape(mapdata[0,:],(mapper.nypix,mapper.nxpix))
            hits   = np.reshape(mapdata[1,:],(mapper.nypix,mapper.nxpix))
        else:
            signal = mapdata[0,:]
            hits   = mapdata[1,:]
        h += [hits]
        s += [signal]
        hall = np.array(h)
        sall = np.array(s)

        rms = np.sqrt(2)*tsys/np.sqrt(np.sum(hall,axis=0)/sample_rate*1e9)

        if mapper.__name__ == 'NaiveMapper':
            from astropy.io import fits
            hdu = fits.PrimaryHDU((np.sum(sall,axis=0)/np.sum(hall,axis=0)-tsys)*1e3, header=wcs.to_header())
            hdu_hits = fits.ImageHDU(np.sum(hall,axis=0)/sample_rate, header=wcs.to_header())
            hdu_rms = fits.ImageHDU(rms, header=wcs.to_header())
            hdu_noise_free = fits.ImageHDU(noise_free[0,:], header=wcs.to_header())
            hdul = fits.HDUList([hdu,hdu_hits,hdu_rms,hdu_noise_free])
            hdul.writeto('outputs/circ_chunk1{:03d}.fits'.format(i),overwrite=True)
        elif mapper.__name__ =='NaiveMapperHPX':
            maps = [(np.sum(sall,axis=0)/np.sum(hall,axis=0)-tsys)*1e3,\
                    np.sum(hall,axis=0)/sample_rate,\
                    rms,\
                    noise_free[0,:]]
            hp.write_map('outputs/hpx_new{:03d}.fits'.format(i),maps,overwrite=True)
    for k,shm in shm_data['shm_list'].items():
        print('unlink',k)
        shm.close()
        shm.unlink() # free memory
