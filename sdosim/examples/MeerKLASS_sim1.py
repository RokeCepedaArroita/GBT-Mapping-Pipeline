import numpy as np
from matplotlib import pyplot
from comancpipeline.Tools import Coordinates
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
from profile import profile 

tsys = 20
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
    feed_pos = [[0,0]]
    
    print('Building Telescope')
    frequencies = [1.0]
    telescope = Telescope(sample_rate=sample_rate,lons=[0],lats=[-30],feedpositions=feed_pos, frequencies=frequencies)

    # Define receiver modes
    mapmodes = []#HaslamAllSky(filename='SkyMaps/haslam408_dsds_Remazeilles2014.fits')]
    rcvrmodes = [WhiteNoise(tsys=tsys, sample_rate=1./sample_rate, band_width=100e6,k=np.sqrt(2)),\
                 PinkNoise(tsys=tsys, sample_rate=1/sample_rate, band_width=100e6,k=np.sqrt(2),alpha=1.2,knee=1)] + mapmodes


    # Define write modes
    writemodes = [MapperHPX(nside=512)]#crval=crval,cdelt=[cdelt,cdelt],crpix=[nxpix//2,nypix//2],ctype=['GLON-TAN','GLAT-TAN'])]

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


    # We allocate the shared memory after initially creating receiver datasets
    create_shared_memory(rcvrmodes)
    create_shared_memory(writemodes)

    return telescope, rcvrmodes, writemodes, noisefree_map

def define_observations(telescope, rcvrmodes, writemodes):

    start_date = Time('2019-01-01T00:00:00.000',format='isot')

    # Now we may have a list of targets 
    print('Creating Schedules')
    schedules = [AzRaster(az0=0,el0=30,az_radius=540,slew_speed=1)]

    length = schedules[0].length.sec
    NperDay = int(86400./length)

    print('Assigning Schedules to Receivers')
    observations = Observations(telescope(), 
                                schedules,
                                rcvrmodes,
                                writemodes,
                                start=start_date,
                                repeat=NperDay*360)

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

@profile
def proc_func(feeds, i0, i1):

    for i in range(i0,i1):
        existing_shms = []
        # First load the shared memory needed for the input data maps
        for rcvrmode in feeds[i].rcvrmodes:
            if hasattr(rcvrmode,'data'):
                X_shape = var_dict['shape_{}'.format(rcvrmode.__name__)]
                X_type  = var_dict[ 'type_{}'.format(rcvrmode.__name__)]
                existing_shm  = shared_memory.SharedMemory(name=shm_data['shm_list'][rcvrmode.__name__].name)
                rcvrmode.data = np.ndarray(X_shape, dtype=X_type, buffer=existing_shm.buf) 
                existing_shms += [existing_shm]

        # Second load the shared memory needed for the output write modes
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
    clear_shared_memory('NaiveMapperHPX')

    print('Parallel Loop')
    feeds = observations()
    t0 = time.time()
    nproc = 8
    processes = []
    select = np.sort(np.mod(np.arange(len(feeds)),nproc))
    begin_end = [np.where((select == f))[0][[0,-1]] for  f in range(nproc)]
    print(begin_end)
    step = len(feeds)//nproc
    
    t0 = time.time()
    for i in range(nproc):
        _process = Process(target=proc_func,args=(feeds,begin_end[i][0],begin_end[i][1],))#i*step,(i+1)*step,))
        processes.append(_process)
        _process.start()

    for process in processes:
        process.join()
    print(time.time()-t0)

    # return the mapper 
    return feeds[0].writemodes[0]
    

if __name__ =="__main__":
    

    
    
    #SchedulePositions()
    h = []
    s = []

    telescope, rcvrmodes, writemodes, noise_free = RunSimulation()
    # Loop over the full survey 60 times
    nRepeats = 1 
    obs = []
    for i in range(nRepeats):
        if i == 0:
            observation = define_observations(telescope, rcvrmodes, writemodes)
            obs += [{'observation':observation}]
        else:
            observation = obs[icen]['observation']

        mapper = run_feeds(observation)

        mapdata = get_shared_memory('NaiveMapperHPX')
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
        hdul.writeto('outputs/new{:03d}.fits'.format(i),overwrite=True)
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
