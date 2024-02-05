import numpy as np
from astropy import wcs
from matplotlib import pyplot
import h5py
from scipy.interpolate import interp1d
import os
import healpy as hp

from tqdm import tqdm 
from matplotlib.patches import Ellipse

from sdosim.Tools import binFuncs


#from Utilities import Source, sources

class NormaliseFilter:
    def __init__(self,**kwargs):
        pass

    def __call__(self,DataClass, tod, **kwargs):
        rms = np.nanstd(tod[1:tod.size//2*2:2] - tod[0:tod.size//2*2:2])
        tod = (tod - np.nanmedian(tod))/rms # normalise
        return tod

class AtmosphereFilter:
    def __init__(self,**kwargs):
        pass

    def __call__(self,DataClass, tod, **kwargs):
        feed = kwargs['FEED']
        el   = DataClass.el[feed,:]
        mask = DataClass.atmmask

        gd = (np.isnan(tod) == False) & (mask == 1)
        try:
            # Calculate slab
            A = 1./np.sin(el*np.pi/180.)
            # Build atmospheric model
            pmdl = np.poly1d(np.polyfit(A[gd],tod[gd],1))
            # Subtract atmospheric slab
            tod -= pmdl(A)

            # Bin by elevation, and remove with interpolation (redundant?) 
            binSize = 12./60.
            nbins = int((np.nanmax(el)-np.nanmin(el) )/binSize)
            elEdges= np.linspace(np.nanmin(el),np.nanmax(el),nbins+1)
            elMids = (elEdges[:-1] + elEdges[1:])/2.
            s = np.histogram(el[gd], elEdges, weights=tod[gd])[0]
            w = np.histogram(el[gd], elEdges)[0]
            pmdl = interp1d(elMids, s/w, bounds_error=False, fill_value=0)
            tod -= pmdl(el)
            tod[el < elMids[0]] -= s[0]/w[0]
            tod[el > elMids[-1]] -= s[-1]/w[-1]
        except TypeError:
            return tod 


        return tod
  
class Mapper:

    def __init__(self, 
                 crval=None, 
                 cdelt=[1.,1.], 
                 crpix=[128,128],
                 ctype=['RA---TAN','DEC--TAN']):

        # Cdelt given in arcmin
        self.crval = crval
        self.cdelt = [cd/60. for cd in cdelt]
        self.crpix = crpix
        self.ctype = ctype

        self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)

        self.nxpix = int(crpix[0]*2)
        self.nypix = int(crpix[1]*2)

        # Data containers
        self.data = np.zeros((2,self.nxpix*self.nypix)) 
        
        #self.hits_local   = np.zeros(self.nxpix*self.nypix)
        #self.signal_local = np.zeros(self.nxpix*self.nypix)

        self.rot = hp.rotator.Rotator(coord=['C','G'])

        self.__name__ = 'NaiveMapper'

    def __call__(self,feed):

        theta, phi = self.rot((90-feed.dec)*np.pi/180., feed.ra*np.pi/180.)
        self.accumulateHits(phi*180./np.pi, (np.pi/2.-theta)*180./np.pi)
        self.accumulateSignal(phi*180./np.pi, (np.pi/2.-theta)*180./np.pi, feed.tod[0,:])

    def write_shared_memory(self):
        
        self.data[0,:] += self.signal_local
        self.data[-1,:] += self.hits_local
        
                
    def getFlatPixels(self, x, y):
        """
        Convert sky angles to pixel space
        """
        if isinstance(self.wcs, type(None)):
            raise TypeError( 'No WCS object declared')
            return
        else:
            pixels = self.wcs.wcs_world2pix(x+self.wcs.wcs.cdelt[0]/2.,
                                            y+self.wcs.wcs.cdelt[1]/2.,0)
            pflat = (pixels[0].astype(int) + self.nxpix*pixels[1].astype(int)).astype(int)
            

            # Catch any wrap around pixels
            pflat[(pixels[0] < 0) | (pixels[0] > self.nxpix)] = -1
            pflat[(pixels[1] < 0) | (pixels[1] > self.nypix)] = -1

            return pflat

    def setWCS(self, crval, cdelt, crpix, ctype):
        """
        Declare world coordinate system for plots
        """
        self.wcs = wcs.WCS(naxis=2)
        self.wcs.wcs.crval = crval
        self.wcs.wcs.cdelt = cdelt
        self.wcs.wcs.crpix = crpix
        self.wcs.wcs.ctype = ctype

    def accumulateHits(self, x,y):
        """
        Generate sky maps
        """

        # Get pixels from sky coordinates
        pixels = self.getFlatPixels(x, y)
        binFuncs.binValues(self.data[-1,:], pixels)

    def accumulateSignal(self, x,y, tod):
        """
        Generate sky maps
        """

        # Get pixels from sky coordinates
        pixels = self.getFlatPixels(x, y)
        binFuncs.binValues(self.data[0,:], pixels, weights=tod)

    def setCrval(self):
        if isinstance(self.crval, type(None)):
            if self.source in sources:
                sRa,sDec = sources[self.source]()
                self.crval = [sRa,sDec]
            else:
                self.crval = [np.median(self.x[0,:]),
                              np.median(self.y[0,:])]

            

    def clear_shared_data(self,shm_data):

        # Second load the sahred memory needed for the output write modes
        for writemode in feeds[i].writemodes:
            if hasattr(writemode,'data'):
                X_shape = var_dict['shape_{}'.format(writemode.__name__)]
                X_type  = var_dict[ 'type_{}'.format(writemode.__name__)]
                existing_shm  = shared_memory.SharedMemory(name=shm_data['shm_list'][writemode.__name__].name)
                writemode.data = np.ndarray(X_shape, dtype=X_type, buffer=existing_shm.buf) 
                existing_shms += [existing_shm]


class MapperHPX:

    def __init__(self, nside):

        # map parameters
        self.nside = nside


        # Data containers
        self.data = np.zeros((2,12*self.nside**2)) 

        self.rot = hp.rotator.Rotator(coord=['C','G'])

        self.__name__ = 'NaiveMapperHPX'

    def __call__(self,feed):

        theta, phi = self.rot((90-feed.dec)*np.pi/180., feed.ra*np.pi/180.)
        self.accumulateHits(phi*180./np.pi, (np.pi/2.-theta)*180./np.pi)
        self.accumulateSignal(phi*180./np.pi, (np.pi/2.-theta)*180./np.pi, feed.tod[0,:])

        
                
    def getFlatPixels(self, x, y):
        """
        Convert sky angles to pixel space
        """

        return hp.ang2pix(self.nside, (90-y)*np.pi/180., x*np.pi/180.)


    def accumulateHits(self, x,y):
        """
        Generate sky maps
        """

        # Get pixels from sky coordinates
        pixels = self.getFlatPixels(x, y)
        binFuncs.binValues(self.data[-1,:], pixels)

    def accumulateSignal(self, x,y, tod):
        """
        Generate sky maps
        """

        # Get pixels from sky coordinates
        pixels = self.getFlatPixels(x, y)
        binFuncs.binValues(self.data[0,:], pixels, weights=tod)
            

    # def clear_shared_data(self,shm_data):

    #     # Second load the sahred memory needed for the output write modes
    #     for writemode in feeds[i].writemodes:
    #         if hasattr(writemode,'data'):
    #             X_shape = var_dict['shape_{}'.format(writemode.__name__)]
    #             X_type  = var_dict[ 'type_{}'.format(writemode.__name__)]
    #             existing_shm  = shared_memory.SharedMemory(name=shm_data['shm_list'][writemode.__name__].name)
    #             writemode.data = np.ndarray(X_shape, dtype=X_type, buffer=existing_shm.buf) 
    #             existing_shms += [existing_shm]
