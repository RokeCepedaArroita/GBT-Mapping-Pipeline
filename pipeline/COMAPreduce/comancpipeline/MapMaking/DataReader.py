import numpy as np
import h5py
from astropy import wcs
from matplotlib import pyplot
from tqdm import tqdm
import pandas as pd
from scipy import linalg as la
import healpy as hp
from comancpipeline.Tools import  binFuncs, stats

from comancpipeline.MapMaking import MapTypes, OffsetTypes


class ReadDataLevel2:

    def __init__(self, filelist, parameters,
                 ifeature=5,iband=0,ifreq=0,
                 keeptod=False,subtract_sky=False,**kwargs):


        # -- constants -- a lot of these are COMAP specific
        self.ifeature = ifeature
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.keeptod = keeptod
        self.iband = iband
        self.ifreq = ifreq

        # READ PARAMETERS
        self.offset_length = parameters['Destriper']['offset']

        self.Feeds  = [0]
        self.Nfeeds = 1


        title = parameters['Inputs']['title']
        self.output_map_filename = f'{title}.fits'


        # SETUP MAPS:
        crval = parameters['Destriper']['crval']
        cdelt = parameters['Destriper']['cdelt']
        crpix = parameters['Destriper']['crpix']
        ctype = parameters['Destriper']['ctype']
        nxpix = int(parameters['Destriper']['nxpix'])
        nypix = int(parameters['Destriper']['nypix'])

        self.naive  = MapTypes.FlatMapType(crval, cdelt, crpix, ctype,nxpix,nypix)


        # Will define Nsamples, datasizes[], and chunks[[]]
        for filename in filelist:
            self.countDataSize(filename)


        # Store the Time ordered data as required
        self.pixels = np.zeros(self.Nsamples,dtype=int)
        self.all_weights = np.zeros(self.Nsamples)
        if self.keeptod:
            self.all_tod = np.zeros(self.Nsamples)


        # First read in all the data
        # Remember we want to solve Ax = b,
        # "b" contains all the data, so we construct that now:
        # 1a) Create a naive binned map
        # 1b) Sum all the data into offsets
        # 2) Subtract the naive weighted map from the offsets
        # "b" residual vector is saved in residual Offset object
        self.Noffsets  = self.Nsamples//self.offset_length

        # Contains the difference between the TOD and the map average
        self.offset_residuals = OffsetTypes.Offsets(self.offset_length, self.Noffsets, self.Nsamples)

        for i, filename in enumerate(filelist):
            self.readPixels(i,filename)


        for i, filename in enumerate(filelist):
            self.readData(i,filename)

        #pyplot.subplot(projection=self.naive.wcs)
        #m = self.naive.get_map()
        #m[m==0]=np.nan
        #pyplot.imshow(m,aspect='auto')
        #pyplot.show()

        self.naive.average()

        self.offset_residuals.accumulate(-self.naive.sky_map[self.pixels],self.all_weights,[0,self.pixels.size])
        self.offset_residuals.average()

    def countDataSize(self,filename):
        """
        Opens each datafile and determines the number of samples

        Uses the features to select the correct chunk of data

        """

        # ROKE: REPLACE THIS WITH self.Nsamples = LEN(TOD)

        try:
            d = h5py.File(filename,'r')
        except:
            print(filename)
            return

        N = int((d['tod'].size//self.offset_length)*self.offset_length)
        d.close()

        N = N*self.Nfeeds # 2 for KU, 1 FOR C

        # Store the beginning and end point of each file
        self.chunks += [[int(self.Nsamples), int(self.Nsamples+N)]]

        # We also want to know how big each file is per feed
        self.datasizes += [int(N/self.Nfeeds)]

        # Finally, add to the total number of files
        self.Nsamples += int(N)


    def getTOD(self,i,d):
        """
        Want to select each feed and average the data over some frequency range
        """

        tod = d['tod'][:self.datasizes[i]] # this is the tod
        tod -= np.nanmedian(tod) # normalize tod
        weights = np.ones(np.shape(tod)) # can replace with np.ones
        # eventually get weights from 1/whitenmoise**2 per scan

        return tod, weights


    def readPixels(self, i, filename):
        """
        Reads data
        """


        d = h5py.File(filename,'r')

        # We store all the pointing information
        x  = d['ra'][:self.datasizes[i]]
        y  = d['dec'][:self.datasizes[i]]

        pixels = self.naive.getFlatPixels(x.flatten(),y.flatten())

        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = pixels.flatten()
        d.close()
        return


    def readData(self, i, filename):
        """
        Reads data
        """

        d = h5py.File(filename,'r')

        # Now accumulate the TOD into the naive map
        tod, weights     = self.getTOD(i,d)

        # Remove any bad data
        tod     = tod.flatten()
        weights = weights.flatten()
        bad = np.isinf(tod) | np.isnan(tod) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        offpix_chunk= self.offset_residuals.offsetpixels[self.chunks[i][0]:self.chunks[i][1]]
        bad_offsets = np.unique(offpix_chunk[bad])
        for bad_offset in bad_offsets:
            tod[offpix_chunk == bad_offset] = 0
            weights[offpix_chunk == bad_offset] = 0

        # Store TOD
        if self.keeptod:
            self.all_tod[self.chunks[i][0]:self.chunks[i][1]] = tod*1.
        self.all_weights[self.chunks[i][0]:self.chunks[i][1]] = weights

        # Bin data into maps
        self.naive.sum_data(tod,self.pixels[self.chunks[i][0]:self.chunks[i][1]],weights)

        # And then bin the data into the offsets vector
        self.offset_residuals.accumulate(tod,weights,self.chunks[i])
