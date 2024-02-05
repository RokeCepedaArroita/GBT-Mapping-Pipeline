import numpy as np
from matplotlib import pyplot
from sdosim.Tools import Coordinates
import h5py
import healpy as hp
from astropy.time import Time, TimeDelta
import copy
import time
import os

class ReceiverMode:

    def __init__(self):
        pass

    def clear(self):
        pass

class WhiteNoise(ReceiverMode):

    def __init__(self, tsys, sample_rate, band_width, k=1):

        self.rms = k * tsys/np.sqrt(band_width/sample_rate)
        self.tsys = tsys
        self.sample_rate = sample_rate
        self.band_width  = band_width

        self.__name__ = 'WhiteNoise'

        self.seed = None

    def __call__(self, feed):

        if isinstance(self.seed, type(None)):
            self.seed = int(time.time() + os.getpid())
            np.random.seed(self.seed)
        noise = np.random.normal(loc=self.tsys,scale=self.rms,size=(feed.nFrequencies, feed.nSamples))
        feed.tod +=  noise

class PinkNoise(ReceiverMode):

    def __init__(self, tsys, sample_rate, band_width, k=1, knee=1, alpha=1):

        self.rms = k * tsys/np.sqrt(sample_rate*band_width)
        self.sample_rate = sample_rate
        self.tsys = tsys
        self.knee = knee
        self.alpha = alpha

        self.seed = None

        self.__name__ = 'PinkNoise'

    def __call__(self,feed):


        if isinstance(self.seed, type(None)):
            self.seed = int(time.time() +os.getpid())
            np.random.seed(self.seed)

        nSamples_2bit = int(2**(np.log(feed.nSamples)/np.log(2)))

        nu = np.fft.fftfreq(nSamples_2bit, d=self.sample_rate)
        nu[0] = nu[1]
        ps = np.abs(self.knee/nu)**self.alpha
        ps[0] = ps[1]

        w = np.random.normal(loc=0,scale=self.rms,size=(feed.nFrequencies, nSamples_2bit))

        wf = np.fft.fft(w,axis=1)*np.sqrt(ps[np.newaxis,:])

        feed.tod += np.real(np.fft.ifft(wf,axis=1))[:,:feed.nSamples]

class SkyMap(ReceiverMode):

    def __init__(self,filename=None):
        self.filename = filename
        self.__name__ = 'SkyMap'

    def frequency_model(self,frequency):
        return 1

    def get_map(self,phi, theta, frequencies):

        output = []
        for i in range(len(frequencies)):
            output += [hp.get_interp_val(self.data,
                                         theta,
                                         phi) *self.frequency_model(frequencies[i])]

        return np.array(output)

class CGPShpx(SkyMap):

    def __init__(self, filename=None):
        """
        Expects the Canadian Galactic Plane Survey in Healpix format.
        Taken from the CADE IRAP database.
        """
        self.filename = filename
        self.data  = hp.read_map(filename)
        self.data[self.data == hp.UNSEEN] = 0
        self.nside = int(np.sqrt(self.data.size/12.))

        self.nu0 = 1.420 # GHz
        self.alpha = 2.7
        # The map is in Galactic coordinates so set up rotation
        self.gal_rot = hp.rotator.Rotator(coord=['C','G'])

        self.__name__ = 'CGPShpx'

    def frequency_model(self,frequency):
        return (self.nu0/frequency)**self.alpha


    def __call__(self, feed):
        """
        Simple power law extrapolation
        """

        theta, phi = self.gal_rot((90-feed.dec)*np.pi/180., feed.ra*np.pi/180.)
        for i in range(feed.tod.shape[0]):
            feed.tod[i,:] += hp.get_interp_val(self.data,
                                              theta,
                                              phi) * self.frequency_model(feed.frequencies[i])

class HaslamAllSky(SkyMap):

    def __init__(self, filename=None):
        """
        Expects the Canadian Galactic Plane Survey in Healpix format.
        Taken from the CADE IRAP database.
        """
        self.filename = filename
        self.data  = hp.read_map(filename)
        self.data[self.data == hp.UNSEEN] = 0
        self.nside = int(np.sqrt(self.data.size/12.))

        self.nu0 = 0.408 # GHz
        self.alpha = 3
        # The map is in Galactic coordinates so set up rotation
        self.gal_rot = hp.rotator.Rotator(coord=['C','G'])

        self.__name__ = 'CGPShpx'

    def frequency_model(self,frequency):
        return (self.nu0/frequency)**self.alpha


    def __call__(self, feed):
        """
        Simple power law extrapolation
        """

        theta, phi = self.gal_rot((90-feed.dec)*np.pi/180., feed.ra*np.pi/180.)
        for i in range(feed.tod.shape[0]):
            feed.tod[i,:] += hp.get_interp_val(self.data,
                                              theta,
                                              phi) * self.frequency_model(feed.frequencies[i])



class AMEPlanck857(SkyMap):

    def __init__(self, filename=None):
        """
        Expects the Planck 857GHz map converted to dust K at 22.8GHz in Healpix format.

        """
        self.filename = filename
        self.data  = hp.read_map(filename)
        self.data[self.data == hp.UNSEEN] = 0
        #self.data *= 26.45/1e6
        self.nside = int(np.sqrt(self.data.size/12.))

        self.nu0 = 22.8 # GHz
        self.alpha = 2.85
        # The map is in Galactic coordinates so set up rotation
        self.gal_rot = hp.rotator.Rotator(coord=['C','G'])

        self.__name__ = 'AMEPlanck857'

    def frequency_model(self,frequency):
        return (self.nu0/frequency)**self.alpha

    def __call__(self, feed):
        """
        Simple power law extrapolation
        """
        theta, phi = self.gal_rot((90-feed.dec)*np.pi/180., feed.ra*np.pi/180.)
        #for i in range(feed.tod.shape[0]):
        #   d = hp.get_interp_val(self.data,
        #                         theta,
        #                         phi) * self.frequency_model(feed.frequencies[i])
        #   feed.tod[i,:] += d

class FFHIPASS(SkyMap):

    def __init__(self, filename=None):
        """
        HIPASS RRL free-free map at 1.420GHz in HPX format.

        This map is at 14.1arcmin resolution, so only provides
        diffuse background.

        """
        self.filename = filename
        self.data  = hp.read_map(filename)
        self.data[self.data == hp.UNSEEN] = 0
        self.nside = int(np.sqrt(self.data.size/12.))

        self.nu0 = 1.420 # GHz
        self.alpha = 2.1
        # The map is in Galactic coordinates so set up rotation
        self.gal_rot = hp.rotator.Rotator(coord=['C','G'])

        self.__name__ = 'FFHIPASS'

    def frequency_model(self,frequency):
        return (self.nu0/frequency)**self.alpha

    def __call__(self, feed):
        """
        Simple power law extrapolation
        """

        theta, phi = self.gal_rot((90-feed.dec)*np.pi/180., feed.ra*np.pi/180.)
        for i in range(feed.tod.shape[0]):
            d = hp.get_interp_val(self.data,
                                  theta,
                                  phi) * self.frequency_model(feed.frequencies[i])
            feed.tod[i,:] += d


class RokeGBTModel(SkyMap):

    def __init__(self, filename=None):
        """

        """
        self.filename = filename
        self.data  = hp.read_map(filename)*1e-3 # convert to K
        self.data[self.data == hp.UNSEEN] = 0
        self.nside = int(np.sqrt(self.data.size/12.))

        #self.nu0 = self.telescope.frequencies[0] # GHz
        # The map is in Galactic coordinates so set up rotation
        self.gal_rot = hp.rotator.Rotator(coord=['C','G'])

        self.__name__ = 'FFHIPASS'

    def frequency_model(self,frequency):
        return 1

    def __call__(self, feed):
        """
        Simple power law extrapolation
        """

        theta, phi = self.gal_rot((90-feed.dec)*np.pi/180., feed.ra*np.pi/180.)
        for i in range(feed.tod.shape[0]):
            d = hp.get_interp_val(self.data,
                                  theta,
                                  phi) * self.frequency_model(feed.frequencies[i])
            feed.tod[i,:] += d
