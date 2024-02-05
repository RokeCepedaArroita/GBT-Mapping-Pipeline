import numpy as np
from matplotlib import pyplot
from sdosim.Tools import Coordinates
import h5py
import healpy as hp
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord

class Telescope: # A helper class that creates lots of feeds
    def __init__(self,tsys=20, band_width=1e9, sample_rate=1, lons=[], lats=[], feedpositions=[], frequencies=[1]):

        assert len(lons) == len(lats)
        
        self.tsys = tsys
        self.band_width = band_width
        self.sample_rate = sample_rate

        self.lons = lons
        self.lats = lats
        self.feedpositions = feedpositions
        self.frequencies = frequencies

        self.feeds = []
        for itele, (lon, lat) in enumerate(zip(lons,lats)):
            for ifeed, (xpos, ypos) in enumerate(feedpositions):
                self.feeds += [Feed(tsys=tsys, band_width=band_width, sample_rate=sample_rate, longitude=lon, latitude=lat, frequencies=frequencies, xoffset=xpos, yoffset=ypos, name='T{}F{}'.format(itele, ifeed))]

    def __call__(self):
        return self.feeds
    
class Feed: # Fundamental chunk of data too
    # A feed should store:
    # 1) Location of telescope and focal plane position
    # 2) Beam information
    # 3) A dictionary of classes that define each TOD job
    # 4a) Since schedules need telescope location, a schedule should be associated with a feed
    # 4b) There can be many instances of the same feed, with many different schedules
    def __init__(self,
                 writemodes=None,
                 rcvrmodes=None,
                 tsys = 20,
                 band_width=1e9,
                 sample_rate=1,
                 longitude=0,
                 latitude=0,
                 xoffset=0,
                 yoffset=0, 
                 frequencies=[1], # GHz
                 schedule=None,
                 name=None):

        self.name = name

        # Constants about data
        self.sample_rate = sample_rate
        self.frequencies = frequencies
        self.nFrequencies= len(self.frequencies)
        self.tsys = tsys
        self.band_width = band_width
        
        # Constants about feed location
        self.longitude = longitude
        self.latitude  = latitude
        self.xoffset   = xoffset
        self.yoffset   = yoffset

        # Information about 
        self.rcvrmodes = rcvrmodes
        self.schedule  = schedule
        self.writemodes = writemodes

    def __call__(self, lock=None):
        if not isinstance(self.schedule,type(None)):
            self.schedule(self)
        else:
            print('No schedule set for {}'.format(self.name))


        if not isinstance(self.rcvrmodes, type(None)):
            self.tod = np.zeros((self.nFrequencies, self.nSamples))
            for rcvrmode in self.rcvrmodes:
                rcvrmode(self)

        if not isinstance(self.writemodes, type(None)):
            for writemode in self.writemodes:
                if isinstance(lock, type(None)):
                    writemode(self)
                else:
                    with lock:
                        writemode(self)

    def set_schedule(self, schedule):
        self.schedule = schedule
        self.nSamples = int(self.sample_rate * self.schedule.length.sec)

    def set_rcvrmodes(self, rcvrmodes):
        self.rcvrmodes = rcvrmodes

    def set_writemodes(self, writemodes):
        self.writemodes = writemodes


    def clear(self):

        del self.tod
        del self.ra
        del self.dec
        del self.az
        del self.el
        del self.mjd

        for rcvrmode in self.rcvrmodes:
            if hasattr(rcvrmode,'data'):
                rcvrmode.data = None
        for writemode in self.writemodes:
            if hasattr(writemode,'data'):
                writemode.data = None
