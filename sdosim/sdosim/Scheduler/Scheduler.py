import numpy as np
from matplotlib import pyplot
#from comancpipeline.Tools import Coordinates

import h5py
import healpy as hp
from astropy.time import Time, TimeDelta
import copy
from tqdm import tqdm

from sdosim.Tools import Coordinates
from sdosim.Tools import pysla


sidereal_day_jd = (23 + 56/60. + 4.0905/60.**2)/24.

class Schedule:
    """
    Calculate the azimuth and elevation of an observation
    """
    def __init__(self, skycoord,
                 longitude, latitude,
                 observingMode = None,
                 source_lst=None, source_lha=None,  # Either specify source hour angle or source lst
                 length=None, span_ha=None, # Specify either observing time length or the angular span in hour angle
                 lead_source = True):

        """
        source_lst - (hours)
        source_lha  - (hours)
        """

        self.longitude = longitude
        self.latitude  = latitude

        self.observingMode = observingMode

        self.skycoord = skycoord
        self.lead_source = lead_source

        # Check inputs:
        if not isinstance(source_lst, type(None)):
            self.source_lst = source_lst*15 # convert to degrees
            self.source_lha = np.mod(self.source_lst - self.skycoord.ra.value,360)
        elif not isinstance(source_lha, type(None)):
            self.source_lha  = source_lha*15 # convert to degrees
            self.source_lst = np.mod(self.source_lha + self.skycoord.ra.value,360)
        else:
            raise ValueError('Must specify either source_lst or source_ha')

        if not isinstance(length, type(None)):
            self.length  = length
            self.span_ha = self.length.sec/3600.*15 * np.cos(self.skycoord.dec.value*np.pi/180.)
        elif not isinstance(span_ha, type(None)):
            self.span_ha = span_ha
            self.length  = TimeDelta(self.span_ha / np.cos(self.skycoord.dec.value*np.pi/180.)/15./24., format='jd')
        else:
            raise ValueError('Must specify either length or span_ha parameters')

    def telescope_azel(self):
        """
        Calculate the source azimuth and elevation given the telescope position
        """
        self.az0, self.el0 = pysla.ps_de2h((self.source_lha+self.span_ha/2)*np.pi/180, self.skycoord.dec.value*np.pi/180, self.latitude*np.pi/180)
        self.az0 *= 180./np.pi
        self.el0 *= 180./np.pi


        return self.az0, self.el0

    def set_start_time(self,mjd0):
        self.mjd0 = mjd0
        self.start = Time(self.mjd0, format='mjd')

    def __call__(self,feed):
        """
        Used when producing simulated time ordered data
        """

        assert not isinstance(self.observingMode,type(None)), 'No observing mode set!'

        self.observingMode(feed)


class Scheduler:
    """
    Calculate the azimuth and elevation of an observation
    """
    def __init__(self,
                 start_time=None,
                 skycoord=None,
                 longitude=None,
                 latitude=None,
                 observingMode= None,
                 nRepointings=1,
                 nDays=1,
                 minElevation = 0,
                 source_lst0=None, source_lha0=None,  # Either specify source hour angle or source lst
                 length=None, span_ha=None, # Specify either observing time length or the angular span in hour angle
                 lead_source = True,
                 schedule_list=None):

        """
        nRepointings - How many times do you want to retrack the source in a single day (defining a schedule "set"), limits to max repointing between rise/set times
        nDays        - How many days to want to repeat the same "set" of schedules

        minElevation - Defines the elevation of the local horizon (e.g., defines the rise/set times of source)

        source_lst - (hours)
        source_lha  - (hours)
        """

        # If we initialise with a schedule list, do nothing else
        if not isinstance(schedule_list, type(None)):
            self.schedules = schedule_list
            return

        # Else procede with the usual initialisation
        self.start_time    = start_time
        self.skycoord     = skycoord

        self.longitude = longitude
        self.latitude  = latitude

        self.lead_source  = lead_source
        self.nRepointings = nRepointings
        self.nDays        = nDays
        self.minElevation = minElevation

        self.observingMode = observingMode

        # Check inputs:
        if not isinstance(source_lst0, type(None)):
            self.source_lst0 = source_lst0*15 # convert to degrees
            self.source_lha0 = np.mod(self.source_lst0 - self.skycoord.ra.value,360)
        elif not isinstance(source_lha0, type(None)):
            self.source_lha0  = source_lha0*15 # convert to degrees
            self.source_lst0 = np.mod(self.source_lha0 + self.skycoord.ra.value,360)
        else:
            raise ValueError('Must specify either source_lst or source_ha')

        if not isinstance(length, type(None)):
            self.length  = length
            self.span_ha = self.length.sec/3600*15 * np.cos(self.skycoord.dec.value*np.pi/180.)
        elif not isinstance(span_ha, type(None)):
            self.span_ha = span_ha
            self.length  = TimeDelta(self.span_ha / np.cos(self.skycoord.dec.value*np.pi/180.)/15./24., format='jd')
        else:
            raise ValueError('Must specify either length or span_ha parameters')


        # Check elevation range
        self.set_telescope_info()
        print(f'Source elevation at start time = {self.el0:.2f}', flush=True)
        assert (self.el0 > self.minElevation), "Source below minimum elevation!"




        self.mjd_start = None
        self.schedule_set = []
        for i in range(self.nRepointings):
            if self.lead_source:
                offset = self.span_ha/2.
            else:
                offset = 0

            # We have the starting local hour angle
            lha = np.mod(self.source_lha0 + i*self.span_ha + offset,360)
            if lha > 180:
                lha -= 360

            # When is this hour angle next up?
            if isinstance(self.mjd_start, type(None)):
                self.source_tracks(lha)

            mjd_start = self.mjd_start + i * self.length.jd

            # Create schedules
            s = Schedule(self.skycoord, longitude, latitude, source_lha=lha/15., length = self.length)
            s.set_start_time(mjd_start)
            s_az, s_el = s.telescope_azel()
            if s_el > self.minElevation:
                self.schedule_set += [s]
            else:
                break

        # Create master list of schedules, increment start days by 1 sidereal day
        self.schedules = []
        for i in range(self.nDays):
            for schedule in self.schedule_set:
                s = copy.copy(schedule)
                s.observingMode = copy.copy(observingMode)
                s.set_start_time(s.mjd0+i*sidereal_day_jd)
                self.schedules += [s]

    @property
    def end_time(self):
        """
        Calculate the end time of set of schedules
        """
        end_times = []
        for sch in self.schedules:
            end_times += [sch.mjd0 + sch.length.jd]

        return Time(np.max(end_times),format='mjd')

    @property
    def total_time(self):
        """
        Calculate the total observing time of a set of schedules
        """
        times = []
        for sch in self.schedules:
            times += [sch.length.sec]

        return TimeDelta(np.sum(times), format='sec')

    @property
    def start_time(self):
        if hasattr(self,'_start_time'):
            return self._start_time
        else:
            times = []
            for sch in self.schedules:
                times += [sch.mjd0 + sch.length.jd]

            return Time(np.min(times),format='mjd')

    @start_time.setter
    def start_time(self,s):
        self._start_time = s

    def __add__(self, other):
        return Scheduler(schedule_list = self.schedules + other.schedules)



    def source_tracks(self, lha0):
        """
        Calculate the azimuth/elevation tracks of a source
        """

        mjd = self.start_time.mjd + np.linspace(0,sidereal_day_jd,86400)
        ra  = np.ones(mjd.size) * self.skycoord.ra.value
        dec = np.ones(mjd.size) * self.skycoord.dec.value
        az, el, ha = Coordinates.e2h(ra, dec, mjd, self.longitude, self.latitude, return_lha=True)
        i_mjd0 = np.argmin((ha - lha0)**2)
        self.mjd_start = mjd[i_mjd0]

        #print(lha0)
        #pyplot.plot(ha, el,',')
        #pyplot.axvline(ha[i_mjd0],color='g')
        #pyplot.axhline(el[i_mjd0],color='r')
        #pyplot.show()

        return self.mjd_start

    def set_telescope_info(self):
        """
        Calculate the source azimuth and elevation given the telescope position
        """
        self.az0, self.el0 = pysla.ps_de2h(np.multiply(self.source_lha0,np.pi/180.), np.multiply(self.skycoord.dec.value,np.pi/180.), np.multiply(self.latitude,np.pi/180.))
        self.az0 *= 180./np.pi
        self.el0 *= 180./np.pi

    def plot_coverage(self):

        self.start_time.mjd + np.linspace(0,self.length.jd,1000)

    def todms(self, v, return_str=True):

        d = np.floor(v)
        m = np.floor((v-d)*60)
        s = ((v-d)*60 - m)*60

        if return_str:
            return '{:02d}:{:02d}:{:.2f}'.format(int(d),int(m),s)
        else:
            return d, m, s

    def print(self):

        output_str = """
        Azimuth (deg)      Elevation (deg)      MJD      MJD_frac (hrs)     Hour_Angle (hrs)     LST (hrs)
        ---------------------------------------------------------------------
        {}
        """

        lines = []
        for sch in self.schedules:
            az, el = sch.telescope_azel()

            lines += ['{:.0f}    {:.0f}    {:d}     {:.2f}      {}    {}'.format(self.todms(az), self.todms(el), int(sch.mjd0), (sch.mjd0-np.floor(sch.mjd0))*24, self.todms(sch.source_lha/15.), self.todms(sch.source_lst/15.))]

        all_lines = '\n'.join(lines)

        print(output_str.format(all_lines))

    def plot_elvlst(self):

        els = []
        lsts = []
        for sch in self.schedules:
            az, el = sch.telescope_azel()
            els += [el]
            lsts+= [sch.source_lst/15]

        mjd = self.start_time.mjd + np.linspace(0,sidereal_day_jd,86400)
        ra  = np.ones(mjd.size) * self.skycoord.ra.value
        dec = np.ones(mjd.size) * self.skycoord.dec.value
        az, el, ha = Coordinates.e2h(ra, dec, mjd, self.longitude, self.latitude, return_lha=True)
        lst = np.mod(ha + self.skycoord.ra.value,360)

        els = np.array(els)
        lsts = np.array(lsts)
        asort = np.argsort(lsts)
        asort = np.argsort(lst)
        pyplot.plot(lst[asort]/15,el[asort])#lsts[asort], els[asort],'-')

        for lst, el in zip(lsts, els):
            pyplot.plot(lst, el, 'kx')

        pyplot.xlabel('Local Sidereal Time')
        pyplot.ylabel('Elevation')
        pyplot.xlim(0,24)
        pyplot.ylim(0,90)
        pyplot.axhline(self.minElevation,linestyle='--',color='k')
        pyplot.grid()
