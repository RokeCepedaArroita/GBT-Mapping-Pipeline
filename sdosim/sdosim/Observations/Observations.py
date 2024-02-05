import numpy as np
from matplotlib import pyplot
#from comancpipeline.Tools import Coordinates
from sdosim.Tools import Coordinates
import h5py
import healpy as hp
from astropy.time import Time, TimeDelta
import copy
from tqdm import tqdm
from sdosim.Tools import pysla
from concurrent import futures

class OldScheduler:
    """
    Calculate the azimuth and elevation of an observation
    """
    def __init__(self, skycoord, width=1, height=1, field_of_view=None, hourangle=0, which='rising', pointsource=True):
        
        self.skycoord = skycoord
        self.width = width
        self.height = height
        self.which = which
        self.field_of_view = field_of_view
        self.hourangle = hourangle*15

        self.pointsource=pointsource

    def box(self,mjd0=None):

        ra_c = self.skycoord.ra.value
        dec_c= self.skycoord.dec.value
        N = 100
        ra = np.concatenate((np.linspace(ra_c-self.width/2., ra_c+self.width/2.,N),
                             np.ones(N)*(ra_c+self.width/2.),
                             np.linspace(ra_c+self.width/2.,ra_c-self.width/2,N),
                             np.ones(N)*(ra_c-self.width/2.)))
        dec =np.concatenate((np.ones(N)*(dec_c+self.height/2.),
                             np.linspace(dec_c+self.height/2., dec_c-self.height/2.,N),
                             np.ones(N)*(dec_c-self.height/2.),
                             np.linspace(dec_c-self.height/2., dec_c+self.height/2.,N)))

        if not isinstance(mjd0, type(None)):
            ra, dec = Coordinates.precess2year(ra,dec,mjd0*np.ones(ra.size))

        return ra, dec


    def __call__(self, mjd0, lon, lat):
        return self.point_source_target_fortran(mjd0, lon, lat)
        #if self.pointsource:
        #    return self.point_source_target(mjd0, lon, lat)
        #else:
        #    return self.diffuse_source_target(mjd0,lon,lat)

    def diffuse_source_target(self,mjd0,lon,lat):
        """        
        Take the start time (mjd0) and calculate how long to observe.
        
        This starts and stops based on any of the target area still being
        within the target sky area. This is more optimal for diffuse source
        targets (e.g., science fields).
        """
        ra, dec = self.skycoord.ra.value*np.ones(1), self.skycoord.dec.value*np.ones(1)
        ra, dec = Coordinates.precess2year(ra,dec,mjd0*np.ones(1))

        sky_corners = [[ra[0]-self.width/2., dec[0]-self.height/2.],
                       [ra[0]-self.width/2., dec[0]+self.height/2.],
                       [ra[0]+self.width/2., dec[0]-self.height/2.],
                       [ra[0]+self.width/2., dec[0]+self.height/2.]]
        daysec = 86400.
        mjd = mjd0 + np.arange(0,daysec,10)/daysec

        daysec = 86400.
        mjd = mjd0 + np.arange(0,daysec,10)/daysec
        ra = np.ones(mjd.size)*self.skycoord.ra.value
        dec= np.ones(mjd.size)*self.skycoord.dec.value
        ra, dec = Coordinates.precess2year(ra,dec,mjd)
        az, el, lha = Coordinates.e2h(ra, dec, mjd, lon, lat, return_lha=True)
        cross = np.argmin((lha-self.hourangle)**2)
        rot = hp.rotator.Rotator(rot=[lha[cross], dec[cross]])
        theta, phi = (90-dec)*np.pi/180., lha*np.pi/180.
        theta, phi = rot(theta,phi)
        lha_r, dec_r = phi*180./np.pi, (np.pi/2. - theta)*180./np.pi


        select = np.zeros(mjd.size,dtype=bool)
        # Loop over corners checking for when each transits
        for (c_ra, c_dec) in sky_corners:
            _ra = np.ones(mjd.size)*c_ra
            _dec= np.ones(mjd.size)*c_dec
            #ra, dec = Coordinates.precess2year(ra,dec,mjd)
            _az, _el, _lha = Coordinates.e2h(_ra, _dec, mjd, lon, lat, return_lha=True)
            # find when the next local hour angle crossing is
            theta, phi = (90-dec)*np.pi/180., _lha*np.pi/180.
            theta, phi = rot(theta,phi)
            lha_r, dec_r = phi*180./np.pi, (np.pi/2. - theta)*180./np.pi

            select = select | ((lha_r > -self.width/2) & (lha_r < self.width/2) & \
                               (dec_r > -self.height/2) & (dec_r < self.height/2))


        s = np.where(select)[0]
        if (np.max(s) - np.min(s)) < select.size//2:
            select[np.min(s):np.max(s)] = True
        else:
            select[0:np.min(s)] = True
            select[np.max(s):]  = True
        #pyplot.plot(select)
        #pyplot.show()

        select=np.where(select)[0]
        if (np.max(s) - np.min(s)) < mjd.size//2:
            midpos = select[select.size//2]
        else:
            select[select > mjd.size//2] -= mjd.size
            midpos = int(np.mean(select))
            if midpos < 0:
                midpos += mjd.size

        mjd_mid = mjd[midpos]
        delta_mjd = np.mod(mjd-mjd_mid, 1) 
        delta_mjd[delta_mjd > 0.5] -= 1


        az0, el0, mjd0 = az[midpos], el[midpos], Time(np.min(mjd),format='mjd').mjd
        ra, dec = Coordinates.h2e(az0*np.ones(mjd.size),
                                  el0*np.ones(mjd.size),
                                  mjd,# mjd0*np.ones(1), 
                                  lon, lat)

        mjd = mjd_mid + delta_mjd


        mjd = mjd[select]

        box_ra, box_dec = self.box(mjd0)

        return az[midpos], el[midpos], Time(np.min(mjd),format='mjd'), TimeDelta(np.max(mjd)-np.min(mjd),format='jd')

    def point_source_target_fortran(self,mjd0, lon,lat):
        """
        A more efficient algorithm for calculating the source position
        """
        d2r = np.pi/180.
        az0, el0, mjd_start, mjd_delta = pysla.source_target(self.width*d2r,
                                                             self.hourangle*d2r,
                                                             self.skycoord.ra.value*d2r,
                                                             self.skycoord.dec.value*d2r,
                                                             mjd0,
                                                             lon*d2r,
                                                             lat*d2r)
        
        az0, el0, mjd_start, mjd_delta = az0*180./np.pi, el0*180./np.pi, Time(mjd_start,scale='utc',format='mjd'), TimeDelta(mjd_delta,format='jd')

        return az0, el0, mjd_start, mjd_delta
                                                        

    def point_source_target(self,mjd0, lon, lat):
        """
        Take the start time (mjd0) and calculate how long to observe.
        
        Starts and stop based on central source target location,
        this is optimal if observing a point source (e.g. a calibrator)
        """

        
        daysec = 86400.
        mjd = mjd0 + np.arange(0,daysec,10)/daysec
        ra = np.ones(mjd.size)*self.skycoord.ra.value
        dec= np.ones(mjd.size)*self.skycoord.dec.value
        ra, dec = Coordinates.precess2year(ra,dec,mjd)
        az, el, lha = Coordinates.e2h(ra, dec, mjd, lon, lat, return_lha=True)


        # cut low elevations
        good = (el > 0)
        az  = az[good]
        el  = el[good]
        lha = lha[good]
        dec = dec[good]
        ra  = ra[good]
        mjd = mjd[good]

        # find when the next local hour angle crossing is
        cross = np.argmin((lha-self.hourangle)**2)

        rot = hp.rotator.Rotator(rot=[lha[cross], dec[cross]])
        theta, phi = (90-dec)*np.pi/180., lha*np.pi/180.
        theta, phi = rot(theta,phi)
        lha_r, dec_r = phi*180./np.pi, (np.pi/2. - theta)*180./np.pi

        select = np.where((lha_r > -self.width/2) & (lha_r < self.width/2) & \
                          (dec_r > -self.height/2) & (dec_r < self.height/2))[0]

        mjd_mid = mjd[select[select.size//2]]
        delta_mjd = np.mod(mjd-mjd_mid, 1) 
        delta_mjd[delta_mjd > 0.5] -= 1

        mjd = mjd_mid + delta_mjd


        mjd = mjd[select]



        
        return az[select[select.size//2]], el[select[select.size//2]], Time(np.min(mjd),format='mjd'), TimeDelta(np.max(mjd)-np.min(mjd),format='jd')

class Observations: # Stores all the observations
    # Observations should:
    # 1) Take a list of schedules and feeds and pair them up
    def __init__(self,feeds, schedules, rcvrmodes=None, writemodes=None):

        self.schedules = schedules
        self.feeds = []
        for feed in tqdm(feeds):
            for schedule in self.schedules:
                s = copy.copy(schedule)
                f = copy.copy(feed)            
                f.set_schedule(s)
                if not isinstance(rcvrmodes, type(None)):
                    # Each feed gets its own copy of a "light" receiver mode
                    # Shared memory is called for any large rcvrmode.data 
                    f.set_rcvrmodes(copy.copy(rcvrmodes)) 

                if not isinstance(writemodes, type(None)):
                    f.set_writemodes(copy.copy(writemodes))

                self.feeds += [f]

    def __call__(self):
        return self.feeds

class OldSchedule: # Fundamental chunk of data
    # A schedule should:
    # 1) A start and end time
    # 2) Define the fundamental smallest chunk of data
    # 3) Define a dictionary that contain a set of objects that describe each step of the schedule
    
    def __init__(self,obs):
        self.obs = obs # A list containing each observation (in order!)

        # Calculate the length
        #self.length = TimeDelta(0,format='jd')
        #for chunk in self.obs:
        #    self.length += chunk.length
            
    
    def __call__(self, feed):
        start = self.start
        for chunk in self.obs:
            chunk.start = start.copy()
            chunk(feed) # Each observation takes a feed object, creates data stores it in the feed
            start += chunk.length



class ObservationChunk:
    def __init__(self, start=None):
        self.start = start
        self.length = TimeDelta(0,format='jd')
        self.target = None

        # For celestial to galactic rotations
        self.e2g = hp.rotator.Rotator(coord=['C','G'])
        self.g2e = hp.rotator.Rotator(coord=['G','C'])

    def __call__(self, feed):
        self.run(feed)

    def run(self, feed):
        pass

    def apply_offsets(self,feed):
        pass

# --- Definitions for all the different observing strategies

class DriftScan(ObservationChunk):
    """
    Drift Scan sets the telescope to a fixed azimuth/elevation, and waits
    """
    def __init__(self, start=None, length=None, az0=0, el0=0):
        self.start = start
        self.length= length
        self.az0 = az0
        self.el0 = el0

    def run(self, feed):
        """
        
        """

        self.nSamples = int(feed.sample_rate * self.length.sec)
        self.dt = self.length/self.nSamples

        # First define the telescope coordinates
        feed.az = np.ones(self.nSamples) * self.az0
        feed.el = np.ones(self.nSamples) * self.el0
        self.apply_offsets(feed)
        feed.mjd= (self.start.mjd + (np.arange(self.nSamples)+0.5)*self.dt).value

        # Then the sky coordinates
        feed.ra, feed.dec = Coordinates.h2e(feed.az, feed.el, feed.mjd, 
                                            feed.longitude, feed.latitude)
        # Precess to J2000 since all ancillary maps are in this frame
        feed.ra, feed.dec = Coordinates.precess(feed.ra, feed.dec, feed.mjd)


class AzRaster(ObservationChunk):
    """
    """

    def __init__(self, target=None, 
                 az0=0, el0=0, 
                 az_radius=1, 
                 slew_speed=0.5, 
                 repointings = 1,
                 noElCorrection=True,
                 length=TimeDelta(1800,format='sec')):
        """
        az0 - degrees
        el0 - degrees
        az_radius - degrees
        slew_speed - degrees/second
        length - TimeDelta object in jd format
        """
        self.target = target
        if isinstance(target, type(None)):
            self.az0 = az0
            self.el0 = el0

        self.az_radius = az_radius
        if length.format != 'jd':
            self.length = TimeDelta(length.jd, format='jd')
        else:
            self.length = length

        if noElCorrection:
            self.elcorr = 1
        else:
            self.elcorr = np.cos(self.el0*np.pi/180.)
        if isinstance(self.target, type(None)):
            tdelta = az_radius/self.elcorr/slew_speed/86400.*(2*np.pi)
            self.length =  TimeDelta(tdelta, format='jd')

        self.slew_speed=slew_speed

        self.repointings = repointings

    def run(self,feed):
        
        self.nSamples = int(feed.sample_rate * self.length.sec)
        self.dt = self.length/self.nSamples

        # First define the telescope coordinates
        rate = self.slew_speed/self.az_radius*self.elcorr 
        theta   = np.arange(self.nSamples) * self.dt.sec * rate
        feed.az = np.mod(np.sin(theta) * self.az_radius/self.elcorr  +  self.az0 + feed.xoffset,360)
        feed.el = np.ones(self.nSamples) * self.el0 + feed.yoffset
        self.apply_offsets(feed)
        feed.mjd= (self.start.mjd + (np.arange(self.nSamples)+0.5)*self.dt).value

        # Then the sky coordinates
        feed.ra, feed.dec = Coordinates.h2e(feed.az, feed.el, feed.mjd, 
                                            feed.longitude, feed.latitude)

        # Precess to J2000 since all ancillary maps are in this frame
        feed.ra, feed.dec = Coordinates.precess(feed.ra, feed.dec, feed.mjd)

class AzElLissajous(ObservationChunk):
    """
    """

    def __init__(self, 
                 az_radius=1, 
                 el_radius=1,
                 az_slew_speed=0.5, 
                 el_slew_speed=0.5,
                 phase = np.pi/2.,
                 repointings = 1,
                 length=TimeDelta(1800,format='sec')):
        """
        az0 - degrees
        el0 - degrees
        az_radius - degrees
        slew_speed - degrees/second
        length - TimeDelta object in jd format
        """

        self.phase = phase

        #self.target = target
        #if isinstance(target, type(None)):
        #    self.az0 = az0
        #    self.el0 = el0

        self.az_radius = az_radius
        self.el_radius = el_radius
        if length.format != 'jd':
            self.length = TimeDelta(length.jd, format='jd')
        else:
            self.length = length
        self.az_slew_speed=az_slew_speed
        self.el_slew_speed=el_slew_speed

 
    def run(self,feed):

        az0, el0 = feed.schedule.telescope_azel()
        
        self.nSamples = int(feed.sample_rate * self.length.sec)
        self.dt = self.length/self.nSamples

        # First define the telescope coordinates
        rate = self.az_slew_speed/self.az_radius
        theta   = np.arange(feed.nSamples) * self.dt.sec * rate
        feed.az = np.mod(np.sin(theta + self.phase) * self.az_radius/np.cos(el0*np.pi/180.)  +  az0 + feed.xoffset,360)

        rate = self.el_slew_speed/self.el_radius
        theta   = np.arange(feed.nSamples) * self.dt.sec * rate 
        feed.el = np.mod(np.sin(theta) * self.el_radius  +  el0 + feed.yoffset,360)


        self.apply_offsets(feed)
        feed.mjd= (feed.schedule.start.mjd + (np.arange(feed.nSamples)+0.5)*self.dt).value

        # Then the sky coordinates
        feed.ra, feed.dec = Coordinates.h2e(feed.az, feed.el, feed.mjd, 
                                            feed.longitude, feed.latitude)

        # Precess to J2000 since all ancillary maps are in this frame
        feed.ra, feed.dec = Coordinates.precess(feed.ra, feed.dec, feed.mjd)

class RaScans(ObservationChunk):
    
        def __init__(self, target, 
                     ra_distance=1, 
                     dec_step=1,
                     dec_distance=1,
                     slew_speed=0.5):
            """
            az0 - degrees
            el0 - degrees
            az_radius - degrees
            slew_speed - degrees/second
            length - TimeDelta object in jd format
            """
            self.target = target
            
            self.slew_speed = slew_speed

            # Basic parameters
            self.ra_distance = ra_distance
            self.dec_distance = dec_distance
            self.dec_step = dec_step
            self.dec_n    = int(np.ceil(self.dec_distance/self.dec_step)) + 1


            # Now we are limited by telescope az/el drive speeds?
            # -- leave this for now basic version can just assume 
            # the user has been sensible...

            self.total_distance = (self.ra_distance+self.dec_distance)*self.dec_n
            
            self.length = TimeDelta(self.total_distance/self.slew_speed, format='sec')

        def define_scans(self,sample_rate=None, feed=None):

            if isinstance(sample_rate,type(None)):
                sample_rate=feed.sample_rate
            

            self.nSamples = int(sample_rate * self.length.sec)
            self.dt = self.length/self.nSamples

            ra0 = self.target.skycoord.ra.value
            dec0 = self.target.skycoord.dec.value

            #  samples/second * (distance)/(distance/seconds) 
            x_slice_samples  = int(sample_rate * self.ra_distance/self.slew_speed)
            y_step_samples   = int(sample_rate * self.dec_step/self.slew_speed)

            print(x_slice_samples,y_step_samples, self.ra_distance/self.dec_step)
            x_slices = []
            offset = self.ra_distance/2.

            x_move = np.zeros((self.nSamples))
            y_move = np.zeros((self.nSamples))
            last = 0
            for i in range(self.dec_n):
                # define main part of each slew
                rdir = (-1)**np.mod(i,2)
                x_slew = rdir*np.arange(x_slice_samples)*self.slew_speed/sample_rate + -rdir*offset 
                y_slew = np.zeros(x_slice_samples)  - self.dec_distance/2. + i*self.dec_distance/(self.dec_n-1)

                # then the link between slews
                x_step = np.zeros(y_step_samples)  + -rdir*offset 
                y_step = np.arange(y_step_samples)*self.slew_speed/sample_rate + y_slew[0]
                
                x_move[last:last+x_slew.size],y_move[last:last+x_slew.size] = x_slew, y_slew
                x_move[last+x_slew.size:last+x_slew.size+xstep_size] = x_step
                y_move[last+x_slew.size:last+x_slew.size+xstep_size] = y_step
                y_move[:,x_slew.size:] = x_step, y_step

                last += x_slew.size + x_step.size
            pyplot.plot(x_move, y_move,'.')
            pyplot.show()
               # ra_slices = [ra_slew]

            print(ra_slices)

        def run(self,feed):

            # So in this instance we assume we know ra/dec and
            #  are deriving the az/el positions
        
            self.nSamples = int(feed.sample_rate * self.length.sec)
            self.dt = self.length/self.nSamples
            
            ra0 = self.target.ra.value
            dec0 = self.target.dec.value
            
            ra_slice_samples  = int(feed.sample_rate * self.ra_distance/self.slew_speed)
            dec_slice_samples = int(feed.sample_rate * self.dec_distance/self.slew_speed/self.dec_n)

            ra_slices = []
            for i in range(self.dec_n):
                ra_slew = -1**np.mod(i,2)*np.arange(ra_slice_samples)*self.slew_speed/feed.sample_rate + -1**(1+np.mod(i,2))*ra_offset + ra0
                ra_slices = [ra_slew]

            # First define the telescope coordinates
            rate = self.az_slew_speed/self.az_radius
            theta   = np.arange(self.nSamples) * self.dt.sec * rate
            feed.az = np.mod(np.sin(theta) * self.az_radius/np.cos(self.el0*np.pi/180.)  +  self.az0 + feed.xoffset,360)

            rate = self.el_slew_speed/self.el_radius
            theta   = np.arange(self.nSamples) * self.dt.sec * rate + np.pi/2.
            feed.el = np.mod(np.sin(theta) * self.el_radius  +  self.el0 + feed.yoffset,360)
            #np.ones(self.nSamples) * self.el0 + feed.yoffset
            self.apply_offsets(feed)
            feed.mjd= (self.start.mjd + (np.arange(self.nSamples)+0.5)*self.dt).value
            
            # Then the sky coordinates
            feed.ra, feed.dec = Coordinates.h2e(feed.az, feed.el, feed.mjd, 
                                                feed.longitude, feed.latitude)

            # Precess to J2000 since all ancillary maps are in this frame
            feed.ra, feed.dec = Coordinates.precess(feed.ra, feed.dec, feed.mjd)


class DaisyAzEl(ObservationChunk):
    
        def __init__(self, 
                     r0=22/60,
                     tau=25.,
                     phi1=0,
                     phi2=0):
            """
            az0 - degrees
            el0 - degrees
            az_radius - degrees
            slew_speed - degrees/second
            length - TimeDelta object in jd format
            """

            self.r0 = r0
            self.tau = tau
            self.phi1 = phi1
            self.phi2 = phi2

        def run(self,feed):

            # So in this instance we assume we know ra/dec and
            #  are deriving the az/el positions

            az0, el0 = feed.schedule.telescope_azel()
        
            self.nSamples = int(feed.sample_rate * self.length.sec)
            self.dt = self.length/self.nSamples
            feed.az,feed.el, feed.mjd, feed.ra, feed.dec = self.generate_scans(az0, 
                                                                                el0, 
                                                                                self.r0, 
                                                                                self.tau, 
                                                                                self.phi1, 
                                                                                self.phi2, 
                                                                                self.nSamples,
                                                                                self.dt,
                                                                                self.start.mjd,
                                                                                feed.longitude,
                                                                                feed.latitude)

        def generate_scans(self, x0, y0, r0, tau, phi1, phi2, nSamples, dt, mjd0,longitude, latitude):

            t = np.arange(nSamples)*dt

            dx = r0 * np.sin(2*np.pi*t/tau + phi1) * np.cos(2*t/tau + phi2)/np.cos(y0)
            dy = r0 * np.sin(2*np.pi*t/tau + phi1) * np.sin(2*t/tau + phi2)
        
            x = x0 + dx
            y = y0 + dy
            mjd = mjd0 + t/86400.
            ra, dec = Coordinates.h2e(x, y, mjd, longitude, latitude)

            return x,y,mjd,ra, dec


class DaisyRaDec(ObservationChunk):
    
        def __init__(self, 
                     r0=22/60,
                     tau=25.,
                     phi1=0,
                     phi2=0):
            """
            az0 - degrees
            el0 - degrees
            az_radius - degrees
            slew_speed - degrees/second
            length - TimeDelta object in jd format
            """

            self.r0 = r0
            self.tau = tau
            self.phi1 = phi1
            self.phi2 = phi2

        def run(self,feed):

            # So in this instance we assume we know ra/dec and
            #  are deriving the az/el positions

            ra0, dec0 = feed.schedule.skycoord.ra.value, feed.schedule.skycoord.dec.value

            self.nSamples = int(feed.sample_rate * feed.schedule.length.sec)
            self.dt = feed.schedule.length.sec/self.nSamples
            feed.az,feed.el, feed.mjd, feed.ra, feed.dec = self.generate_scans(ra0, 
                                                                               dec0, 
                                                                               self.r0, 
                                                                               self.tau, 
                                                                               self.phi1, 
                                                                               self.phi2, 
                                                                               self.nSamples,
                                                                               self.dt,
                                                                               feed.schedule.start.mjd,
                                                                               feed.longitude,
                                                                               feed.latitude)

        def generate_scans(self, x0, y0, r0, tau, phi1, phi2, nSamples, dt, mjd0,longitude, latitude):

            t = np.arange(nSamples)*dt

            dx = r0 * np.sin(2*np.pi*t/tau + phi1) * np.cos(2*t/tau + phi2)/np.cos(y0)
            dy = r0 * np.sin(2*np.pi*t/tau + phi1) * np.sin(2*t/tau + phi2)
        
            x = x0 + dx
            y = y0 + dy
            mjd = mjd0 + t/86400.
            az,el = Coordinates.h2e(x, y, mjd, longitude, latitude)

            return az,el,mjd,x,y
