''' coordinates.py: Contains coordinate conversion tools '''

# import numpy as np

class Coordinates():



    def __init__(self, settings):
        ''' Initialise object, pass independent instance of IO class '''
        # super().__init__(settings) # copies all the IO attibutes to self, but this doesn't work when inheriting multiple classes
        # IO.__init__(self, settings)   # use this when inheriting multiple parent classes
        return


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


    def lst_calculator(lon, utc_time):
        ''' lon in degrees, utc_time  as 'MMDDYY HHMM'
        Only valid for dates between 1901 and 2099. Accurate to within 1.1s '''
        # Calculate longitude in DegHHMM format for edification of user
        hemisphere = 'W'
        if lon > 0: # if the number is positive it's in the Eastern hemisphere
            hemisphere = 'E'
        LongDeg = int(lon)
        LongMin = (lon - int(lon))*60
        LongSec = (LongMin - int(LongMin))*60
        LongMin = int(LongMin)
        LongSec = int(LongSec)
        TD = utc_time
        # Split TD into individual variables for month, day, etc. and convert to floats
        MM = float(TD[0:2])
        DD = float(TD[2:4])
        YY = float(TD[4:6])
        YY = YY+2000
        hh = float(TD[7:9])
        mm = float(TD[9:11])
        # Convert mm to fractional time
        mm = mm/60.
        # Reformat UTC time as fractional hours
        UT = hh+mm
        # Calculate the Julian date
        JD = (367*YY) - int((7*(YY+int((MM+9)/12)))/4) + int((275*MM)/9) + DD + 1721013.5 + (UT/24)
        # Calculate the Greenwhich mean sidereal time
        GMST = 18.697374558 + 24.06570982441908*(JD - 2451545)
        GMST = GMST % 24    # use modulo operator to convert to 24 hours
        GMSTmm = (GMST - int(GMST))*60          # convert fraction hours to minutes
        GMSTss = (GMSTmm - int(GMSTmm))*60      # convert fractional minutes to seconds
        GMSThh = int(GMST)
        GMSTmm = int(GMSTmm)
        GMSTss = int(GMSTss)
        # Convert to the local sidereal time by adding the longitude (in hours) from the GMST
        lon = lon/15.      # fonvert longitude to hours
        LST = GMST+lon     # fraction LST. If negative we want to add 24...
        if LST < 0:
            LST = LST + 24
        LSTmm = (LST - int(LST))*60          # convert fraction hours to minutes
        LSTss = (LSTmm - int(LSTmm))*60      # convert fractional minutes to seconds
        LSThh = int(LST)
        LSTmm = int(LSTmm)
        LSTss = int(LSTss)
        LST_hour = LSThh+LSTmm/60.+LSTss/3600.
        return LST_hour
