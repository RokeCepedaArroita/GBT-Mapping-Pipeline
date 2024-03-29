import numpy as np
import matplotlib.pyplot as plt
import datetime
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u




def calculate_lstel_track(calibrator_coords, calibrator_name):

    def lst_calculator(lon, utc_time):
        ''' lon in degrees, utc_time  as 'MMDDYY HHMM' '''

        #Only valid for dates between 1901 and 2099. Accurate to within 1.1s.

        Long = lon      #Longitude of location in question (BMX LAT = 40.869 [40deg 52' 8"], BMX LONG = -72.866 [-72deg 51' 57"], Custer LONG = -72.435)

        #Calculate longitude in DegHHMM format for edification of user:
        hemisphere = 'W'
        if Long > 0:        # if the number is positive it's in the Eastern hemisphere
            hemisphere = 'E'
        LongDeg = int(Long)
        LongMin = (Long - int(Long))*60
        LongSec = (LongMin - int(LongMin))*60
        LongMin = int(LongMin)
        LongSec = int(LongSec)

        #print('\n\n\nThe longitude is set to %sdeg, [%s %sdeg %s\' %s\"]' %(Long, hemisphere, LongDeg, LongMin, LongSec))
        #TD = raw_input('\nEnter the UTC time and date as MMDDYY HHMM. (UTC = EST+5, EDT+4):\n')
        TD=utc_time

        #split TD into individual variables for month, day, etc. and convert to floats:
        MM = float(TD[0:2])
        DD = float(TD[2:4])
        YY = float(TD[4:6])
        YY = YY+2000
        hh = float(TD[7:9])
        mm = float(TD[9:11])

        # Convert mm to fractional time:
        mm = mm/60.

        # Reformat UTC time as fractional hours:
        UT = hh+mm

        # Calculate the Julian date:
        JD = (367*YY) - int((7*(YY+int((MM+9)/12)))/4) + int((275*MM)/9) + DD + 1721013.5 + (UT/24)
        #print('\nJulian Date: JD%s' %(JD))

        #calculate the Greenwhich mean sidereal time:
        GMST = 18.697374558 + 24.06570982441908*(JD - 2451545)
        GMST = GMST % 24    #use modulo operator to convert to 24 hours
        GMSTmm = (GMST - int(GMST))*60          #convert fraction hours to minutes
        GMSTss = (GMSTmm - int(GMSTmm))*60      #convert fractional minutes to seconds
        GMSThh = int(GMST)
        GMSTmm = int(GMSTmm)
        GMSTss = int(GMSTss)
        #print('\nGreenwhich Mean Sidereal Time: %s:%s:%s' %(GMSThh, GMSTmm, GMSTss))

        #Convert to the local sidereal time by adding the longitude (in hours) from the GMST.
        #(Hours = Degrees/15, Degrees = Hours*15)
        Long = Long/15.      # Convert longitude to hours
        LST = GMST+Long     #Fraction LST. If negative we want to add 24...
        if LST < 0:
            LST = LST +24
        LSTmm = (LST - int(LST))*60          #convert fraction hours to minutes
        LSTss = (LSTmm - int(LSTmm))*60      #convert fractional minutes to seconds
        LSThh = int(LST)
        LSTmm = int(LSTmm)
        LSTss = int(LSTss)

        LST_deg = LSThh+LSTmm/60.+LSTss/3600.

        #print('\nLocal Sidereal Time %s:%s:%s \n\n' %(LSThh, LSTmm, LSTss))
        return LST_deg





    # GBT Coordinates

    gbt_lon = -79.839856
    gbt_lat = +38.433021

    # Calculate sidereal time of object

    observing_location = EarthLocation(lat=gbt_lat*u.deg, lon=gbt_lon*u.deg)


    # Get elevation of object

    coords = SkyCoord(l=calibrator_coords[0]*u.degree,b=calibrator_coords[1]*u.degree, frame='galactic')

    lst_hour = []
    el = []

    # Create lots of time strings

    for i in np.arange(0,24):
        for min in np.arange(0,60):
            time = '2020-1-01 ' + f'{i:02d}'+f':{min:02d}:00'
            utc_time=f'010120 {i:02d}{min:02d}'
            time = Time(time, scale='utc')
            LST = lst_calculator(lon=gbt_lon, utc_time=utc_time)
            coordsaltaz = coords.transform_to(AltAz(obstime=time,location=observing_location))
            lst_hour.append(LST)
            el.append(coordsaltaz.alt.degree)


    # print(lst_hour)
    # print(el)

    xnew = np.linspace(0.02,23.98,100000)

    from scipy.interpolate import interp1d
    interpolate = interp1d(lst_hour, el, kind='cubic')

    ynew = interpolate(xnew)

    max_lst = xnew[ynew==np.nanmax(ynew)][0]
    max_lst_hours = np.int(max_lst)
    max_lst_mins = (max_lst-np.int(max_lst))*60
    print(f'Maximum elevation {np.nanmax(ynew)} at LST of {max_lst_hours} h {max_lst_mins:.2f} mins or {max_lst} hours')


    def lst_info(lst, el, min_el):

        def find_lsts(f, g, x):
            idx = np.argwhere(np.diff(np.sign(np.subtract(f, g)))).flatten()
            lst_cross = [x[index] for index in idx]
            el_Cross = [f[index] for index in idx]
            #print(lst_cross, el_Cross)
            return lst_cross, el_Cross

        lst_times, el_times = find_lsts(f=el, g=np.ones(np.shape(el))*min_el, x=lst)

        rise_lst_hours = np.int(lst_times[0])
        rise_lst_mins = np.round((lst_times[0]-np.int(lst_times[0]))*60)
        set_lst_hours = np.int(lst_times[1])
        set_lst_mins = np.round((lst_times[1]-np.int(lst_times[1]))*60)

        print(f'\n*** Requested elevation is {min_el} deg')
        print(f'Rise is at LST = {rise_lst_hours} h {rise_lst_mins:.2f} min' )
        print(f'Set is at LST = {set_lst_hours} h {set_lst_mins:.2f} min' )
        print(f'Total duration from rise to set is {(lst_times[1]-lst_times[0]):.2f} hours\n' )

        return

    # Calculate angular distance to central pdr point

    def haversine(lon1, lat1, lon2, lat2):
        ''' Calculate the angular distance between two points in the sky '''

        # Convert degrees to radians
        lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        theta = np.rad2deg(c) # Angular distance in degrees

        return theta

    theta = haversine(lon1=192.41, lat1=-11.51, lon2=calibrator_coords[0], lat2=calibrator_coords[1])
    print(f'\n** One way distance to calibrator is {theta} deg **\n')

    for elevation_query in [40]:
        lst_info(lst=xnew, el=ynew, min_el=elevation_query)


    return xnew, ynew







# Settings

linestyles = ['r--','g-.','b:','k-']
linewidths = [1,1,1,2]
calibrator_name = ['3C147', '3C48', '3C161','Target']
calibrator_coords = [[161.68636686748, +10.29833172063], [133.9625726319084, -28.7194212748621], [215.4389, -08.0670],[192.41, -11.51]]

fig, ax = plt.subplots(figsize=(8*1.3,3.2*1.3))
ax.axvspan(0+00/60, 5, -90, 90, alpha=0.3, color='g')
ax.axvspan(7, 10+27/60, -90, 90, alpha=0.3, color='r')
ax.axvspan(5.54-0.5, 5.54+0.5, -90, 90, alpha=0.3, color='b')
ax.axvspan(2+19/60, 5.54-0.5, -90, 90, color='#D3D3D3')
ax.axvspan(5.54+0.5, 8+46/60, -90, 90, color='#D3D3D3')


for calcoords, calname, linestyle, linewidth in zip(calibrator_coords, calibrator_name, linestyles, linewidths):
    cal_lst, cal_el = calculate_lstel_track(calcoords, calname)
    plt.plot(cal_lst, cal_el, linestyle, linewidth=linewidth, label=calname)
    print(f'Just finished with {calname}')

asdkaskd


plt.ylim([30,90])
plt.xlim([0,10.4])
plt.xlabel('LST (h)')
plt.ylabel('Elevation (deg)')
plt.legend(loc='upper right')
plt.savefig(f'calibrator_schedule.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.savefig(f'calibrator_schedule.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

asdmlkaskdasd


#
# # Plot
# plt.title(f'GBT: {calibrator_name}')
# plt.xlabel('LST (h)')
# plt.ylabel('Elevation (deg)')
# plt.plot([0,24],[30,30], 'r-', alpha=0.3, label='el=30 deg')
# plt.plot([0,24],[0,0], 'k-', linewidth=0.5)
# plt.plot([5.55909059,5.55909059],[np.nanmin(el)/1.1,np.nanmax(el)*1.1], 'k--', label='LST=5.56h')
# plt.plot([9.65036,9.65036],[np.nanmin(el)/1.1,np.nanmax(el)*1.1], 'g--', label='LST=9.65h')
# plt.plot([1.45456,1.45456],[np.nanmin(el)/1.1,np.nanmax(el)*1.1], 'g--', label='LST=1.45h')
# plt.plot(lst_hour, el,'k,')
# plt.plot(xnew, ynew)
# plt.xlim([0,24])
# plt.ylim([np.nanmin(el)/1.1,np.nanmax(el)*1.1])
# plt.legend()
# plt.savefig(f'GBT_{calibrator_name}.png', dpi=600, bbox_inches='tight', pad_inches=0)
# plt.show()
