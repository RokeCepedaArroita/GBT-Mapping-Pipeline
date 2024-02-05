import numpy as np
import matplotlib.pyplot as plt
import datetime
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')


# GBT Coordinates

gbt_lon = -79.839856
gbt_lat = +38.433021

# Calculate sidereal time of object

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from astropy.utils import iers
# iers.conf.auto_download = False
observing_location = EarthLocation(lat=gbt_lat*u.deg, lon=gbt_lon*u.deg)

from astropy.time import Time
t = Time.now()
t.delta_ut1_utc = 0.
t.sidereal_time('mean', 'greenwich')

print(t)


# Get elevation of object

LDN1582 = SkyCoord(l=192.41*u.degree,b=-11.51*u.degree, frame='galactic') #SkyCoord.from_name('LDN1582')
B30 = SkyCoord.from_name('Barnard 30')

print(LDN1582, B30)

lst_hour = []
el = []

# Create lots of time strings

for i in np.arange(0,24):
    time = '2020-1-28 ' + f'{i:02d}'+':00:00'
    time = Time(time)

    observing_time = Time(time, scale='utc', location=observing_location)
    LST = observing_time.sidereal_time('mean')

    LDN1582altaz = LDN1582.transform_to(AltAz(obstime=time,location=observing_location))
    #print(LST, LDN1582altaz)


    # for att in dir(LST):
    #     print (att, getattr(LST,att))

    lst_hour.append(LST.hour)
    el.append(LDN1582altaz.alt.degree)


print(lst_hour)
print(el)

xnew = np.linspace(8,22,100000)

from scipy.interpolate import interp1d
interpolate = interp1d(lst_hour, el, kind='cubic')

ynew = interpolate(xnew)
print(f'Maximum elevation {np.nanmax(ynew)} at LST of {xnew[ynew==np.nanmax(ynew)]}')

# Plot

plt.title('GBT: ' + r'$\ell,b=(192.41, -11.51)$')
plt.xlabel('LST (h)')
plt.ylabel('Elevation (deg)')
plt.plot([0,24],[30,30], 'r-', alpha=0.3, label='el=30 deg')
# plt.plot([0,24],[40,40], 'r-', alpha=0.3, label='el=40 deg')
# plt.plot([0,24],[50,50], 'r-', alpha=0.3, label='el=50 deg')
# plt.plot([0,24],[60,60], 'r-', alpha=0.3, label='el=60 deg')
plt.plot([0,24],[0,0], 'k-', linewidth=0.5)
plt.plot([5.55909059,5.55909059],[-45,70], 'k--', label='LST=5.56h')
plt.plot([9.65036,9.65036],[-45,70], 'g--', label='LST=9.65h')
plt.plot([1.45456,1.45456],[-45,70], 'g--', label='LST=1.45h')
plt.plot(lst_hour, el,'k,')
plt.plot(xnew, ynew)
plt.xlim([0,24])
plt.ylim([-45,70])
plt.legend()
plt.savefig('GBT_pdrcoords.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()




# LDN1582altaz = LDN1582.transform_to(AltAz(obstime=time,location=observing_location))
# print("LDN1582's Altitude = {0.alt:.2}".format(LDN1582altaz))
