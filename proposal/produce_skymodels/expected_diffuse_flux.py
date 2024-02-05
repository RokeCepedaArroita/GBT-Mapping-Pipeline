import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

central_coordinates = [192.37-360, -11.55]

nside = 256

print(nside)

def distancemap(pixel_lon, pixel_lat, nside):
    ''' Return a map of angular distances in degrees from a
    given pixel. All inputs and outputs are in degrees'''

    lon, lat = hp.pix2ang(nside,np.arange(0,12*nside**2),lonlat=True)

    # Initialise distance map as map of np.nans
    distance_map = np.zeros((12*nside**2))*np.nan

    # Define good pixels as all pixels
    good_pixels = np.arange(0,12*nside**2) # use all of them

    # Parallelized Haversine formula goes here

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

    # Fill in distance map
    distance_map[good_pixels] = np.array(haversine(lon1=pixel_lon, lat1=pixel_lat, lon2=lon[good_pixels], lat2=lat[good_pixels]))

    return distance_map


distance_map = distancemap(pixel_lon=central_coordinates[0], pixel_lat=central_coordinates[1], nside=nside)

lon, lat = hp.pix2ang(nside,np.arange(0,12*nside**2),lonlat=True)

pnumber = np.where(distance_map==np.min(distance_map))[0]
plon = lon[pnumber]
plat = lat[pnumber]

print(f'At nside {nside}, the pixel number is {pnumber} with lon {plon} and lat {plat}, at a distance of {distance_map[pnumber]} deg.')


''' NOW CALCULATE FLUX EXPECTED AT 13.7 GHz FROM THE SED '''

beam = 0.00023924596203935044
sed_names = ['EM', 'A_AME', 'nu_AME', 'W_AME', 'T_d', 'tau', 'beta']
sed_params = [116.97048938996598, 4.491496718012366, 26.16453170192183, 0.7343722791084526, 20.725612053845186, -4.464240709376972, 1.5205134903449673]
sed_errors = [7.0292213, 0.14024409, 0.46699644, 0.03227444, 0.55110161, 0.0249024, 0.03608295]


def ame(nu, beam, A_AME, nu_AME, W_AME):
    ''' Here the beam doesn't do anything, but you still need it '''
    # TODO: scale by the beam so that amplitude of AME is in janskys!!!
    nlog = np.log(nu)
    nmaxlog = np.log(nu_AME)
    S = A_AME*np.exp(-0.5 * ((nlog-nmaxlog)/W_AME)**2)
    return S

def freefree(nu, beam, EM):
    T_e = 7500. # fixed electron temperature
    c = 299792458.
    k = 1.3806488e-23
    h = 6.62606957e-34
    a = 0.366 * np.power(nu,0.1)* np.power(T_e,-0.15) * (np.log(np.divide(4.995e-2, nu)) + 1.5 * np.log(T_e))
    T_ff = 8.235e-2 * a * np.power(T_e,-0.35) * np.power(nu,-2.1) * (1. + 0.08) * EM
    S = 2. * k * beam * np.power(np.multiply(nu,1e9),2)  / c**2 * T_ff * 1e26
    return S

def thermaldust(nu, beam, T_d, tau, beta):
    c = 299792458.
    k = 1.3806488e-23
    h = 6.62606957e-34
    nu = np.multiply(nu,1e9)
    planck = np.exp(h*nu/k/T_d) - 1.
    modify = 10**tau * (nu/353e9)**beta # set to tau_353
    S = 2 * h * nu**3/c**2 /planck * modify * beam * 1e26
    return S



nu = 13.7
flux_13POINT7 = ame(nu=nu, beam=beam, A_AME=sed_params[1], nu_AME=sed_params[2], W_AME=sed_params[3]) + freefree(nu=nu, beam=beam, EM=sed_params[0]) + thermaldust(nu=nu, beam=beam, T_d=sed_params[4], tau=sed_params[5], beta=sed_params[6])


nu = 4.85
flux_4POINT85 = ame(nu=nu, beam=beam, A_AME=sed_params[1], nu_AME=sed_params[2], W_AME=sed_params[3]) + freefree(nu=nu, beam=beam, EM=sed_params[0]) + thermaldust(nu=nu, beam=beam, T_d=sed_params[4], tau=sed_params[5], beta=sed_params[6])


nu = 30
flux_30 = ame(nu=nu, beam=beam, A_AME=sed_params[1], nu_AME=sed_params[2], W_AME=sed_params[3]) + freefree(nu=nu, beam=beam, EM=sed_params[0]) + thermaldust(nu=nu, beam=beam, T_d=sed_params[4], tau=sed_params[5], beta=sed_params[6])



print(f'Flux at 13.7 GHz is {flux_13POINT7} Jy')
print(f'Flux at 4.85 GHz is {flux_4POINT85} Jy')
print(f'Flux at 30 GHz is {flux_30} Jy')
