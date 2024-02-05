import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')


''' CONFIGURATION '''

# Frequency to evaluate curve at
frequency_of_choice = 30 # GHz

# Directory where parameter file MERGED.fits is at NSIDE256
main_folder = 'alltogether3nofilter'

# Beam of my data
data_beam = 0.00023924596203935044 # sr

# Masking radius
masking_radius = 5.5

# Plotting settings
use_planck_cmap = True
phi = [-164.3]
theta = [-11.6]
reson = 3.4

# Savedir
savedir = '.'



''' PROGRAM STARTS HERE '''

# Forced setting
nside = 256

# Open my data
parameter_map = '/scratch/nas_falcon/scratch/rca/projects/lambda_orionis/maps/PAPER/'+main_folder+'/MERGED.fits'
m_AME = hp.fitsfunc.read_map(parameter_map, 2, verbose=False)
m_AME_err = hp.fitsfunc.read_map(parameter_map, 3, verbose=False)
m_nu = hp.fitsfunc.read_map(parameter_map, 4, verbose=False)
m_nu_err = hp.fitsfunc.read_map(parameter_map, 5, verbose=False)
m_nu_width = hp.fitsfunc.read_map(parameter_map, 10, verbose=False)
m_nu_width_err = hp.fitsfunc.read_map(parameter_map, 11, verbose=False)
m_EM = hp.fitsfunc.read_map(parameter_map, 12, verbose=False)
m_EM_err = hp.fitsfunc.read_map(parameter_map, 13, verbose=False)



def distancemap(pixel_lon, pixel_lat, nside, lon, lat):
    ''' Return a map of angular distances in degrees from a
    given pixel. All inputs and outputs are in degrees'''

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



lon, lat = hp.pix2ang(nside,np.arange(0,12*nside**2),lonlat=True)



def mask_radius(input_map, nside, masking_radius, lon_central, lat_central):

    # Get the coordinate grid
    lon, lat = hp.pix2ang(nside,np.arange(0,12*nside**2),lonlat=True)

    # Calculate distance map from (0,0)
    distance_map = distancemap(pixel_lon=lon_central, pixel_lat=lat_central, nside=nside, lon=lon, lat=lat)

    # Mask all values with distances > masking_radius
    masked_map = np.copy(input_map)
    masked_map[distance_map>masking_radius] = np.nan

    return masked_map


# Anomalous Microwave Emission (Lognormal Approximation)

def ame(nu, beam, A_AME, nu_AME, W_AME):
    ''' Here the beam doesn't do anything, but you still need it '''

    nlog = np.log(nu)
    nmaxlog = np.log(nu_AME)
    S = A_AME*np.exp(-0.5 * ((nlog-nmaxlog)/W_AME)**2)

    return S


# Free-free

def freefree(nu, beam, EM, T_e):

    c = 299792458.
    k = 1.3806488e-23
    h = 6.62606957e-34

    a = 0.366 * np.power(nu,0.1)* np.power(T_e,-0.15) * (np.log(np.divide(4.995e-2, nu)) + 1.5 * np.log(T_e))
    T_ff = 8.235e-2 * a * np.power(T_e,-0.35) * np.power(nu,-2.1) * (1. + 0.08) * EM

    S = 2. * k * beam * np.power(np.multiply(nu,1e9),2)  / c**2 * T_ff * 1e26

    return S


# Calculate My Fluxes at frequency_of_choice GHz

m_AME[m_AME<0] = 0.0001

my_ame = ame(frequency_of_choice, data_beam, m_AME, m_nu, m_nu_width)
my_ff = freefree(frequency_of_choice, data_beam, m_EM, 7500.)

my_ff[my_ff==0] = np.nan

# Colour scale

if use_planck_cmap:
    ############### CMB colormap
    from matplotlib.colors import ListedColormap
    planck_cmap = ListedColormap(np.loadtxt('/scratch/nas_falcon/scratch/rca/data/colourmaps/planck.txt')/255.)
    planck_cmap.set_bad('gray') # color of missing pixels
    planck_cmap.set_under('gray') # color of background, necessary if you want to use
    cmap = planck_cmap


# Crop latitudes above -7

my_ame[lat>-6.6] = np.nan
my_ff[lat>-6.6] = np.nan

# Mask maps

my_ame = mask_radius(input_map=my_ame, nside=nside, masking_radius=masking_radius, lon_central=phi, lat_central=theta)
my_ff = mask_radius(input_map=my_ff, nside=nside, masking_radius=masking_radius, lon_central=phi, lat_central=theta)


hp.write_map(f'lambda_orionis_ame_{frequency_of_choice}GHz_1deg.fits', my_ame)
hp.write_map(f'lambda_orionis_ff_{frequency_of_choice}GHz_1deg.fits', my_ff)

hp.gnomview(my_ame, rot=[phi, theta],reso=reson, title=f'ame {frequency_of_choice} GHz')
hp.gnomview(my_ff, rot=[phi, theta],reso=reson, title=f'ff {frequency_of_choice} GHz')
plt.show()
