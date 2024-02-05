import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')

# Settings

frequency = 22.8
phi = [-164.3]
theta = [-11.6]
reson = 3.4

# Read both dust 857 map in MJy/sr 1deg smoothed and AME map at given frequency
planck_857 = hp.read_map('/scratch/nas_falcon/scratch/rca/data/ancillary_data/CMBon/HFI_SkyMap_857_512_smth.fits', verbose=False) # in MJysr
my_ame = hp.read_map(f'./lambda_orionis_ame_{frequency}GHz_1deg.fits', verbose=False) # in Jy

# Downgrade 857 to NSIDE 256
planck_857 = hp.pixelfunc.ud_grade(planck_857, nside_out=256)


''' Convert my map to microjanksys '''

# First convert to MJy/sr

photometry_beam = 0.000239 # sr
my_ame = np.divide(my_ame,photometry_beam) # Jy/sr
my_ame = np.divide(my_ame,1e6) # MJy/sr


def Jy2K(nu, beam):
    '''
    nu - in GHz
    beam - in steradians
    '''
    c = 299792458.
    k = 1.3806488e-23

    T = (c**2) / (2*k*((nu*1e9)**2)*beam*1e26) # conversion factor from jy to K

    return T


def convert_to_mK(nu, nside, units):
    '''
    Convert map to mK units
    '''

    pixbeam = 4.*np.pi/(12.*nside**2) # in steradians

    # conversion factors below are the number that the map must be
    # multiplied by to convert it to mK

    conversions = {'K': 1e3,
                   'mK_RJ': 1.,
                   'mK': 1.,
                   'MJysr': 1e6*pixbeam*Jy2K(nu, pixbeam)*1e3}

    return conversions[units]

# Convert to mK
my_ame = my_ame*convert_to_mK(nu=frequency, nside=256, units='MJysr') # mK

# Convert to uK (micro kelvin)
my_ame = my_ame*1e3 # uK

# Filter out values of my AME map smaller than about 1e-3
planck_857[my_ame<1] = np.nan
my_ame[my_ame<1] = np.nan
planck_857[np.isnan(my_ame)] = np.nan


# Plot both side to side in NSIDE256
hp.gnomview(my_ame, rot=[phi, theta],reso=reson, title=f'Ame {frequency} GHz (microK)')
hp.gnomview(planck_857, rot=[phi, theta],reso=reson, title=f'Planck 857 GHz (MJysr)')
plt.show()


errors = np.divide(1,my_ame)

# Fit gradient to get best fit value
from fit_leastsq import fit_leastsq
p, p_err, fitted_function, chi_sq = fit_leastsq(x=planck_857[~np.isnan(planck_857)], y=my_ame[~np.isnan(planck_857)], modelname='poly1', initial_guess=None, errors=errors[~np.isnan(planck_857)])


# Show T-T plot between the two
def poly1(x, p0, p1):
	return np.multiply(p1,x) + p0
plt.plot(planck_857, my_ame,'.')
xmodel = [np.nanmin(planck_857), np.nanmax(planck_857)]
ymodel = poly1(xmodel, p[0], p[1])
plt.plot(xmodel, ymodel, 'k--')
plt.xlabel('Planck (MJysr)')
plt.ylabel('AME (uK)')
print(p,p_err, chi_sq)
plt.show()
