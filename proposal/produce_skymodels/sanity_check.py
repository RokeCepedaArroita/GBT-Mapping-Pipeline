import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('mathtext',fontset='cm',rm='serif')


# Open maps

ame_model_13 = hp.read_map('./others/sky_models/ame_model_13.7GHz_mK.fits')
ame_model_5 = hp.read_map('./others/sky_models/ame_model_4.85GHz_mK.fits')
ff_model_13 = hp.read_map('./others/sky_models/free-free_model_13.7GHz_mK.fits')
ff_model_5 = hp.read_map('./others/sky_models/free-free_model_4.85GHz_mK.fits')
ff_model_23 = hp.read_map('./others/sky_models/free-free_model_22.8GHz_mK.fits')
ame_model_23 = hp.read_map('./others/sky_models/ame_model_22.8GHz_mK.fits')
ff_model_129 = hp.read_map('./others/sky_models/free-free_model_12.9GHz_mK.fits')
ame_model_129 = hp.read_map('./others/sky_models/ame_model_12.9GHz_mK.fits')

# Downgrade resolution to 1024

ff_model_13 = hp.ud_grade(ff_model_13, nside_out=2048)
ff_model_5 = hp.ud_grade(ff_model_5, nside_out=2048)
ff_model_23 = hp.ud_grade(ff_model_23, nside_out=2048)
ff_model_129 = hp.ud_grade(ff_model_129, nside_out=2048)

# Calculate totals

total_model_13 = ame_model_13+ff_model_13
total_model_5 = ame_model_5+ff_model_5
total_model_23 = ame_model_23+ff_model_23
total_model_129 = ame_model_129+ff_model_129

# hp.write_map('skymodel_13.7GHz_mK.fits',total_model_13)
# hp.write_map('skymodel_4.85GHz_mK.fits',total_model_5)


# Plot totals

hp.gnomview(total_model_13, rot=[-164.3, -11.6],reso=3.4/10, xsize=2000, title=f'Total Simulated 13.7 GHz (mK)')
plt.savefig('Total Simulated 13.7 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(total_model_5, rot=[-164.3, -11.6],reso=3.4/10, xsize=2000, title=f'Total Simulated 4.85 GHz (mK)')
plt.savefig('Total Simulated 4.85 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
#plt.show()

#
# hp.gnomview(total_model_13, rot=[-155.299, -11.80],reso=1/10, xsize=2000, title=f'LDN1622 Total Simulated 13.7 GHz (mK)')
# hp.gnomview(total_model_5, rot=[-155.299, -11.80],reso=1/10, xsize=2000, title=f'LDN1622 Total Simulated 4.85 GHz (mK)')
# plt.show()


# Smooth to 1deg FHWM

def smooth_map(m, fwhm_final, fwhm_map):
    ''' All input angles in degrees '''

    # Calculate transfer FWHM
    fwhm = np.sqrt(fwhm_final**2-fwhm_map**2)

    # Filter map
    m2 = np.copy(m)
    m[np.isnan(m2)] = 0

    # Smooth
    smoothed_map = hp.smoothing(m, fwhm=fwhm*np.pi/180.)
    smoothed_map[np.isnan(m2)] = np.nan

    return smoothed_map

total_model_13_1deg = smooth_map(m=total_model_13, fwhm_final=1.0, fwhm_map=5./60)
total_model_5_1deg = smooth_map(m=total_model_5, fwhm_final=1.0, fwhm_map=5./60)
total_model_23_1deg = smooth_map(m=total_model_23, fwhm_final=1.0, fwhm_map=5./60)
total_model_129_1deg = smooth_map(m=total_model_129, fwhm_final=1.0, fwhm_map=5./60)


# Open WMAP 22.8

wmap23 = hp.read_map('/scratch/nas_falcon/scratch/rca/data/ancillary_data/CMBoff/wmap_band_smth_imap_r9_9yr_K_v5.fits')

# Convert to milikelvin

def planckcorr(nu_in):
    '''
    Expects nu in GHz
    '''
    c = 299792458.
    k = 1.3806488e-23
    h = 6.62606957e-34
    T_cmb = 2.725

    nu = nu_in * 1e9

    x = h*nu/k/T_cmb

    return x**2*np.exp(x)/(np.exp(x) - 1.)**2


def convert_to_mK(nu, units, nside):
    '''
    Convert map to mK units
    '''

    pixbeam = 4.*np.pi/(12.*nside**2) # in steradians


    # conversion factors below are the number that the map must be
    # multiplied by to convert it to mK

    conversions = {'K': 1e3,
                   'mK_RJ': 1.,
                   'mK': 1.,
                   'mKCMB': planckcorr(nu),
                   'KCMB': planckcorr(nu)*1e3}

    return conversions[units]

print(convert_to_mK(nu=22.8, units='mKCMB', nside=512))
wmap23 = wmap23*convert_to_mK(nu=22.8, units='mKCMB', nside=512)


# Open QUIJOTE map

quijote129 = hp.read_map('/scratch/nas_falcon/scratch/rca/data/quijote/release_nov2019/quijote_1deg_13GHz.fits')
quijote129 = quijote129*1e3 # mK

# Open C-BASS map

cbass = hp.read_map('/scratch/nas_falcon/scratch/rca/data/cbass/science_maps/v28allels/NIGHTMERID20/filtered_map/tauA_cal_NIGHTMERID20_v28allelsNs_ALL_NIGHTMERID20_noiseCut_masked5pc_G_1024_ol500_lessTol_g_map_g_1deg_0512_wiener.fits')
cbass = cbass*1e3 # mK

# Plot totals

commander_total_5 = hp.read_map('./others/sky_models/ancillary_data/commander_total_fg_4.76.fits')
commander_total_13 = hp.read_map('./others/sky_models/ancillary_data/commander_total_fg_13.0.fits')
commander_total_23 = hp.read_map('./others/sky_models/ancillary_data/commander_total_fg_22.8.fits')

hp.gnomview(total_model_13_1deg, rot=[-164.3, -11.6],reso=3.4, title=f'Total Simulated 13.7 GHz (mK)')
plt.savefig('1deg Total Simulated 13.7 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(total_model_5_1deg, rot=[-164.3, -11.6],reso=3.4, min=0, max=60, title=f'Total Simulated 4.85 GHz (mK)')
plt.savefig('1deg Total Simulated 4.85 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(total_model_23_1deg, rot=[-164.3, -11.6],reso=3.4, min=0.5, max=2.8, title=f'Total Simulated 22.8 GHz (mK)')
plt.savefig('1deg Total Simulated 22.8 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(total_model_129_1deg, rot=[-164.3, -11.6],reso=3.4, min=1, max=9, title=f'Total Simulated 12.9 GHz (mK)')
plt.savefig('1deg Total Simulated 12.9 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(commander_total_13, rot=[-164.3, -11.6],reso=3.4, min=1, max=9, title=f'Total COMMANDER 13.0 GHz (mK)')
plt.savefig('1deg Total COMMANDER 12.9 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(commander_total_5, rot=[-164.3, -11.6],reso=3.4, min=0, max=60, title=f'Total COMMANDER 4.76 GHz (mK)')
plt.savefig('1deg Total COMMANDER 4.76 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(commander_total_23, rot=[-164.3, -11.6],reso=3.4, min=0.5, max=2.8, title=f'Total COMMANDER 22.8 GHz (mK)')
plt.savefig('1deg Total COMMANDER 22.8 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(wmap23, rot=[-164.3, -11.6],reso=3.4, min=0.5, max=2.8, title=f'WMAP 22.8 GHz (mK)')
plt.savefig('1deg WMAP 22.8 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(quijote129, rot=[-164.3, -11.6],reso=3.4, min=1, max=9, title=f'QUIJOTE 12.9 GHz (mK)')
plt.savefig('1deg QUIJOTE 12.9 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
hp.gnomview(cbass, rot=[-164.3, -11.6],reso=3.4, min=0, max=60, title=f'C-BASS 4.76 GHz (mK)')
plt.savefig('1deg C-BASS 4.76 GHz.png', dpi=600, bbox_inches='tight', pad_inches=0)
#plt.show()
