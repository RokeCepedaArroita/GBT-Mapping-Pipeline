import numpy as np
from tqdm import tqdm
from astropy.io import fits
from subprocess import call
import matplotlib.pyplot as plt

# Config
name = 'cminimisation'
filelist = 'TOD_BandC_firstC_filelist_FULL'
reference_config = './destriper_configs/default_C.ini'
baseline_lengths = np.arange(200)[2::1]

# Create init files and destripe maps one by one
MAD_mK = []
for bs in tqdm(baseline_lengths):
    print(f'Computing baseline {bs}...')
    import os
    final_map_exists = os.path.isfile(f'./maps/{name}_{bs}.fits')
    if not final_map_exists:
        # Open file
        str = open(reference_config, 'r').read()
        # Replace name and baseline length fields
        str = str.replace('putyourbaselinelengthhere', f'{bs}')
        str = str.replace('putyourfilelisthere', f'{filelist}')
        str = str.replace('putyournamehere', f'{name}_{bs}')
        with open(f'./destriper_configs/{name}_{bs}.ini', 'w') as output_file:
            output_file.write(str)
            output_file.close()

        # Run destriper
        call([f'python run_destriper.py ./destriper_configs/{name}_{bs}.ini'], shell=True)

    # Open map with astropy
    hdu_list = fits.open(f'./maps/{name}_{bs}.fits')
    image_data = hdu_list[0].data

    def MAD(d,axis=0):
        ''' Return Median Absolute Deviation for array along one axis '''
        med_d = np.nanmedian(d,axis=axis)
        rms = np.sqrt(np.nanmedian((d-med_d)**2,axis=axis))*1.48
        return rms

    MAD_mK.append(MAD(image_data.flatten())*1e3)


plt.plot(baseline_lengths,MAD_mK)
plt.title('Destriping Length Optimisation')
plt.ylabel('Pixel Scale Median Standard Deviation (mK)')
plt.xlabel('Destriping Baseline Length')
# from scipy.signal import savgol_filter
# MAD_mK_smooth = savgol_filter(MAD_mK, 11, 2) # window size 51, polynomial order 3
# plt.plot(baseline_lengths, MAD_mK_smooth, 'k--')
plt.show()




# Open maps and get noise in each



# # Merger folders if it doesn't exist
# from subprocess import call
# ''' CALL SCRIPT TO CREATE ALL PROJECTED MAPS'''
# if merged_map_exists==False:
#     print('Creating merged map since it does not exist...')
#     call([f'python ../merge_maps.py {main_folder}'], shell=True)
#     print('Done!')
