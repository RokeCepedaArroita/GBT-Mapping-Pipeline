''' Read TODs and call Stuart's mapmaker '''

import numpy as np
import matplotlib.pyplot as plt
from config import settings
import healpy as hp

#
# # Read data
# def read_tod(name, session, object=None):
#     ''' Reads in the TOD and passes a TOD dictionary'''
#
#     import os
#     import h5py
#
#     def keys(f): # function to list the keys of an h5 object
#         return [key for key in f.keys()]
#
#     def h5decode(stringlist):
#         return [n.decode("ascii", "ignore") for n in stringlist]
#
#     # If fits file doesn't exist, warn the user
#     h5_filename = f'./tods/TOD_Session{session}_{name}.h5'
#     if not os.path.isfile(h5_filename):
#         import sys
#         print(f'ERROR: Filename {h5_filename} does not exist! Check the directory or run todmaker()')
#         sys.exit(1)
#
#     # Open the file and pass the dictionary containing all the information
#     else:
#         print(f'Reading TOD {h5_filename}...')
#         TOD = {} # initialise empty dictionary
#         hf = h5py.File(h5_filename, 'r')
#
#         # Set up slicing so that we can select a given object
#         if object is None:
#             objectindexes = np.arange(len(h5decode(np.array(hf['object']))))
#         else:
#             objectindexes = [i for i, item in enumerate(h5decode(np.array(hf['object']))) if item==f'{object}']
#             if objectindexes == []:
#                 import sys
#                 print(f'ERROR: Object \'{object}\' does not exist! Make sure you select an object in the data')
#                 sys.exit(1)
#
#         # Copy all the keys to a TOD dictionary
#         from tqdm import tqdm
#         for key in tqdm(keys(hf)): # copy every key
#             if key in ['average_frequency']: # single numbers require no slicing
#                 TOD[f'{key}'] = np.array(hf[f'{key}'])
#             else:
#                 if key in ['object','obsmode']:
#                     currentstringlist = h5decode(np.array(hf[f'{key}']))
#                     TOD[f'{key}'] =[currentstringlist[i] for i in objectindexes]
#                 else:
#                     TOD[f'{key}'] = np.array(hf[f'{key}'])[objectindexes]
#         hf.close()
#
#     return TOD
#
#
# TOD = read_tod(name='test', session=4, object='daisy_center')

# Save as a pickle
import pickle
# with open('./tods/test.pickle', 'wb') as handle:
#     pickle.dump(TOD, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./tods/test.pickle', 'rb') as handle:
    TOD = pickle.load(handle)


# Pass the data to the mapmaker

data_timestep = 0.1 # seconds
baseline_time = 67 # seconds
baseline_length = int(baseline_time/data_timestep)
nside = 2048 # 2048 gives 1.7 armin side pixels
npix = 12*nside**2

# Get the pixels
pix = hp.ang2pix(nside, (90.-TOD['dec'])*np.pi/180., TOD['ra']*np.pi/180.)

tod = TOD['tod'][~np.isinf( TOD['tod'])]
pix = pix[~np.isinf( TOD['tod'])]

plt.plot(TOD['tod'],',')
plt.show()


import sys
sys.path.append('./MapMaker')
from Destriper import Control
map = Control.Destriper(tod=tod, bl=baseline_length, pix=pix, npix=npix, maxiter=5, Verbose=True) #############

hp.write_map(filename='test.fits', m=map.m)

print(map.m)
hp.gnomview(map.m,rot=[np.nanmean(TOD['ra']),np.nanmean(TOD['dec'])], vmin=np.nanmedian(map.m)/2, vmax=np.nanmedian(map.m)*2)
plt.show()
