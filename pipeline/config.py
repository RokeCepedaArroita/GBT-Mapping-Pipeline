''' config.py: GBT pipeline configuration file '''

import numpy as np

settings = {# Default data selection, to be overwritten by some procedures
            'data': {'session':   3,               # session chosen
                     'bank':     'C',              # bank chosen
                     'scan':      124,             # scan chosen
                     'pol':       1,               # polarisation chosen
                     'hdu':      'SINGLE DISH'},   # primary data header name

            # Flagging settings
            'flagging': {# Spectral RFI cuts
                         'bandwidth': [[0,125],[1375,1500]], # bandwidth ranges to cut in MHz (mode 1 can only use the central 1250 MHz)
                         'kurtosis': 5,                      # mask all channel with more than 5 times the median kurtosis in a given scan-bank-pol
                         'varcoeff': 7.5,                    # mask all channel with more than 7.5 times the median kurtosis in a given scan-bank-pol
                         'minsep_MHz': 32,                   # minimum separation in MHz between bad channels for spectral RFI flags

                         # Noise statistic cuts, often corresponding to the 4-channel noise stats
                         'Tsys_minmax': {'C':  [15,190],        # C-band  Tsys cuts in K (i.e. anything lower than 10K and higher than 330K is thrown away)
                                         'Ku': [15,400]},       # Ku-band Tsys cuts in K
                         'dataset_nbins':  128,                 # dataset of noisestats to make cuts on, where nbins is the number of channels averaged together before the correlated noise is fitted
                         'OneOverF_timesc': 20,                 # timescale in seconds for the amplitude of 1/f to be used for flagging, it doesn't really matter as long as we are consistent in its use since the aoof cuts depend on this number
                         'OneOverF_minmax': {'C':  [0,0.75],    # cut on amplitude of 1/f in K at 60 second timescales
                                             'Ku': [0,0.75]},   # same for Ku-band
                         'alpha_minmax': {'C':  [0.5,4],        # cut on the 4-bin alpha
                                          'Ku': [0.5,4]},       # same for Ku-band
                         'fknee_minmax': {'C':  [0.01,0.5],     # cut on the 4-bin knee frequency (Hz)
                                          'Ku': [0.01,0.5]},    # same for Ku-band

                         # Dead bank/polarisation combination cuts
                         'dead_bankpols': ['E1', 'F1', 'G1', 'H1'], # mask all of these pol-bank combinations in all sessions
                         'dead_sesbanks': ['1A', '1E', '4A'],       # mask all of these session-bank combinations in all polarisations
                         'noisy_sesbankpol': ['9X1'],               # mask all of these session, bank, pol combinations because of high Tsys, not used currently

                         # Final TOD cuts
                         'min_bandwidth':         {'C':  1.0,         # minimum weighted bandwidth in GHz
                                                   'Ku': 0.5},        # same for Ku
                         'final_tsys_minmax':     {'C':  [15,800],    # cut on the final tsys in K, this K is not AT ALL related to the previous one, do not trust absolute value. Treat it as a relative one!!
                                                   'Ku': [15,1600]},  # same for Ku, this K is not AT ALL related to the previous one, do not trust absolute value. Treat it as a relative one!!
                         'final_fknee_minmax':    {'C':  [0.01,0.5],  # cut on the final knee frequency (Hz)
                                                   'Ku': [0.01,0.5]}, # same for Ku-band
                         'final_alpha_minmax':    {'C':  [0.5,4],     # cut on the final alpha
                                                   'Ku': [0.5,4]},    # same for Ku-band
                         'final_OneOverF_minmax': {'C':  [0,0.75],    # cut on the final 1/f amplitude at 60 seconds (K)
                                                   'Ku': [0,0.75]}    # same for Ku-band
                         },

            # Procedure selection (add more later)
            'procedures': {'diode':          1,     # diode calibration
                           'spectralRFI':    1,     # spectral RFI detection procedure
                           'specRFIFlag':    1,     # whether you want to copy the flags over
                           'noisestatsFlag': 1,     # whether to make cuts based on noisestats
                           'deadFlag':       1,     # whether you want to throw out dead bank/session/pol combinations
                           'ku_elevation_corr': 1}, # whether to subtract an elevation model in Ku

            # Directories and settings
            'dir': {'data': '/scratch/nas_falcon/scratch/rca/projects/gbt/gbtdata/processed/processed_data', # location of sdfits files
                    'figures': './figures',  # figure save directory
                    'results': './results'}, # results save directory

            # Plotting Settings
            'plotting': {'savefigs':          True,     # save figures
                         'showfigs':          False,    # show figures
                         'savefigs_oneoverf': False,    # save 1/f figures
                         'hits':              False,    # produce hits plot
                         'schedule':          True,     # schedule plot
                         'simple_plots':      False},   # one to one data plots

            # Verbosity Setting
            'verbose': True,

            }
