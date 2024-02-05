''' IO.py: Contains Input/Output tools '''
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt

class IO():

    def __init__(self, settings):
        ''' Initialise object '''

        # Initialise directories
        self.directories = {}

        # Copy settings
        self.settings = settings

        # Fill directories
        self.returnfiledirs(self)

        # Fill data
        self.getdata(self)

        # Fill central frequencies of each band and add flags
        self.getfrequencies(self)
        self.bandwidthcut(self)

        # Determine whether the current bank needs to be thrown out or not
        self.deadpolflag = {'0': 0,
                            '1': 0}
        if self.settings['procedures']['deadFlag']==1:
            for pol in [0,1]:
                # Flag certain polarisation/bank combinations
                if f"{self.settings['data']['bank']}{pol}" in self.settings['flagging']['dead_bankpols']:
                    self.deadpolflag[f'{pol}'] = 1
                # Flag certain session/bank combinations
                if f"{self.settings['data']['session']}{self.settings['data']['bank']}" in self.settings['flagging']['dead_sesbanks']:
                    self.deadpolflag[f'{pol}'] = 1
                # Flag certain session/bank/pol combinations
                if f"{self.settings['data']['session']}{self.settings['data']['bank']}{pol}" in self.settings['flagging']['noisy_sesbankpol']:
                    self.deadpolflag[f'{pol}'] = 1


        # Add spectral RFI flags
        if self.settings['procedures']['specRFIFlag']==1:
            self.copy_spectralRFI_flags(self)

        # Add noise statistics flags
        if self.settings['procedures']['noisestatsFlag']==1:
            try:
                self.copy_noisestats_flags(self)
            except:
                print('Could not find noisestats. You will have to create them.')


    @staticmethod
    def returnfiledirs(self):
        ''' Returns the data directories '''
        # Remember than when using f-strings and dictionaries together we need to combine single and double quotes for unambiguity
        self.directories['fitdir']   = self.settings['dir']['data'] + f"/AGBT20B_336_0{self.settings['data']['session']}.raw.vegas/AGBT20B_336_0{self.settings['data']['session']}.raw.vegas.{self.settings['data']['bank']}.fits"
        self.directories['indexdir'] = self.settings['dir']['data'] + f"/AGBT20B_336_0{self.settings['data']['session']}.raw.vegas/AGBT20B_336_0{self.settings['data']['session']}.raw.vegas.{self.settings['data']['bank']}.index"
        self.directories['flagdir']  = self.settings['dir']['data'] + f"/AGBT20B_336_0{self.settings['data']['session']}.raw.vegas/AGBT20B_336_0{self.settings['data']['session']}.raw.vegas.{self.settings['data']['bank']}.flag"
        self.directories['session_full_name'] = f"AGBT20B_336_0{self.settings['data']['session']}_BANK_{self.settings['data']['bank']}"
        # Set band name
        if self.settings['data']['session'] in [1,2,3]:
            self.settings['data']['band'] = 'Ku'
        elif self.settings['data']['session'] == 4:
            self.settings['data']['band'] = 'C'
        return

    @staticmethod
    def openfits(fitdir, hdu, verbose=True):
        '''Opens fits file'''
        from astropy.io import fits
        # Open fits file
        hdul = fits.open(fitdir)
        data = hdul[hdu].data
        # Prints info
        if verbose:
            print(hdul.info())
            print(hdul[hdu].header)
        return data

    @staticmethod
    def getdata(self):
        ''' Simple way to read in the data '''
        self.data = self.openfits(self.directories['fitdir'],hdu=self.settings['data']['hdu'], verbose=False)
        return

    @staticmethod
    def getfrequencies(self):
        ''' Fill the frequency information to a convenient dictionary '''

        # Initialise array
        self.frequencies = {'center': [],        # central frequency in GHz
                            'bandwidth': [],     # total bandwidth
                            'delta': [],         # channel width in GHz
                            'nchan': [],         # number of channels
                            'channelfreq': [],   # central frequencies of each channel in GHz
                            'channelflag': [],   # 0 for pass, 1 for bad channels
                            'spectralRFIflag': {'0': [],   # 0 for pass, 1 for bad kurtosis, 2 for bad sigma clipping
                                                '1': []} }

        # Initialise flags
        self.noiseflags = {'0': [],
                           '1': [],
                           'scan_number': []}

        # Copy values over
        self.frequencies['center'] = np.unique(self.data.field('CRVAL1'))[0]/1e9 # GHz
        self.frequencies['delta'] = np.unique(self.data.field('CDELT1'))[0]/1e9 # GHz
        self.frequencies['bandwidth'] = np.unique(self.data.field('BANDWID'))[0]/1e9 # GHz
        self.frequencies['nchan'] = np.shape(self.data.field('DATA'))[1] # number of channels
        self.frequencies['centerpix'] = np.unique(self.data.field('CRPIX1'))[0] # pixel number of the center frequency

        # The center IF frequency of each channel, i, is given by: CRVAL1 + CDELT1 * (CRPIX1 - i), where i goes from 1 to NCHAN
        self.frequencies['channelfreq'] = self.frequencies['center'] + self.frequencies['delta'] * ( (np.arange(self.frequencies['nchan']) + 1) - self.frequencies['centerpix'] )

        # Define the extent for imshow plotting (i.e. the edges of the first and last channels). Since delta can be +ive or -ive, this works
        self.frequencies['extent'] = [self.frequencies['channelfreq'][0]-self.frequencies['delta']/2, self.frequencies['channelfreq'][-1]+self.frequencies['delta']/2] # max and min in GHz

        # Initialise the RFI flags
        for pol in [0,1]:
            self.frequencies['spectralRFIflag'][f'{pol}'] = np.zeros(self.frequencies['nchan'])

        return


    @staticmethod
    def bandwidthcut(self):
        ''' Flag the edges of each bank with label 1 '''
        self.frequencies['channelflag']  = np.zeros(self.frequencies['nchan'])
        for cut_range in self.settings['flagging']['bandwidth']:
            start = cut_range[0]/1e3 + self.frequencies['center'] - self.frequencies['bandwidth']/2  # in GHz
            end   = cut_range[1]/1e3 + self.frequencies['center'] - self.frequencies['bandwidth']/2  # in GHz
            bad_channels = np.where( (self.frequencies['channelfreq'] > start) & (self.frequencies['channelfreq'] < end) )[0]
            self.frequencies['channelflag'][bad_channels] = 1
        return


    def update_spectralRFI(self, session, bank):
        ''' Writes an HDF5 dataset containing the full spectral RFI
        statistics for a given bank in ./rfi/SpectralRFI_Bank*.h5.
        Example: kurtosis has size (2,NSCANS,1024) where 2 is the
        number of polarizations and 1024 is the number of channels'''

        # TODO: paralellize this so that it takes 40 min in total (currently ~40 min per bank in series)

        import os
        import h5py

        # Change settings for one bank
        from config import settings
        settings['data']['bank'] = bank
        settings['data']['session'] = session
        custom = IO(settings)

        # Initialise arrays
        scans = np.unique(self.data.field('SCAN'))
        nchans = len(custom.frequencies['channelfreq'])

        # If fits file doesn't exist, create one with an empty dictionary with the correct sizes
        h5_filename = f'./rfi/SpectralRFI_Session{session}_Bank{bank}.h5'
        if not os.path.isfile(h5_filename):
            hf = h5py.File(h5_filename, 'w')
            hf.create_dataset('scan_number', data=scans)
            hf.create_dataset('channelfreq', data=custom.frequencies['channelfreq'])
            hf.create_dataset('channelflag', data=custom.frequencies['channelflag'])
            hf.create_dataset('delta', data=custom.frequencies['delta'])
            hf.create_dataset('extent', data=custom.frequencies['extent'])
            hf.create_dataset('kurtosis',              data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('variation_coefficient', data=np.zeros([2,len(scans),nchans]))
            print(f'Initialised h5py file in {h5_filename}')
            hf.close()

        # For every scan, calculate and write RFI stats
        from tqdm import tqdm
        for i, scan in enumerate(tqdm(scans)):
            # Update scan number
            settings['data']['scan'] = scan
            # Get statistics
            from processing import Processing
            process = Processing(settings)
            spectral_kurtosis, variation_coefficient = process.spectralRFI()
            # Modify file
            hf = h5py.File(h5_filename, 'a')
            hf['kurtosis'][0,i,:] = spectral_kurtosis['0']
            hf['kurtosis'][1,i,:] = spectral_kurtosis['1']
            hf['variation_coefficient'][0,i,:] = variation_coefficient['0']
            hf['variation_coefficient'][1,i,:] = variation_coefficient['1']
            hf.close()

        # Final message
        print(f'Saved RFI for S{session} Bank{bank} in {h5_filename}')

        return



    def read_spectralRFI(self, session, bank):
        ''' Reads the RFI statistics files '''

        import os
        import h5py

        def keys(f): # function to list the keys of an h5 object
            return [key for key in f.keys()]

        # If fits file doesn't exist, warn the user
        h5_filename = f'./rfi/SpectralRFI_Session{session}_Bank{bank}.h5'
        if not os.path.isfile(h5_filename):
            import sys
            print(f'ERROR: Filename {h5_filename} does not exist! Check the directory or run update_spectralRFI()')
            sys.exit(1)

        # Open the file and pass the dictionary containing all the information
        else:
            SpectralRFI = {} # initialise empty dictionary
            hf = h5py.File(h5_filename, 'r')
            for key in keys(hf): # copy every key
                SpectralRFI[f'{key}'] = np.array(hf[f'{key}'])
            hf.close()

        return SpectralRFI


    def read_noisestats(self, session, bank, nchannels):
        ''' Reads the RFI statistics files '''

        import os
        import h5py

        def keys(f): # function to list the keys of an h5 object
            return [key for key in f.keys()]

        def h5decode(stringlist): # function to read h5 ascii lists
            return [n.decode("ascii", "ignore") for n in stringlist]

        # If fits file doesn't exist, warn the user
        h5_filename = f'./noise/NoiseStats_Session{session}_Bank{bank}_Bin{nchannels}.h5'
        if not os.path.isfile(h5_filename):
            import sys
            print(f'ERROR: Could not read noise stats {h5_filename} because they do not exist! Check the directory or run noisestats()')
            sys.exit(1)

        # Open the file and pass the dictionary containing all the information
        else:
            NoiseStats = {} # initialise empty dictionary
            hf = h5py.File(h5_filename, 'r')
            for key in keys(hf): # copy every key
                if key in ['object','obsmode']:
                    NoiseStats[f'{key}'] = h5decode(np.array(hf[f'{key}']))
                else:
                    NoiseStats[f'{key}'] = np.array(hf[f'{key}'])
            hf.close()

        return NoiseStats


    @staticmethod
    def copy_spectralRFI_flags(self):
        ''' Read spectral RFI data files and apply flags based on that '''

        # Read the RFI data
        SpectralRFI = self.read_spectralRFI(session=self.settings['data']['session'], bank=self.settings['data']['bank'])

        # Assign flags
        for pol in [0,1]:
            # Extract single scan and median accross the entire session
            varcoeff = SpectralRFI['variation_coefficient'][pol,self.settings['data']['scan'],:]
            kurtosis = SpectralRFI['kurtosis'][pol,self.settings['data']['scan'],:]
            median_varcoeff = np.nanmedian(SpectralRFI['variation_coefficient'][pol,:,:])
            median_kurtosis = np.nanmedian(SpectralRFI['kurtosis'][pol,:,:])
            # Calculate and write the flags to self
            self.frequencies['spectralRFIflag'][f'{pol}'][kurtosis>=self.settings['flagging']['kurtosis']*median_kurtosis] = 1
            self.frequencies['spectralRFIflag'][f'{pol}'][varcoeff>=self.settings['flagging']['varcoeff']*median_varcoeff] = 2
            # If any two bad frequencies are separated by less than minsep_MHz, flag those as bad data too
            min_nchannel_separation = int(np.ceil(self.settings['flagging']['minsep_MHz']/np.abs(self.frequencies['delta']*1e3))) # minimum number of channels
            def patchflags(array, minimum_separation):
                ''' Set flag=3 to any short spaces in-between '''
                window = np.convolve(array, np.ones(minimum_separation), 'same')
                array[(window!=0) & (array==0)] = 3
                return array
            self.frequencies['spectralRFIflag'][f'{pol}'] = patchflags(self.frequencies['spectralRFIflag'][f'{pol}'], min_nchannel_separation)
        return


    @staticmethod
    def copy_noisestats_flags(self):
        ''' Read spectral RFI data files and apply flags based on that '''

        # Read the noise statistics data
        NoiseStats = self.read_noisestats(session=self.settings['data']['session'], bank=self.settings['data']['bank'], nchannels=self.settings['flagging']['dataset_nbins'])

        # For each polarization convert to 0=Pass, 1=Tsys failure, 2=1/f failure, 3=invalid noise measurement, ... alpha and fknee too
        for pol in [0,1]:

            # Extract Tsys, fknee, alpha
            tsys  = NoiseStats['tsys'][pol,:,:]
            fknee = NoiseStats['fknee'][pol,:,:]
            alpha = NoiseStats['alpha'][pol,:,:]

            # Get amplitude of 1/f at certain timelength, and take square root so that it is in units K
            def OneOverF(freq, alpha, whitenoise, fknee):
                ''' 1/f noise model '''
                return np.multiply( np.power(whitenoise,2) , np.add(1, np.power(np.divide(fknee, freq), alpha)))
            Aoof = np.sqrt(OneOverF(freq=1./self.settings['flagging']['OneOverF_timesc'], alpha=NoiseStats['alpha'][pol,:,:], whitenoise=NoiseStats['whitenoise'][pol,:,:], fknee=NoiseStats['fknee'][pol,:,:]))

            # Initialise noiseflags
            self.noiseflags[f'{pol}'] = np.zeros(np.shape(tsys))

            # Calculate and write the flags to self
            self.noiseflags[f'{pol}'][(tsys<self.settings['flagging']['Tsys_minmax'][self.settings['data']['band']][0]) | (tsys>self.settings['flagging']['Tsys_minmax'][self.settings['data']['band']][1])] = 1
            self.noiseflags[f'{pol}'][(Aoof<self.settings['flagging']['OneOverF_minmax'][self.settings['data']['band']][0]) | (Aoof>self.settings['flagging']['OneOverF_minmax'][self.settings['data']['band']][1])] = 2
            self.noiseflags[f'{pol}'][(fknee<self.settings['flagging']['fknee_minmax'][self.settings['data']['band']][0]) | (fknee>self.settings['flagging']['fknee_minmax'][self.settings['data']['band']][1])] = 4
            self.noiseflags[f'{pol}'][(alpha<self.settings['flagging']['alpha_minmax'][self.settings['data']['band']][0]) | (alpha>self.settings['flagging']['alpha_minmax'][self.settings['data']['band']][1])] = 5
            self.noiseflags['scan_number'] = NoiseStats['scan_number']

            # Also use 3 to patch any data that does not have a number for Aoof or Tsys
            self.noiseflags[f'{pol}'][(np.isinf(Aoof)) | (np.isinf(tsys)) | (np.isnan(Aoof)) | (np.isnan(tsys))] = 3

            # TODO: add padding here so if two channels are flagged with a small gap inbetween everything gets flagged
        return



    def read_tod(self, name, session, feed, object=None, bad_scan_ranges=None, read_destriping_tod=False):
        ''' Reads in the TOD and passes a TOD dictionary'''

        import os
        import h5py

        def keys(f): # function to list the keys of an h5 object
            return [key for key in f.keys()]

        def h5decode(stringlist):
            return [n.decode("ascii", "ignore") for n in stringlist]

        # If fits file doesn't exist, warn the user
        if read_destriping_tod:
            h5_filename = f'./tods/TOD_Session{session}_feed{feed}_{name}_FULL_DES.h5'
        else: # if we just want the raw version
            h5_filename = f'./tods/TOD_Session{session}_feed{feed}_{name}_RAW.h5'
        if not os.path.isfile(h5_filename):
            import sys
            print(f'ERROR: Filename {h5_filename} does not exist! Check the directory or run todmaker()')
            sys.exit(1)

        else: # Open the file and pass the dictionary containing all the information
            print(f'Reading TOD {h5_filename}...')
            TOD = {} # initialise empty dictionary
            hf = h5py.File(h5_filename, 'r')

            # Set up slicing so that we can select a given object
            if object is None:
                objectindexes = np.arange(len(h5decode(np.array(hf['object']))))
            else:
                objectindexes = [i for i, item in enumerate(h5decode(np.array(hf['object']))) if item==f'{object}']
                if objectindexes == []:
                    import sys
                    print(f'ERROR: Object \'{object}\' does not exist! Make sure you select an object in the data')
                    sys.exit(1)

            # Filter bad scans
            if bad_scan_ranges is not None:
                scan_numbers = np.array(hf['scan_number'])
                scan_indexes = np.ones(len(h5decode(np.array(hf['object']))))
                for current_range in bad_scan_ranges:
                    print(f'Excluding scans {current_range[0]} to {current_range[1]}.')
                    scan_indexes[(scan_numbers>current_range[0]) & (scan_numbers<current_range[1])] = 0
                # Merge bad scans flag into objectindexes
                objectindexes = [i for i, item in enumerate(h5decode(np.array(hf['object']))) if (item==f'{object}') & (scan_indexes[i]==1)]

            # Copy all the keys to a TOD dictionary
            from tqdm import tqdm
            for key in tqdm(keys(hf)): # copy every key
                if key in ['object','obsmode']:
                    currentstringlist = h5decode(np.array(hf[f'{key}']))
                    TOD[f'{key}'] =[currentstringlist[i] for i in objectindexes]
                else:
                    TOD[f'{key}'] = np.array(hf[f'{key}'])[objectindexes]
            hf.close()

        return TOD


    def merge_filelists(self, band, name):
        ''' Takes the filelists from several feed/session combinations and merges them
        into a single filelist that can be called from the mapmaker '''

        if band=='C':
            sessions = [4]
            feeds = [1]
        elif band=='Ku':
            sessions = [1,2,3]
            feeds = [1,2]
        else:
            import sys
            print(f'ERROR: band needs to be either \'C\' or \'Ku\' for merge_filelists().')
            sys.exit(1)

        read_files = []
        for session in sessions:
            for feed in feeds:
                # List of files to read
                read_files.append(f'./tods/TOD_Session{session}_feed{feed}_{name}_filelist.txt')

        with open(f'./tods/TOD_Band{band}_{name}_filelist_FULL.txt', 'wb') as outfile:
            for f in read_files:
                with open(f, 'rb') as infile: # read current text file
                    outfile.write(infile.read()) # append to master filelist

        return




    def shape(self,element):
        ''' Get shape of an element '''
        return print(f'{element} has shape {np.shape(self.data.field(element))}')

    def values(self,element):
        ''' Print all the values of an element '''
        return print(f'{element} has values {self.data.field(element)}...')

    def unique(self,element):
        ''' Print the unique values of an element '''
        return print(f'{element} has elements {np.unique(self.data.field(element))}')




    def frequency_noise(self):
        ''' Get a a mean and standard deviation per channel using auto subtraction '''
        import matplotlib.pyplot as plt


        scan_data         = self.data.field('DATA')  [ np.where(self.data.field('SCAN') == self.settings['data']['scan'])[0] ]
        calibration_flag  = self.data.field('CAL')   [ np.where(self.data.field('SCAN') == self.settings['data']['scan'])[0] ]
        polarisation_flag = self.data.field('PLNUM') [ np.where(self.data.field('SCAN') == self.settings['data']['scan'])[0] ]

        cal_on_loc  = np.where(calibration_flag=='T')[0]
        cal_off_loc = np.where(calibration_flag=='F')[0]

        def std(data,pol,cal):
            ''' Calculates noise by autosubtraction '''
            y = data[(polarisation_flag==pol)&(calibration_flag==cal),:]
            noise = np.nanstd(np.diff(y,axis=0),axis=0)
            return noise

        noise = std(data=scan_data,pol=self.settings['data']['pol'],cal='F')


        def plot_noise(noise):
            ''' Plot frequency noise with red showing parts of the spectrum that are not used '''
            start = 0
            jumplist = np.where(np.diff(self.frequencies['channelflag']))[0]
            jumplist = np.append(jumplist , len(self.frequencies['channelflag']))
            used_zero = False
            used_one  = False
            for jump in jumplist:
                end = jump
                if self.frequencies['channelflag'][start]==0:
                    if not used_zero:
                        plt.plot(self.frequencies['channelfreq'][start:end], noise[start:end], color='C2', label='Kept')
                        used_zero = True
                    else:
                        plt.plot(self.frequencies['channelfreq'][start:end], noise[start:end], color='C2')
                if self.frequencies['channelflag'][start]==1:
                    if not used_one:
                        plt.plot(self.frequencies['channelfreq'][start:end], noise[start:end], color='C3', label='Flagged')
                        used_one = True
                    else:
                        plt.plot(self.frequencies['channelfreq'][start:end], noise[start:end], color='C3')
                start = jump+1
            plt.title(f"RFI Identification: Scan {self.settings['data']['scan']}, Pol. {self.settings['data']['pol']}")
            plt.xlabel('Channel Central Frequency (GHz)')
            plt.ylabel('White Noise (RAW Units)')
            plt.legend()
            if self.settings['plotting']['savefigs']:
                plt.savefig(f"./figures/rfi/RFI_S{self.settings['data']['session']}_Scan{self.settings['data']['scan']}_Pol{self.settings['data']['pol']}.png")
            #plt.show()
            return


        plot_noise(noise)

        def select(data,pol,cal,chan):
            x = np.where((polarisation_flag==pol)&(calibration_flag==cal))[0]
            y = data[(polarisation_flag==pol)&(calibration_flag==cal),chan]
            return x, y

        # ALSO FILTER BY POLARISATION!
        plt.plot( select(scan_data,0,'T',500)[0], select(scan_data,0,'T',500)[1]  ,'g.', label='Calibration ON, POL0')
        plt.plot( select(scan_data,0,'F',500)[0], select(scan_data,0,'F',500)[1]  ,'r.', label='Calibration OFF, POL0')
        plt.plot( select(scan_data,1,'T',500)[0], select(scan_data,1,'T',500)[1]  ,'g4', label='Calibration ON, POL1')
        plt.plot( select(scan_data,1,'F',500)[0], select(scan_data,1,'F',500)[1]  ,'r4', label='Calibration OFF, POL1')


        pol0_on  = np.array([select(scan_data,0,'T', chan)[1] for chan in np.arange(1024)])
        pol0_off = np.array([select(scan_data,0,'F', chan)[1] for chan in np.arange(1024)])
        difference = pol0_on-pol0_off


        cal_data = 6.*pol0_off/difference

        # rms = np.nanstd(cal_data[:,0::2] - cal_data[:,1::2],axis=1) # rms excluding system gain
        #
        # Tsys = rms/(np.sqrt(2)/(np.sqrt(0.05)*np.sqrt(self.frequencies['delta']*1e9)))
        # plt.close()
        # plt.plot(Tsys)
        # plt.ylim([0,500])
        # plt.show()
        #
        #
        # plt.close()

        plt.close()
        #print(np.nanstd(cal_data[750,:])/(np.sqrt(2)/(np.sqrt(0.05)*np.sqrt(self.frequencies['delta']*1e9)))

        #plt.plot(cal_data[750,:])

        plt.imshow(cal_data,aspect='auto',vmin=10,vmax=100)




        plt.show()

        # frequency_mean_data = np.nanmean(scan_data,axis=1)
        # frequency_mean_data = frequency_mean_data.reshape(int(len(frequency_mean_data)/4),4)
        # pol_mean_data = np.nanmean(frequency_mean_data,axis=1)
        #
        # time = np.arange(len(pol_mean_data))*0.1/60
        # plt.plot(time, pol_mean_data)
        # plt.title(f"Scan number {self.settings['data']['scan']}")
        # plt.xlabel(f'Time (min)')
        # plt.show()



        return




    @staticmethod
    def load_galactic_coordinates(self):
        ''' Creates Galactic coordinates and saves them to a fileself.
        If the file already exists, it just reads them in '''

        import os.path
        gal_coords_file = f"{self.settings['dir']['data']}/AGBT20B_336_0{self.settings['data']['session']}.raw.vegas/AGBT20B_336_0{self.settings['data']['session']}_galactic_coordinates.txt"
        if os.path.isfile(gal_coords_file): # If file with l & b exists, then skip conversion step and just load l and b
            print(f"Galactic coordinates found for session {self.settings['data']['session']}, reading them in.")
            l, b = np.loadtxt(gal_coords_file)

        else: # If no l & b are found, then create them and save them
            print(f"No file for Galactic coordinates found for session {self.settings['data']['session']}, creating them.")
            l, b = AltAz2Galactic(time=self.data.field('DATE-OBS'), alt=self.self.data.field('ELEVATIO'), az=self.self.data.field('AZIMUTH'))
            np.savetxt(gal_coords_file, [l, b])

        return l, b


    @staticmethod
    def load_all_galactic_coordinates(self, band, object_name='daisy_center'):
        ''' Returns full l, b on a given source if the files already exist
        for galactic coordinates. Bands can be 'C' and 'Ku' '''

        sessions = {'C': [4],
                   'Ku': [1,2,3]}

        # Concatenate coordinates
        import os.path
        l = []
        b = []
        for session in sessions[band]:
            data = getdata(self.settings['dir']['data'], session, bank='A') # we are only interested in coordinates, so I set bank to A
            gal_coords_file = f"{self.settings['dir']['data']}/AGBT20B_336_0{session}.raw.vegas/AGBT20B_336_0{session}_galactic_coordinates.txt"
            if os.path.isfile(gal_coords_file): # If file with l & b exists, then skip conversion step and just load l and b
                print(f"Galactic coordinates found for session {session}, reading them in.")
                l_temp, b_temp = np.loadtxt(gal_coords_file)
            l.append(l_temp[self.data.field('OBJECT')==object_name])
            b.append(b_temp[self.data.field('OBJECT')==object_name])

        return l[0],b[0]
