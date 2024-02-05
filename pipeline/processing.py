from IO import IO
import matplotlib.pyplot as plt
import numpy as np

class Processing(IO):


    def __init__(self, settings):
        ''' Initialise object, pass independent instance of IO class '''
        # super().__init__(settings) # copies all the IO attibutes to self, but this doesn't work when inheriting multiple classes
        IO.__init__(self, settings)   # use this when inheriting multiple parent classes
        return


    def diode(self):
        ''' Calibrates the diode to Tsys and returns the
        calibrated pol dictionary with all the data '''

        # Rename relevant columns
        data              = self.data.field('DATA')
        calibration_flag  = self.data.field('CAL')
        polarisation_flag = self.data.field('PLNUM')
        diode_temperature = self.data.field('TCAL')

        # Initialise calibrated polarisation object
        cal = {'0':    [], # left polarisation data
               '1':    [], # right polarisation data
               'scan': []} # scan number

        # Data selection
        def select(data,pol,cal):
            return data[(polarisation_flag==pol)&(calibration_flag==cal)]

        # Apply diode calibration: T_sys = T_cal * OFF/(ON-OFF)
        for pol_number in [0,1]:
            TCAL = select(diode_temperature, pol_number, 'T')
            OFF  = select(data, pol_number, 'F')
            ON   = select(data, pol_number, 'T')
            TCAL = np.resize(TCAL,np.shape(OFF)) # get TCAL to the right shape
            with np.errstate(divide='ignore'):
                cal[f'{pol_number}'] = TCAL * OFF/(ON-OFF)

        # Add extra information
        cal['scan'] = select(self.data.field('SCAN'), 1, 'F') # scan number is shared across both pols

        return cal




    def spectralRFI(self):
        ''' Calculate the coefficient of variation for
        a particular scan in a particular session '''

        # Get diode-calibrated data
        cal = self.diode()

        # Initialize statistics
        spectral_kurtosis =     {'0': [],
                                 '1': []}
        variation_coefficient = {'0': [],
                                 '1': []}

        for pol in [0,1]: # calculate statitistics for each polarization

            # Coefficient of variation
            mean = np.nanmean(cal[f'{pol}'],axis=0) # possibly can make this a median, although they look pretty similar
            square_diff = (cal[f'{pol}']-mean)**2
            rms = np.sqrt(np.nanmean(square_diff,axis=0)) # this should definitely be a mean
            variation_coefficient[f'{pol}'] = rms/mean

            # Spectral Kurtosis
            spectral_kurtosis[f'{pol}'] = np.nanmean(square_diff**2,axis=0)/rms**4

        return spectral_kurtosis, variation_coefficient


    def noisestats(self, session, bank, nchannels=1):
        ''' Calculate the noise properties for each
         channel for each scan in a particular session
         nchannels decices how many channels to
         measure the 1/f over'''

        import os
        import h5py

        # Change settings for one bank
        from config import settings
        settings['data']['bank'] = bank
        settings['data']['session'] = session
        custom = IO(settings)
        process = Processing(settings)

        # Useful stat functions from Stuart's COMAP pipeline
        def MAD(d,axis=0):
            ''' Return Median Absolute Deviation for array along one axis '''
            med_d = np.nanmedian(d,axis=axis)
            rms = np.sqrt(np.nanmedian((d-med_d)**2,axis=axis))*1.48
            return rms

        def AutoRMS(tod):
            ''' Auto-differenced RMS '''
            # Compute RMS
            if len(tod.shape) == 2:
                N = (tod.shape[0]//2)*2
                diff = tod[1:N:2,:] - tod[:N:2,:]
                rms = np.nanstd(diff,axis=0)/np.sqrt(2)
            else:
                N = (tod.size//2)*2
                diff = tod[1:N:2] - tod[:N:2]
                rms = np.nanstd(diff)/np.sqrt(2)
            return rms

        def TsysRMS(tod,sample_duration,bandwidth):
            ''' Calculate Tsys from the RMS: Trms*sqrt(deltat*bandwidth) = tsys'''
            rms =  AutoRMS(tod)
            Tsys = rms*np.sqrt(bandwidth*sample_duration)
            return Tsys

        def weighted_mean(x,e):
            ''' Calculate the weighted mean '''
            return np.sum(x/e**2)/np.sum(1./e**2)

        def weighted_var(x,e):
            ''' Calculate weighted variance '''
            m = weighted_mean(x,e)
            v = np.sum((x-m)**2/e**2)/np.sum(1./e**2)
            return v

        # Get diode-calibrated data
        cal = process.diode()


        # Function to store lists of strings to h5py by formatting them to ascii
        def h5format(stringlist):
            return [n.encode("ascii", "ignore") for n in stringlist]

        # Get list of scan numbers
        def select(data,pol): # function to pick only caloff, pol0 data
            return data[(custom.data.field('PLNUM')==pol)&(custom.data.field('CAL')=='F')]
        scans = np.unique(custom.data.field('SCAN'))
        obsmodes = [process.scaninfo(number=x, procedureonly=True) for x in scans]
        objects = [process.scaninfo(number=x, sourceonly=True) for x in scans]
        nchans = len(custom.frequencies['channelfreq'])

        # If noisestats file doesn't exist, create one with an empty dictionary with the correct sizes
        h5_filename = f'./noise/NoiseStats_Session{session}_Bank{bank}_Bin{nchannels}.h5'
        if not os.path.isfile(h5_filename):
            hf = h5py.File(h5_filename, 'w')
            hf.create_dataset('scan_number',    data=scans)
            hf.create_dataset('channelfreq',    data=custom.frequencies['channelfreq'])
            hf.create_dataset('channelflag',    data=custom.frequencies['channelflag'])
            hf.create_dataset('delta',          data=custom.frequencies['delta'])
            hf.create_dataset('extent',         data=custom.frequencies['extent'])
            hf.create_dataset('tsys',           data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('MAD',            data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('rms',            data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('alpha',          data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('fknee',          data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('whitenoise',     data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('alpha_err',      data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('fknee_err',      data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('whitenoise_err', data=np.zeros([2,len(scans),nchans]))
            hf.create_dataset('obsmode',        data=h5format(obsmodes) )
            hf.create_dataset('object',         data=h5format(objects) )

            print(f'Initialised h5py file in {h5_filename}')
            hf.close()

        # For every scan, calculate and write RFI stats
        from tqdm import tqdm
        for i, scan in enumerate(tqdm(scans)):
            for pol in [0,1]:

                # Pick the data from that polarization and scan number
                scan_number = select(custom.data.field('SCAN'),pol=pol)
                data = cal[f'{pol}'][scan_number==scan]

                # Remove all points with an exposure of less than 2/3rds the duration
                duration = select(custom.data.field('DURATION'), pol=pol)[scan_number==scan]
                exposure = select(custom.data.field('EXPOSURE'), pol=pol)[scan_number==scan]
                data[exposure<2/3*duration] = np.nan

                # Calculate stats for that scan
                Tsys = TsysRMS(data, sample_duration=np.nanmedian(exposure), bandwidth=np.abs(custom.frequencies['delta']*1e9))/np.sqrt(2)
                mad = MAD(data,axis=0)
                rms = AutoRMS(data)

                # Fit for 1/f noise properties
                alpha = np.zeros(1024)
                whitenoise = np.zeros(1024)
                fknee = np.zeros(1024)
                alpha_err = np.zeros(1024)
                whitenoise_err = np.zeros(1024)
                fknee_err = np.zeros(1024)

                if np.mod(nchans,nchannels) !=0:
                    import sys
                    print(f'ERROR: the number of binning channels, {nchannels}, is not divisible by the total number of channels, {nchans}.')
                    sys.exit(1)

                plotinfo = {'final': False,
                            'session': session,
                            'bank': bank,
                            'scan': scan,
                            'pol': pol,
                            'nchannels': nchannels,
                            'object': process.scaninfo(number=scan, sourceonly=True)}

                # Measure 1/f properties for every channel

                if nchannels == 1:
                    for channel in np.arange(nchans):
                        try:
                            # Get data for the current channel
                            datanow = data[:,channel]
                            # Remove all nan values and infinities before fitting
                            datanow = datanow[~np.isnan(datanow) & ~np.isinf(datanow)]
                            # Fit 1/f profile
                            fitparams, uncertainties = process.OneOverFFit(data=datanow[~np.isnan(datanow)], whitenoise=AutoRMS(datanow), showplot=True, plotinfo=plotinfo)
                            alpha[channel], whitenoise[channel], fknee[channel] = fitparams
                            alpha_err[channel], whitenoise_err[channel], fknee_err[channel] = uncertainties
                        except:
                            alpha[channel] = np.nan
                            whitenoise[channel] = np.nan
                            fknee[channel] = np.nan
                            alpha_err[channel] = np.nan
                            whitenoise_err[channel] = np.nan
                            fknee_err[channel] = np.nan

                else: # measure 1/f properties for a bin of channels
                    for iter in np.arange(int(np.round(nchans/nchannels))):
                        try:
                            # Get mean data for the current channel
                            datanow = np.nanmedian(data[:,iter*nchannels:((iter+1)*nchannels)],axis=1) # use a median to throw away crazy values
                            # Remove all nan values and infinities before fitting
                            datanow = datanow[~np.isnan(datanow) & ~np.isinf(datanow)]
                            # Fit 1/f profile
                            fitparams, uncertainties = process.OneOverFFit(data=datanow, whitenoise=AutoRMS(datanow), showplot=False, plotinfo=plotinfo)
                            alpha         [iter*nchannels:((iter+1)*nchannels)] = fitparams[0]
                            whitenoise    [iter*nchannels:((iter+1)*nchannels)] = fitparams[1]
                            fknee         [iter*nchannels:((iter+1)*nchannels)] = fitparams[2]
                            alpha_err     [iter*nchannels:((iter+1)*nchannels)] = uncertainties[0]
                            whitenoise_err[iter*nchannels:((iter+1)*nchannels)] = uncertainties[1]
                            fknee_err     [iter*nchannels:((iter+1)*nchannels)] = uncertainties[2]
                        except:
                            alpha[iter*nchannels:((iter+1)*nchannels)] = np.nan
                            whitenoise[iter*nchannels:((iter+1)*nchannels)] = np.nan
                            fknee[iter*nchannels:((iter+1)*nchannels)] = np.nan
                            alpha_err[iter*nchannels:((iter+1)*nchannels)] = np.nan
                            whitenoise_err[iter*nchannels:((iter+1)*nchannels)] = np.nan
                            fknee_err[iter*nchannels:((iter+1)*nchannels)] = np.nan

                # Modify file
                hf = h5py.File(h5_filename, 'r+')
                hf['tsys'][pol,i,:] = Tsys
                hf['MAD'] [pol,i,:] = mad
                hf['rms'] [pol,i,:] = rms
                hf['alpha'][pol,i,:] = alpha
                hf['fknee'][pol,i,:] = fknee
                hf['whitenoise'][pol,i,:] = whitenoise
                hf['alpha_err'][pol,i,:] = alpha_err
                hf['fknee_err'][pol,i,:] = fknee_err
                hf['whitenoise_err'][pol,i,:] = whitenoise_err
                hf.close()

        # Final message
        print(f'Saved noise stats for S{session} Bank{bank} in {h5_filename}')


        return



    def OneOverFFit(self, data, whitenoise, showplot=False, plotinfo=None):
        ''' Function to fit a 1/f power spectra model to data '''

        if plotinfo is not None:
            if plotinfo['final']==True:
                session   = plotinfo['session']
                scan      = plotinfo['scan']
                name       = plotinfo['name']
                titlestring = f'[{name}] S{session} Scan{scan}'
            elif plotinfo['final']==False:
                session   = plotinfo['session']
                bank      = plotinfo['bank']
                pol       = plotinfo['pol']
                scan      = plotinfo['scan']
                nchannels = plotinfo['nchannels']
                object    = plotinfo['object']
                titlestring = f'S{session} Scan{scan} Bank{bank} Pol{pol} Bin{nchannels}: {object}'

        def prep_fft(data, data_frequency_Hz):
            ''' Perform FFT and return power spectrum
            A constant data recording rate is assumed '''
            from numpy.fft import fft
            # Clear NaNs and infs
            data = data[~np.isnan(data)]
            data = data[~np.isinf(data)]
            # Number of sample points
            N = len(data)
            # Sample spacing in seconds
            T = 1./data_frequency_Hz
            # Create FFT axes
            yf = fft(data)
            power = 1/N * np.abs(yf[0:N//2])**2 # normalization to return a mean of rms^2
            freq = np.linspace(0.0, 1.0/(2.0*T), N//2)
            return freq[2:], power[2:] # we are not interested in the first zero frequency bin, and we should not be fitting the first point either according to Clive, so ignore the first two points

        def OneOverF(freq, alpha, whitenoise, fknee):
            ''' 1/f noise model '''
            return np.multiply( np.power(whitenoise,2) , np.add(1, np.power(np.divide(fknee, freq), alpha)))

        def Error(lm_param, freq, power, uncertainty):
            ''' LMFIT error function '''
            # Extract current parameters from lmfit object lm_param
            current_params = [x for x in lm_param.valuesdict().values()]
            # Evaluate model
            model = OneOverF(freq, *current_params) # star to unpack list
            # Convert to logspace
            model = np.log10(model)
            power = np.log10(power)
            return ((model-power)/uncertainty)**2 # reduce the difference in log space

        def Residual(r):
            ''' LMFIT residual function '''
            return np.sum(r.dot(r.T))

        def chisquared(lm_param, freq, power):
            # Evaluate model
            model = OneOverF(freq, *lm_param) # star to unpack list
            # Convert to logspace
            model = np.log10(model)
            power = np.log10(power)
            return np.sum((model-power)**2)/len(power)

        # Compute FFT
        freq, power_linear = prep_fft(data=data, data_frequency_Hz=10)
        power = np.log10(power_linear)

        # Bin in logspace and get uncertainty
        newxspace = np.logspace(np.log10(0.0001),np.log10(10), 30)
        counts, bin_edges = np.histogram(freq, bins=newxspace)
        hist, bin_edges   = np.histogram(freq, bins=newxspace, weights=power_linear)
        hist = hist/counts # normalize by number of points in each bin
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        uncertainty = 1/np.sqrt(counts)

        # uncertainty = np.ones(len(hist))
        # for i, start in enumerate(bin_edges[:-1]):
        #     end = bin_edges[i+1]
        #     indices = np.where([(freq>start) & (freq<end)])[1]
        #     powers = power[indices]
        #     if len(powers) == 0: # no data points here
        #         uncertainty[i] = np.nan
        #     elif len(powers) == 1: # only a single data point
        #         # find the next and previous data points and get a rough standard deviation
        #         if indices[0] > 0: # if not first datapoint
        #             uncertainty[i] = 1/np.sqrt(len(powers))#np.nanstd([power[indices[0]-1], power[indices[0]], power[indices[0]+1]])/np.sqrt(3)
        #         else: # if first data point
        #             uncertainty[i] = 1/np.sqrt(len(powers))#np.nanstd([power[indices[0]], power[indices[0]+1]])/np.sqrt(2)
        #     else:
        #         uncertainty[i] = 1/np.sqrt(len(powers))#np.std(powers)/np.sqrt(len(powers))


        # Remove nans because some bins will not have values
        keep_these = np.where((~np.isnan(hist)) & (~np.isnan(uncertainty)) & (uncertainty!=0))[0]
        bin_centers = bin_centers[keep_these]
        uncertainty = uncertainty[keep_these]
        hist        = hist[keep_these]

        # Fit with LMFIT
        from lmfit import minimize, Parameters, Parameter,fit_report, Minimizer
        lm_param = Parameters()
        lm_param.add('alpha', 2, vary=True, min=0.1,max=5)
        lm_param.add('whitenoise', whitenoise, vary=True, min=0)
        lm_param.add('fknee', 0.1, vary=True, min=0.005, max=3)


        fitter = Minimizer(Error, lm_param, reduce_fcn=Residual, fcn_args=(bin_centers, hist, uncertainty))
        lmfit_results = fitter.minimize(method='leastsq')

        # Extract resulting parameters
        lmfit_params = [x for x in lmfit_results.params.valuesdict().values()]
        uncertainties = np.sqrt(np.diag(lmfit_results.covar))
        fknee = lmfit_params[2]
        actualwhitenoise = lmfit_params[1]

        if self.settings['plotting']['savefigs_oneoverf']:
            xspace = np.logspace(np.log10(np.nanmin(freq)),np.log10(np.nanmax(freq)),100)
            model = OneOverF(xspace, *lmfit_params) # star unpacks list to pass as arguments
            newxspace = np.logspace(np.log10(0.0001),np.log10(5), 30)
            newmodel = OneOverF(newxspace, *lmfit_params)
            plt.figure(1, figsize=(5,3.5))
            plt.loglog(freq, power_linear, '-')
            plt.loglog(bin_centers,hist,'x', label='Bins Fitted')
            plt.loglog(xspace,[whitenoise**2] * len(xspace),'g--', label='White Noise')
            plt.loglog(xspace,[actualwhitenoise**2] * len(xspace), 'k:', alpha=0.3)
            plt.loglog([fknee,fknee],[np.nanmin(power_linear)/3,np.nanmax(power_linear)*3], 'k:', alpha=0.3)
            plt.loglog(xspace,model,'k--', label='Fitted Function')
            plt.xlim([np.nanmin(freq),np.nanmax(freq)])
            plt.ylim([np.nanmin(power_linear)/3,np.nanmax(power_linear)*3])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel(r'Power (K$^2$)')
            plt.title(f'{titlestring}\n'+r'1/f Fit: $\chi^2$=' + f'{chisquared(lmfit_params, bin_centers, hist):.2f}')
            plt.legend()

            import os
            if plotinfo['final']==True:
                if not os.path.exists(f'./figures/oneoverf/{name}'):
                    os.makedirs(f'./figures/oneoverf/{name}')
                if not os.path.exists(f'./figures/oneoverf/{name}/pdf'):
                    os.makedirs(f'./figures/oneoverf/{name}/pdf')
                plt.savefig(f'./figures/oneoverf/{name}/session{session}_scan{scan}_fit.png',dpi=300)
                plt.savefig(f'./figures/oneoverf/{name}/pdf/session{session}_scan{scan}_fit.pdf')
                print(f"Saved ./figures/oneoverf/{name}/session{session}_scan{scan}_fit.png")

            if plotinfo['final']==False:
                if not os.path.exists(f'./figures/oneoverf/Bin{nchannels}'):
                    os.makedirs(f'./figures/oneoverf/Bin{nchannels}')
                if not os.path.exists(f'./figures/oneoverf/Bin{nchannels}/pdf'):
                    os.makedirs(f'./figures/oneoverf/Bin{nchannels}/pdf')
                plt.savefig(f'./figures/oneoverf/Bin{nchannels}/session{session}_scan{scan}_bank{bank}_pol{pol}_{object}_fit.png',dpi=300)
                plt.savefig(f'./figures/oneoverf/Bin{nchannels}/pdf/session{session}_scan{scan}_bank{bank}_pol{pol}_{object}_fit.pdf')
                print(f"Saved ./figures/oneoverf/Bin{nchannels}/session{session}_scan{scan}_bank{bank}_pol{pol}_{object}_fit.png")

            if showplot:
                plt.show()

            plt.close('all')

        return lmfit_params, uncertainties



    def AtmosphericFit(self, elevation, data, showplot=False, return_subtracted=False):
        ''' Function to fit atmospheric brightness temperature vs elevation '''

        def Atmosphere(elevation, A, C):
            ''' 1/f noise model '''
            return np.divide( A, np.sin(np.deg2rad(elevation)) ) + C

        def Error(lm_param, elevation, data):
            ''' LMFIT error function '''
            # Extract current parameters from lmfit object lm_param
            current_params = [x for x in lm_param.valuesdict().values()]
            # Evaluate model
            model = Atmosphere(elevation, *current_params) # star to unpack list
            return (model-data)**2 # reduce the difference in log space

        def Residual(r):
            ''' LMFIT residual function '''
            return np.sum(r.dot(r.T))

        def chisquared(lm_param, freq, power):
            # Evaluate model
            model = OneOverF(elevation, *lm_param) # star to unpack list
            return np.sum((model-power)**2)/(len(power)-2)

        # Remove nans because some bins will not have values
        keep_these = np.where((~np.isnan(data)) & (~np.isnan(elevation)) & (data!=0))[0]
        data      = data[keep_these]
        elevation = elevation[keep_these]

        # Fit with LMFIT
        from lmfit import minimize, Parameters, Parameter,fit_report, Minimizer
        lm_param = Parameters()
        lm_param.add('A', 1, vary=True)
        lm_param.add('C', 0, vary=True)

        fitter = Minimizer(Error, lm_param, reduce_fcn=Residual, fcn_args=(elevation, data))
        lmfit_results = fitter.minimize(method='leastsq')

        # Extract resulting parameters
        lmfit_params = [x for x in lmfit_results.params.valuesdict().values()]
        uncertainties = np.sqrt(np.diag(lmfit_results.covar))
        A = lmfit_params[0]
        C = lmfit_params[1]

        # Return subtracted data
        datamodel = Atmosphere(elevation, *lmfit_params)
        data_subtracted = np.subtract(data, datamodel)

        if return_subtracted:
            return data_subtracted
        else:
            return lmfit_params, uncertainties




    def flagdata(self):
        ''' Takes calibrated data and replaces flagged bits with np.nans '''

        # Get diode-calibrated data
        cal = self.diode()

        # Function to select data
        def select(data, pol):
            return data[(self.data.field('PLNUM')==pol)&(self.data.field('CAL')=='F')]

        # For both polarisations
        for pol in [0,1]:

            # Flag bandwidthcut channels
            cal[f'{pol}'][:,self.frequencies['channelflag']!=0] = np.nan
            # Apply the spectral RFI flag
            cal[f'{pol}'][:,self.frequencies['spectralRFIflag'][f'{pol}']!=0] = np.nan
            # Apply the general flag
            if self.deadpolflag[f'{pol}']!=0:
                cal[f'{pol}'][:,:] = np.nan

            # Apply the noise statistics flags
            scan_numbers = select(self.data.field('SCAN'), pol)
            for p, scan in enumerate(self.noiseflags['scan_number']):
                # Get current data matrix, get 1d array for flagging
                noisestats_currentscan = self.noiseflags[f'{pol}'][p,:]
                calibrdata_currentscan = cal[f'{pol}'][scan_numbers==scan,:]

                # Take the data corresponding to the current scan and flag all bad channels
                calibrdata_currentscan[:, noisestats_currentscan!=0] = np.nan

                # Copy back to main data matrix
                cal[f'{pol}'][scan_numbers==scan,:] = calibrdata_currentscan

        return cal



    def todmaker(self, name, session, median_filter_length=120, create_destriping_dataset=True):
        ''' Create single frequency TODs from good data '''
        # TODO: in the future the central frequency should change as a function of time,
        # once we get into noise statistics flagging over time

        from config import settings
        np.seterr(divide='ignore')


        # If the session is Ku, separate feeds 1 and 2
        if session in [1,2,3]:
            feeds = [1,2]
        elif session in [4]:
            feeds = [1]
        else:
            import sys
            print(f'ERROR: session {session} not found.')
            sys.exit(1)


        # Change the settings to the current session
        settings['data']['session'] = session


        for feed in feeds:

            print(f'Producing TODs for session {session}, feed {feed}...')

            # Assign the banks corresponding to the current feed
            if feed == 1:
                available_banks = {'1': ['B', 'C', 'D'],
                                   '2': ['A', 'B', 'C', 'D'],
                                   '3': ['A', 'B', 'C', 'D'],
                                   '4': ['B', 'C', 'D'] }
            if feed == 2:
                available_banks = {'1': ['F', 'G', 'H'],
                                   '2': ['E', 'F', 'G', 'H'],
                                   '3': ['E', 'F', 'G', 'H'],
                                   '4': [ ] }

            # Function to select data
            def select(data, pol):
                return data[(custom.data.field('PLNUM')==pol)&(custom.data.field('CAL')=='F')]

            # Open every bank in the session to create a mega-dataset that includes them all
            def getshape(session):
                ''' Get shape full array '''
                settings['data']['bank'] = 'C'
                custom = Processing(settings)
                cal = custom.flagdata()
                return [np.shape(cal['0'])[0], np.shape(cal['0'])[1], 2, len(available_banks[f'{session}'])]

            # Initialise mega dataset
            megashape = getshape(session)
            cal_allbanks = np.empty((megashape))
            cal_allbanks[:] = np.nan
            wei_allbanks = np.zeros((megashape))
            fre_allbanks = np.empty((megashape))
            fre_allbanks[:] = np.nan

            # Read in ALL the banks
            from tqdm import tqdm
            print(f'Reading session {session}...')
            for i, bank in enumerate(tqdm(available_banks[f'{session}'])):

                # Get diode-calibrated and flagged data
                settings['data']['bank'] = bank
                custom = Processing(settings)
                cal = custom.flagdata()

                # Get Tsys
                NoiseStats = custom.read_noisestats(session=custom.settings['data']['session'], bank=custom.settings['data']['bank'], nchannels=custom.settings['flagging']['dataset_nbins'])
                Tsys = NoiseStats['tsys']

                # Copy the data to a matrix of frequency
                for pol in [0,1]:

                    # Inisialise weights array
                    wei = np.zeros(np.shape(cal[f'{pol}']))
                    fre = np.zeros(np.shape(cal[f'{pol}']))

                    # For every scan, populate wei array
                    scan_numbers = select(custom.data.field('SCAN'), pol)
                    for p, scan in enumerate(custom.noiseflags['scan_number']):
                        # Get current data matrix, get 1d array for flagging
                        wei_currentscan = 1/np.power(Tsys[pol,p,:],2)
                        fre_currentscan = custom.frequencies['channelfreq']

                        # Add it to wei array
                        wei[scan_numbers==scan,:] = list([wei_currentscan,]*np.shape(wei[scan_numbers==scan,:])[0])
                        fre[scan_numbers==scan,:] = list([fre_currentscan,]*np.shape(fre[scan_numbers==scan,:])[0])


                    # Set all nan or inf weights to zero
                    wei[(np.isnan(wei)) | (np.isinf(wei))] = 0

                    # Make sure that the weights are zero whenever the data is a nan
                    wei[(np.isnan(cal[f'{pol}'])) | (np.isinf(cal[f'{pol}'])) | (cal[f'{pol}']==0)] = 0

                    # Copy data and weights to megaarray
                    cal_allbanks[:,:,pol,i] = cal[f'{pol}']
                    wei_allbanks[:,:,pol,i] = wei
                    fre_allbanks[:,:,pol,i] = fre

            print('Done!')

            # Reshape array so y is frequency and x is data
            print(f"Averaging bands {available_banks[f'{session}']} in session {session}...")
            cal_allbanks = np.reshape(cal_allbanks, (megashape[0],megashape[1]*2*megashape[3]), order='F')
            wei_allbanks = np.reshape(wei_allbanks, (megashape[0],megashape[1]*2*megashape[3]), order='F')
            fre_allbanks = np.reshape(fre_allbanks, (megashape[0],megashape[1]*2*megashape[3]), order='F')


            # Convert all nans to zeros and make sure that the corresponding weight is zero too!
            wei_allbanks[(np.isnan(cal_allbanks)) | (np.isinf(cal_allbanks))] = 0 # important - now nans will not be counted
            fre_allbanks[(np.isnan(cal_allbanks)) | (np.isinf(cal_allbanks))] = 0 # patch frequencies
            cal_allbanks[(np.isnan(cal_allbanks)) | (np.isinf(cal_allbanks))] = 0 # finally, and only now, patch data

            # Take an average of all the frequencies weighted by the squared reciprocal of the system temperature
            # TODO: try using the amplitude of correlated noise as a weight
            tod          = np.sum(np.multiply(cal_allbanks, wei_allbanks),axis=1) / np.sum(wei_allbanks, axis=1)
            average_freq = np.sum(np.multiply(fre_allbanks, wei_allbanks),axis=1) / np.sum(wei_allbanks, axis=1)
            # Kish's Effective Sample Size
            eff_bandwidth = np.multiply(  np.power(np.sum(wei_allbanks,axis=1),2) / np.sum(np.power(wei_allbanks,2), axis=1) , np.abs(custom.frequencies['delta']))/2 # effective bandwidth in GHz, the division by 2 accounts for the two polarisations

            # Set any infinities (from all weights being zero in an certain moment) to nan
            tod[(tod==0) | (np.isinf(tod))] = np.nan
            average_freq[(average_freq==0) | (np.isinf(average_freq))] = np.nan
            eff_bandwidth[(eff_bandwidth==0) | (np.isinf(eff_bandwidth))] = np.nan

            # Create median filtered TOD scan by scan - NEED HARRIER FOR THIS
            from comancpipeline.Tools.median_filter.medfilt import medfilt
            median_filtered_tod = np.zeros(np.shape(tod))

            # Function to pick only caloff, pol0 data
            def select(data):
                return data[(custom.data.field('PLNUM')==0)&(custom.data.field('CAL')=='F')]

            # Function to be able to save lists of strings to h5py by formatting them to ascii
            def h5format(stringlist):
                return [n.encode("ascii", "ignore") for n in stringlist]

            # print(np.shape(tod),np.shape(scan_numbers))
            # print(np.nanmin(scan_numbers),np.nanmax(scan_numbers))
            # print(np.nanmin(custom.noiseflags['scan_number']),np.nanmax(custom.noiseflags['scan_number']))
            for scan in custom.noiseflags['scan_number']:

                # First attempt to subtract atmospheric model if the band is Ku
                #if elevation_corrections_ku:
                if session in [99,98]:#[1,2,3]: # if we are dealing with Ku, i turned it off since not yet functional
                    elevation_to_fit = select(custom.data.field('ELEVATIO'))
                    bt_to_fit        = tod[scan_numbers==scan]
                    tod_atmosphere_subtracted = custom.AtmosphericFit(data=bt_to_fit, elevation=elevation_to_fit, return_subtracted=True)
                    tod[scan_numbers==scan] = tod_atmosphere_subtracted



                # Get chunk of tod belonging to this scan, median filter and append
                if len(tod[scan_numbers==scan])>median_filter_length/2:
                    median_filtered_current_tod = tod[scan_numbers==scan] - medfilt(tod[scan_numbers==scan]*1, median_filter_length) # the *1 copies the array into a different bit in memory
                else:
                    median_filtered_current_tod = tod[scan_numbers==scan] - np.nanmean(tod[scan_numbers==scan])
                    print(f'Scan {scan} only has {len(tod[scan_numbers==scan])} samples, which is less than {median_filter_length/2}, half the median filter length.')
                # Append this bit
                median_filtered_tod[scan_numbers==scan] = median_filtered_current_tod




            # Write it all to an h5py file
            import h5py
            h5_filename = f'./tods/TOD_Session{session}_feed{feed}_{name}_RAW.h5'
            print(f'Writing TOD to {h5_filename}...', end=' ')
            hf = h5py.File(h5_filename, 'w')
            hf.create_dataset('tod',                 data=median_filtered_tod                              )
            hf.create_dataset('tod_raw',             data=tod                                              )
            hf.create_dataset('average_frequency',   data=average_freq                                     )
            hf.create_dataset('effective_bandwidth', data=eff_bandwidth                                     )
            hf.create_dataset('scan_number',         data=select(custom.data.field('SCAN'))                )
            hf.create_dataset('ra',                  data=select(custom.data.field('CRVAL2'))              )
            hf.create_dataset('dec',                 data=select(custom.data.field('CRVAL3'))              )
            hf.create_dataset('object',              data=h5format(select(custom.data.field('OBJECT')))    )
            hf.create_dataset('exposure',            data=select(custom.data.field('EXPOSURE'))            )
            hf.create_dataset('lst',                 data=select(custom.data.field('LST'))                 )
            hf.create_dataset('elevation',           data=select(custom.data.field('ELEVATIO'))            )
            hf.create_dataset('huminidy',            data=select(custom.data.field('HUMIDITY'))            )
            hf.create_dataset('pressure',            data=select(custom.data.field('PRESSURE'))            )
            hf.create_dataset('tambient',            data=select(custom.data.field('TAMBIENT'))            )
            hf.create_dataset('obsmode',             data=h5format(select(custom.data.field('OBSMODE')))   )
            print('Done!')
            hf.close()

            if create_destriping_dataset:
                custom.tod_weights(name=name, session=session, object='daisy_center', feed=feed)

        return


    def tod_weights(self, name, session, feed, object='daisy_center', bad_scan_ranges=None, weight_by_whitenoise=False):
        ''' Creates a noise properties column for every daisy scan, and creates a daisy-only TOD '''

        # Create custom object with the correct session
        from config import settings
        settings['data']['session'] = session
        custom = IO(settings)

        # Get full TOD
        TOD = self.read_tod(name, session, object=object, feed=feed, bad_scan_ranges=bad_scan_ranges)
        tsys = np.zeros(np.shape(TOD['tod']))
        rms = np.zeros(np.shape(TOD['tod']))
        alpha = np.zeros(np.shape(TOD['tod']))
        fknee = np.zeros(np.shape(TOD['tod']))
        whitenoise = np.zeros(np.shape(TOD['tod']))
        aoof = np.zeros(np.shape(TOD['tod']))
        average_frequency = TOD['average_frequency']
        effective_bandwidth = TOD['effective_bandwidth']
        exposure = TOD['exposure']

        from tqdm import tqdm
        print('Calculating the final TOD noise properties...')
        for scan in tqdm(np.unique(TOD['scan_number'])):
            # Current TOD
            current_tod = TOD['tod'][TOD['scan_number']==scan]
            current_effective_bandwidth = TOD['effective_bandwidth'][TOD['scan_number']==scan]

            # Get rms and tsys for the final distribution
            def AutoRMS(tod):
                ''' Auto-differenced RMS '''
                if len(tod.shape) == 2:
                    N = (tod.shape[0]//2)*2
                    diff = tod[1:N:2,:] - tod[:N:2,:]
                    rms = np.nanstd(diff,axis=0)/np.sqrt(2)
                else:
                    N = (tod.size//2)*2
                    diff = tod[1:N:2] - tod[:N:2]
                    rms = np.nanstd(diff)/np.sqrt(2)
                return rms

            def TsysRMS(tod,sample_duration,bandwidth):
                ''' Calculate Tsys from the RMS: Trms*sqrt(deltat*bandwidth) = tsys'''
                rms =  AutoRMS(tod)
                Tsys = rms*np.sqrt(bandwidth*sample_duration)
                return Tsys

            rms[TOD['scan_number']==scan] = [AutoRMS(tod=current_tod)] * len(rms[TOD['scan_number']==scan])
            tsys[TOD['scan_number']==scan] = TsysRMS(tod=current_tod,sample_duration=np.nanmedian(exposure),bandwidth=current_effective_bandwidth*1e9)

            plotinfo = {'final': True,
                        'session': session,
                        'name': name,
                        'scan': scan}


            # Get 1/f properties
            process = Processing(custom.settings)
            try:
                fitparams, uncertainties = process.OneOverFFit(data=current_tod, whitenoise=AutoRMS(tod=current_tod), showplot=False, plotinfo=plotinfo)
                alpha[TOD['scan_number']==scan] = [fitparams[0]] * len(rms[TOD['scan_number']==scan])
                whitenoise[TOD['scan_number']==scan] = [fitparams[1]] * len(rms[TOD['scan_number']==scan])
                fknee[TOD['scan_number']==scan] = [fitparams[2]] * len(rms[TOD['scan_number']==scan])
            except:
                alpha[TOD['scan_number']==scan] = [0] * len(rms[TOD['scan_number']==scan])
                whitenoise[TOD['scan_number']==scan] = [0] * len(rms[TOD['scan_number']==scan])
                fknee[TOD['scan_number']==scan] = [0] * len(rms[TOD['scan_number']==scan])


        # Get amplitude of 1/f at certain timelength, and take square root so that it is in units K
        def OneOverF(freq, alpha, whitenoise, fknee):
            ''' 1/f noise model '''
            return np.multiply( np.power(whitenoise,2) , np.add(1, np.power(np.divide(fknee, freq), alpha)))
        aoof = np.sqrt(OneOverF(freq=1./custom.settings['flagging']['OneOverF_timesc'], alpha=alpha, whitenoise=whitenoise, fknee=fknee))


        # Get weights
        if weight_by_whitenoise:
            destriping_weights = 1/np.power(rms,2)
        else: # weight using the 1/f amplitude
            destriping_weights = 1/np.power(aoof,2)

        # Zero weights where alpha is zero
        destriping_weights[alpha==0] = 0

        # Apply 0 weights to points that are not within the aoof, tsys, fknee, alpha criteria in the configuration
        destriping_weights[effective_bandwidth<custom.settings['flagging']['min_bandwidth'][custom.settings['data']['band']]] = 0 # minimum bandwidth
        destriping_weights[(tsys<custom.settings['flagging']['final_tsys_minmax'][custom.settings['data']['band']][0])  | (tsys>custom.settings['flagging']['final_tsys_minmax'][custom.settings['data']['band']][1]) ] = 0
        destriping_weights[(fknee<custom.settings['flagging']['final_fknee_minmax'][custom.settings['data']['band']][0]) | (fknee>custom.settings['flagging']['final_fknee_minmax'][custom.settings['data']['band']][1])] = 0
        destriping_weights[(alpha<custom.settings['flagging']['final_alpha_minmax'][custom.settings['data']['band']][0]) | (alpha>custom.settings['flagging']['final_alpha_minmax'][custom.settings['data']['band']][1])] = 0
        destriping_weights[(aoof<custom.settings['flagging']['final_OneOverF_minmax'][custom.settings['data']['band']][0])  | (aoof>custom.settings['flagging']['final_OneOverF_minmax'][custom.settings['data']['band']][1]) ] = 0

        # Initialise noise statistics objects to pass to the plotting function
        tsys_object      = {'all':  tsys,
                            'good': tsys[destriping_weights!=0]}
        fknee_object     = {'all': fknee,
                            'good': fknee[destriping_weights!=0]}
        alpha_object     = {'all':  alpha,
                            'good': alpha[destriping_weights!=0]}
        oneoverf_object  = {'all':  aoof,
                            'good': aoof[destriping_weights!=0]}
        bandwidth_object = {'all':  effective_bandwidth,
                            'good': effective_bandwidth[destriping_weights!=0]}
        avefreq_object   = {'all':  average_frequency,
                            'good': average_frequency[destriping_weights!=0]}
        rms_object       = {'all':  rms,
                            'good': rms[destriping_weights!=0]}


        # Plot TODcuts and save figures to ./figures/TODcuts with the appropiate name
        from config import settings
        from plotting import Plotting
        plotting = Plotting(settings)
        plotting.TOD_distribution(tsys=tsys_object, fknee=fknee_object, alpha=alpha_object, oneoverf=oneoverf_object, bandwidth=bandwidth_object, avefreq=avefreq_object, rms=rms_object, session=session, feed=feed)

        # Function to be able to save lists of strings to h5py by formatting them to ascii
        def h5format(stringlist):
            return [n.encode("ascii", "ignore") for n in stringlist]

        # Write it all to an h5py file
        import h5py
        print('\nWriting all scans...', end='    ')
        h5_filename = f'./tods/TOD_Session{session}_feed{feed}_{name}_FULL_DES.h5'
        print(f'Writing TOD to {h5_filename}...', end=' ')
        hf = h5py.File(h5_filename, 'w')
        hf.create_dataset('tod',                data=TOD['tod']                 )
        hf.create_dataset('average_frequency',  data=TOD['average_frequency']   )
        hf.create_dataset('scan_number',        data=TOD['scan_number']         )
        hf.create_dataset('ra',                 data=TOD['ra']                  )
        hf.create_dataset('dec',                data=TOD['dec']                 )
        hf.create_dataset('object',             data=h5format(TOD['object'])    )
        hf.create_dataset('exposure',           data=TOD['exposure']            )
        hf.create_dataset('lst',                data=TOD['lst']                 )
        hf.create_dataset('elevation',          data=TOD['elevation']           )
        hf.create_dataset('destriping_weights', data=destriping_weights         )
        hf.create_dataset('tsys',               data=tsys                        )
        hf.create_dataset('rms',                data=rms                        )
        hf.create_dataset('alpha',              data=alpha                      )
        hf.create_dataset('whitenoise',         data=whitenoise                 )
        hf.create_dataset('fknee',              data=fknee                      )
        hf.create_dataset('aoof',               data=aoof                       )
        hf.create_dataset('huminidy',           data=TOD['huminidy']            )
        hf.create_dataset('pressure',           data=TOD['pressure']            )
        hf.create_dataset('tambient',           data=TOD['tambient']            )
        hf.create_dataset('obsmode',            data=h5format(TOD['obsmode'])   )
        print('Done!')
        hf.close()


        # Now write individual files for each scan
        print('\nWriting scans one by one...', end='    ')
        filelist = open(f'./tods/TOD_Session{session}_feed{feed}_{name}_filelist.txt', 'w')


        for scan in tqdm(np.unique(TOD['scan_number'])):
            # Write it all to an h5py file
            h5_filename = f'./tods/TOD_Session{session}_feed{feed}_{name}_scan{scan}_DES.h5'
            filelist.write(h5_filename+'\n')
            hf = h5py.File(h5_filename, 'w')
            hf.create_dataset('tod',                data=TOD['tod'][TOD['scan_number']==scan]                 )
            hf.create_dataset('average_frequency',  data=TOD['average_frequency'][TOD['scan_number']==scan]   )
            hf.create_dataset('scan_number',        data=TOD['scan_number'][TOD['scan_number']==scan]         )
            hf.create_dataset('ra',                 data=TOD['ra'][TOD['scan_number']==scan]                  )
            hf.create_dataset('dec',                data=TOD['dec'][TOD['scan_number']==scan]                 )
            hf.create_dataset('object',             data=h5format([x for i,x in enumerate(TOD['object']) if TOD['scan_number'][i]==scan])    )
            hf.create_dataset('exposure',           data=TOD['exposure'][TOD['scan_number']==scan]            )
            hf.create_dataset('lst',                data=TOD['lst'][TOD['scan_number']==scan]                 )
            hf.create_dataset('elevation',          data=TOD['elevation'][TOD['scan_number']==scan]           )
            hf.create_dataset('destriping_weights', data=destriping_weights[TOD['scan_number']==scan]         )
            hf.create_dataset('tsys',               data=tsys[TOD['scan_number']==scan]                       )
            hf.create_dataset('rms',                data=rms[TOD['scan_number']==scan]                        )
            hf.create_dataset('alpha',              data=alpha[TOD['scan_number']==scan]                      )
            hf.create_dataset('whitenoise',         data=whitenoise[TOD['scan_number']==scan]                 )
            hf.create_dataset('fknee',              data=fknee[TOD['scan_number']==scan]                      )
            hf.create_dataset('aoof',               data=aoof[TOD['scan_number']==scan]                       )
            hf.create_dataset('huminidy',           data=TOD['huminidy'][TOD['scan_number']==scan]            )
            hf.create_dataset('pressure',           data=TOD['pressure'][TOD['scan_number']==scan]            )
            hf.create_dataset('tambient',           data=TOD['tambient'][TOD['scan_number']==scan]            )
            hf.create_dataset('obsmode',            data=h5format([x for i,x in enumerate(TOD['obsmode']) if TOD['scan_number'][i]==scan])   )
            hf.close()
        print('Done!')
        filelist.close()
        return


    def scaninfo(self, number, sourceonly=False, procedureonly=False):
        ''' Returns the source and procedure associated with a particular scan number '''
        procedure = np.unique(self.data.field('OBSMODE')[self.data.field('SCAN')==number])[0]
        source    = np.unique(self.data.field('OBJECT') [self.data.field('SCAN')==number])[0]
        if sourceonly:
            return source
        if procedureonly:
            return procedure
        else:
            return source, procedure
