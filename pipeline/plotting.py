from IO import IO

import matplotlib.pyplot as plt
import numpy as np

class Plotting(IO):

    def __init__(self, settings):
        ''' Initialise object, pass independent instance of IO class '''
        # super().__init__(settings) # copies all the IO attibutes to self, but this doesn't work when inheriting multiple classes
        IO.__init__(self, settings)   # use this when inheriting multiple parent classes
        return


    def noisestats(self, session, bank, nchannels, object=None):
        ''' Visualizes the Tsys and MAD for a given
        bank in a given session. Timescale is the time
        in seconds for which to calculate the 1/f amplitude'''

        fig_dimensions = [7,10]

        # Copy timescale
        timescale = self.settings['flagging']['OneOverF_timesc']

        # Open the data
        NoiseStats = self.read_noisestats(session, bank, nchannels)

        # Get amplitude of 1/f at certain timelength, later only plot daisy ones
        # Take square root so that it is in units K
        def OneOverF(freq, alpha, whitenoise, fknee):
            ''' 1/f noise model '''
            return np.multiply( np.power(whitenoise,2) , np.add(1, np.power(np.divide(fknee, freq), alpha)))
        AmplitudeOneOverF = np.sqrt(OneOverF(freq=1./timescale, alpha=NoiseStats['alpha'], whitenoise=NoiseStats['whitenoise'], fknee=NoiseStats['fknee']))

        # Only plot selected object
        if object is not None:
            objectindexes = [i for i, item in enumerate(NoiseStats['object']) if item!=f'{object}']
            AmplitudeOneOverF       [:, objectindexes, :] = np.nan
            NoiseStats['fknee']     [:, objectindexes, :] = np.nan
            NoiseStats['alpha']     [:, objectindexes, :] = np.nan
            NoiseStats['whitenoise'][:, objectindexes, :] = np.nan
            NoiseStats['tsys']      [:, objectindexes, :] = np.nan

        def aspect(extent, height, width):
            x_extent = np.abs(extent[1]-extent[0])
            y_extent = np.abs(extent[3]-extent[2])
            raw_aspect = y_extent/x_extent
            desired_aspect = height/width
            imshow_aspect = desired_aspect/raw_aspect/2
            return imshow_aspect

        from matplotlib import gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        extent = [np.nanmin(NoiseStats['scan_number'])-0.5, np.nanmax(NoiseStats['scan_number'])+0.5, NoiseStats['extent'][1], NoiseStats['extent'][0]]
        imshow_aspect = aspect(extent, fig_dimensions[1], fig_dimensions[0])

        maximum_tsys = {'1': 500,
                        '2': 500,
                        '3': 500,
                        '4': 300}
        maximum_aoof = {'1': 1, # maximum one over f amplitude when scaling
                        '2': 1,
                        '3': 1,
                        '4': 2}
        maximum_whno = {'1': 1.5, # maximum white noise when scaling
                        '2': 1.5,
                        '3': 1.5,
                        '4': 0.8}

        for pol in [0,1]:

            # Plot Fit Parameters

            fig, axs = plt.subplots(nrows=3, figsize=(fig_dimensions[0],fig_dimensions[1]))

            ax0 = axs[0]
            ax1 = axs[1]
            ax2 = axs[2]

            im0 = ax0.imshow(NoiseStats['alpha'][pol,:,:].transpose(), vmin=0.5, vmax=5, aspect=imshow_aspect, extent=extent)
            ax0.set_ylabel(r'$\nu_{c}$'+' (GHz)')
            ax0.set_title(r'$\alpha$' )

            im1 = ax1.imshow(NoiseStats['fknee'][pol,:,:].transpose(), vmin=0, vmax=0.5, aspect=imshow_aspect, extent=extent)
            ax1.set_ylabel(r'$\nu_{c}$'+' (GHz)')
            ax1.set_title(r'$f_{\rm knee}$'+' (Hz)' )

            im2 = ax2.imshow(NoiseStats['whitenoise'][pol,:,:].transpose(), vmin=0, vmax=maximum_whno[f'{session}'], aspect=imshow_aspect, extent=extent)
            ax1.set_ylabel(r'$\nu_{c}$'+' (GHz)')
            ax2.set_title(r'$\sigma_{w}$'+' (K)')
            ax2.set_xlabel('Scan Number')

            fig.colorbar(im0, ax=ax0,pad=0.03)
            fig.colorbar(im1, ax=ax1,pad=0.03)
            fig.colorbar(im2, ax=ax2,pad=0.03)
            plt.text(0.54, 0.93, f'Session {session}, Bank {bank}, Pol {pol}, Bins {nchannels}', fontsize=14, ha='center',  transform=plt.gcf().transFigure)

            if self.settings['plotting']['savefigs']:
                import os
                if not os.path.exists(f'./figures/noise/Bin{nchannels}'):
                    os.makedirs(f'./figures/noise/Bin{nchannels}')
                if not os.path.exists(f'./figures/noise/Bin{nchannels}/pdf'):
                    os.makedirs(f'./figures/noise/Bin{nchannels}/pdf')
                plt.savefig(f'./figures/noise/Bin{nchannels}/session{session}_bank{bank}_pol{pol}_bin{nchannels}_fitparams.png')
                plt.savefig(f'./figures/noise/Bin{nchannels}/pdf/session{session}_bank{bank}_pol{pol}_bin{nchannels}_fitparams.pdf')
                print(f"Saved ./figures/noise/Bin{nchannels}/session{session}_bank{bank}_pol{pol}_bin{nchannels}_fitparams.png")

            if self.settings['plotting']['showfigs']:
                plt.show()
            else:
                plt.close(1)


            # Plot white and correlated noise amplitudes

            fig, axs = plt.subplots(nrows=2, figsize=(fig_dimensions[0],fig_dimensions[1]))

            ax0 = axs[0]
            ax1 = axs[1]

            im0 = ax0.imshow(NoiseStats['tsys'][pol,:,:].transpose(), vmin=0, vmax=maximum_tsys[f'{session}'], aspect=imshow_aspect, extent=extent)
            ax0.set_ylabel(r'$\nu_{c}$'+' (GHz)')
            ax0.set_title(r'$T_{\rm sys}$' + ' (K)')

            im1 = ax1.imshow(AmplitudeOneOverF[pol,:,:].transpose(), vmin=0, vmax=maximum_aoof[f'{session}'], aspect=imshow_aspect, extent=extent)
            ax1.set_ylabel(r'$\nu_{c}$'+' (GHz)')
            ax1.set_title(r'$A_{\rm 1/f}$'+f', {timescale} s'+' (K)' )

            fig.colorbar(im0, ax=ax0,pad=0.03)
            fig.colorbar(im1, ax=ax1,pad=0.03)
            fig.colorbar(im2, ax=ax2,pad=0.03)
            plt.text(0.45, 0.93, f'Session {session}, Bank {bank}, Pol {pol}, Bins {nchannels}', fontsize=14, ha='center',  transform=plt.gcf().transFigure)

            if self.settings['plotting']['savefigs']:
                plt.savefig(f'./figures/noise/Bin{nchannels}/session{session}_bank{bank}_pol{pol}_bin{nchannels}_noise.png')
                plt.savefig(f'./figures/noise/Bin{nchannels}/pdf/session{session}_bank{bank}_pol{pol}_bin{nchannels}_noise.pdf')
                print(f"Saved ./figures/noise/Bin{nchannels}/session{session}_bank{bank}_pol{pol}_bin{nchannels}_noise.png")

            if self.settings['plotting']['showfigs']:
                plt.show()
            else:
                plt.close(1)

        return



    def noisestat_distribution(self, band, nchannels=None, object=None):
        ''' Plots Tsys, 1/f amplitude, alpha, knee frequency and white noise distributions
        for a specific band, either C or Ku. On top of these it plots the current cuts '''

        from tqdm import tqdm

        if nchannels is None:
            nchannels = self.settings['flagging']['dataset_nbins']

        print(f'Getting statistics from {nchannels} channel noisestats...')

        # Config
        nbins = 100
        limits = {'Tsys':       [0,800],
                  'Aoof':       [0,2],
                  'alpha':      [0.1,4.5],
                  'fknee':      [0.01,0.5],
                  'whitenoise': [0,1]}

        # Ordering
        available_banks = {'1': ['B', 'C', 'D', 'F', 'G', 'H'],
                           '2': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                           '3': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                           '4': ['B', 'C', 'D']}
        if band=='C':
            sessions = [4]
        elif band=='Ku':
            sessions = [1,2,3]

        # Initialise array
        all_Tsys        = []
        all_alpha       = []
        all_fknee       = []
        all_Aoof        = []
        all_whitenoise  = []

        def select(data, pol):
            return data[(custom.data.field('PLNUM')==pol)&(custom.data.field('CAL')=='F')]

        # Open each and every noise stat file
        for session in sessions:
            print(f'Reading session {session}...')
            for i,bank in enumerate(tqdm(available_banks[f'{session}'])):

                # Get diode-calibrated and flagged data
                from config import settings
                from processing import Processing
                settings['data']['bank'] = bank
                settings['data']['session'] = session
                custom = Processing(settings)
                cal = custom.flagdata()

                # Get noisestats
                NoiseStats = self.read_noisestats(session=session, bank=bank, nchannels=nchannels)


                def OneOverF(freq, alpha, whitenoise, fknee):
                    ''' 1/f noise model '''
                    return np.multiply( np.power(whitenoise,2) , np.add(1, np.power(np.divide(fknee, freq), alpha)))
                timescale = custom.settings['flagging']['OneOverF_timesc'] # 60 seconds since we are dominated by 1/f noise there
                AmplitudeOneOverF = np.sqrt(OneOverF(freq=1./timescale, alpha=NoiseStats['alpha'], whitenoise=NoiseStats['whitenoise'], fknee=NoiseStats['fknee']))

                # Only get selected object
                if object is not None:
                    objectindexes = [i for i, item in enumerate(NoiseStats['object']) if item!=f'{object}']
                    AmplitudeOneOverF       [:, objectindexes, :] = np.nan
                    NoiseStats['fknee']     [:, objectindexes, :] = np.nan
                    NoiseStats['alpha']     [:, objectindexes, :] = np.nan
                    NoiseStats['whitenoise'][:, objectindexes, :] = np.nan
                    NoiseStats['tsys']      [:, objectindexes, :] = np.nan

                freq_mean = np.mean(NoiseStats['channelfreq'])
                Tsys_now  = np.copy(NoiseStats['tsys'].flatten()[~np.isnan(NoiseStats['tsys'].flatten())])
                alpha_now = np.copy(NoiseStats['alpha'].flatten()[~np.isnan(NoiseStats['alpha'].flatten())])
                fknee_now = np.copy(NoiseStats['fknee'].flatten()[~np.isnan(NoiseStats['fknee'].flatten())])
                whitenoise_now = np.copy(NoiseStats['whitenoise'].flatten()[~np.isnan(NoiseStats['whitenoise'].flatten())])
                aoof_now = np.copy(AmplitudeOneOverF.flatten()[~np.isnan(AmplitudeOneOverF.flatten())])


                # Copy only data that makes it through
                for pol in [0,1]:
                    scan_numbers = select(custom.data.field('SCAN'), pol)
                    for p, scan in enumerate(custom.noiseflags['scan_number']):
                        noisestats_currentscan = custom.noiseflags[f'{pol}'][p,:]
                        calibrdata_currentscan = cal[f'{pol}'][scan_numbers==scan,:]
                        NoiseStats['tsys'][pol, NoiseStats['scan_number']==scan, noisestats_currentscan!=0] = np.nan
                        NoiseStats['fknee'][pol, NoiseStats['scan_number']==scan, noisestats_currentscan!=0] = np.nan
                        NoiseStats['alpha'][pol, NoiseStats['scan_number']==scan, noisestats_currentscan!=0] = np.nan
                        NoiseStats['whitenoise'][pol, NoiseStats['scan_number']==scan, noisestats_currentscan!=0] = np.nan
                        AmplitudeOneOverF[pol, NoiseStats['scan_number']==scan, noisestats_currentscan!=0] = np.nan

                # Get only non flagged data
                Tsys_good  = np.copy(NoiseStats['tsys'].flatten()[~np.isnan(NoiseStats['tsys'].flatten())])
                alpha_good = np.copy(NoiseStats['alpha'].flatten()[~np.isnan(NoiseStats['alpha'].flatten())])
                fknee_good = np.copy(NoiseStats['fknee'].flatten()[~np.isnan(NoiseStats['fknee'].flatten())])
                whitenoise_good = np.copy(NoiseStats['whitenoise'].flatten()[~np.isnan(NoiseStats['whitenoise'].flatten())])
                aoof_good = np.copy(AmplitudeOneOverF.flatten()[~np.isnan(AmplitudeOneOverF.flatten())])

                # Flatten and copy all non-nans
                all_Tsys.append(Tsys_now)
                all_alpha.append(alpha_now)
                all_fknee.append(fknee_now)
                all_whitenoise.append(whitenoise_now)
                all_Aoof.append(aoof_now)

                # Plot histograms
                import os
                if not os.path.exists(f'./figures/datacuts/pdf'):
                    os.makedirs(f'./figures/datacuts/pdf')

                plt.figure(1) # Tsys
                plt.hist(Tsys_now, bins=np.linspace(limits['Tsys'][0],limits['Tsys'][1],nbins), histtype='step', color=f'C{i}')
                plt.hist(Tsys_good, bins=np.linspace(limits['Tsys'][0],limits['Tsys'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Bank {bank} ({freq_mean:.1f} GHz)', color=f'C{i}')
                plt.xlabel(r'$T_{\rm{sys}}$ (K)')
                plt.xlim(limits['Tsys'])
                plt.title(f'{band}-band Data Cuts: '+r'$T_{\rm{sys}}$'+ f' (S{session})')
                plt.legend()
                if self.settings['plotting']['savefigs']:
                    plt.savefig(f'./figures/datacuts/BIN{nchannels}_{band}_session{session}_Tsys.png')
                    plt.savefig(f'./figures/datacuts/pdf/BIN{nchannels}_{band}_session{session}_Tsys.pdf')

                plt.figure(2) # Aoof
                plt.hist(aoof_now, bins=np.linspace(limits['Aoof'][0],limits['Aoof'][1],nbins), histtype='step', color=f'C{i}')
                plt.hist(aoof_good, bins=np.linspace(limits['Aoof'][0],limits['Aoof'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Bank {bank} ({freq_mean:.1f} GHz)', color=f'C{i}')
                plt.xlabel(r'$A_{\rm{1/f}}$ (K)')
                plt.xlim(limits['Aoof'])
                plt.title(f'{band}-band Data Cuts: '+r'$A_{\rm{1/f}}$ (K)'+ f' (S{session})')
                plt.legend()
                if self.settings['plotting']['savefigs']:
                    plt.savefig(f'./figures/datacuts/BIN{nchannels}_{band}_session{session}_Aoof.png')
                    plt.savefig(f'./figures/datacuts/pdf/BIN{nchannels}_{band}_session{session}_Aoof.pdf')

                plt.figure(3) # alpha
                plt.hist(alpha_now, bins=np.linspace(limits['alpha'][0],limits['alpha'][1],nbins), histtype='step', color=f'C{i}')
                plt.hist(alpha_good, bins=np.linspace(limits['alpha'][0],limits['alpha'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Bank {bank} ({freq_mean:.1f} GHz)', color=f'C{i}')
                plt.xlabel(r'$\alpha$')
                plt.xlim(limits['alpha'])
                plt.title(f'{band}-band Data Cuts: '+r'$\alpha$ (K)'+ f' (S{session})')
                plt.legend()
                if self.settings['plotting']['savefigs']:
                    plt.savefig(f'./figures/datacuts/BIN{nchannels}_{band}_session{session}_alpha.png')
                    plt.savefig(f'./figures/datacuts/pdf/BIN{nchannels}_{band}_session{session}_alpha.pdf')

                plt.figure(4) # fknee
                plt.hist(fknee_now, bins=np.linspace(limits['fknee'][0],limits['fknee'][1],nbins), histtype='step', color=f'C{i}')
                plt.hist(fknee_good, bins=np.linspace(limits['fknee'][0],limits['fknee'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Bank {bank} ({freq_mean:.1f} GHz)', color=f'C{i}')
                plt.xlabel(r'$f_{\rm{knee}}$' + ' (Hz)')
                plt.xlim(limits['fknee'])
                plt.title(f'{band}-band Data Cuts: '+r'$f_{\rm{knee}}$'+ f' (S{session})')
                plt.legend()
                if self.settings['plotting']['savefigs']:
                    plt.savefig(f'./figures/datacuts/BIN{nchannels}_{band}_session{session}_fknee.png')
                    plt.savefig(f'./figures/datacuts/pdf/BIN{nchannels}_{band}_session{session}_fknee.pdf')

                plt.figure(5) # whitenoise
                plt.hist(whitenoise_now, bins=np.linspace(limits['whitenoise'][0],limits['whitenoise'][1],nbins), histtype='step', color=f'C{i}')
                plt.hist(whitenoise_good, bins=np.linspace(limits['whitenoise'][0],limits['whitenoise'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Bank {bank} ({freq_mean:.1f} GHz)', color=f'C{i}')
                plt.xlabel(r'$\sigma_{\rm{w}}$' + ' (K)')
                plt.xlim(limits['whitenoise'])
                plt.title(f'{band}-band Data Cuts: '+r'$\sigma_{\rm{w}}$'+ f' (S{session})')
                plt.legend()
                if self.settings['plotting']['savefigs']:
                    plt.savefig(f'./figures/datacuts/BIN{nchannels}_{band}_session{session}_whitenoise.png')
                    plt.savefig(f'./figures/datacuts/pdf/BIN{nchannels}_{band}_session{session}_whitenoise.pdf')

            if self.settings['plotting']['showfigs']:
                plt.show()

            plt.close('all')

        return




    def TOD_distribution(self, tsys, fknee, alpha, oneoverf, bandwidth, avefreq, rms, session, feed):
        ''' Plots  the final noise properties of the TOD for a specific session and
        feed: Tsys, 1/f amplitude, alpha, knee frequency and effective bandwidth. '''

        # Config
        nbins = 30
        limits = {'Tsys':       [0,2500],
                  'Aoof':       [0,2],
                  'alpha':      [0.1,4.5],
                  'fknee':      [0.01,0.5],
                  'bandwidth':  [0,4],
                  'avefreq':    [4,16],
                  'rms':        [0,1]}

        if session in [4]:
            band = 'C'
        elif session in [1,2,3]:
            band = 'Ku'
        else:
            band = None


        # Plot histograms
        import os
        if not os.path.exists(f'./figures/TODcuts/pdf'):
            os.makedirs(f'./figures/TODcuts/pdf')

        plt.figure(1) # Tsys
        plt.hist(tsys['all'], bins=np.linspace(limits['Tsys'][0],limits['Tsys'][1],nbins), histtype='step', color='C1')
        plt.hist(tsys['good'], bins=np.linspace(limits['Tsys'][0],limits['Tsys'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Feed {feed}', color='C1')
        plt.xlabel(r'$T_{\rm{sys}}$ (K)')
        plt.xlim(limits['Tsys'])
        plt.title(f'{band}-band TOD Cuts: '+r'$T_{\rm{sys}}$'+ f' (S{session}, Feed{feed})')
        plt.legend()
        if self.settings['plotting']['savefigs']:
            plt.savefig(f'./figures/TODcuts/session{session}_feed{feed}_Tsys.png')
            plt.savefig(f'./figures/TODcuts/pdf/session{session}_feed{feed}_Tsys.pdf')

        plt.figure(2) # Aoof
        plt.hist(oneoverf['all'], bins=np.linspace(limits['Aoof'][0],limits['Aoof'][1],nbins), histtype='step', color='C1')
        plt.hist(oneoverf['good'], bins=np.linspace(limits['Aoof'][0],limits['Aoof'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Feed {feed}', color='C1')
        plt.xlabel(r'$A_{\rm{1/f}}$ (K)')
        plt.xlim(limits['Aoof'])
        plt.title(f'{band}-band TOD Cuts: '+r'$A_{\rm{1/f}}$ (K)'+ f' (S{session}, Feed{feed})')
        plt.legend()
        if self.settings['plotting']['savefigs']:
            plt.savefig(f'./figures/TODcuts/session{session}_feed{feed}_Aoof.png')
            plt.savefig(f'./figures/TODcuts/pdf/session{session}_feed{feed}_Aoof.pdf')

        plt.figure(3) # alpha
        plt.hist(alpha['all'], bins=np.linspace(limits['alpha'][0],limits['alpha'][1],nbins), histtype='step', color='C1')
        plt.hist(alpha['good'], bins=np.linspace(limits['alpha'][0],limits['alpha'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Feed {feed}', color='C1')
        plt.xlabel(r'$\alpha$')
        plt.xlim(limits['alpha'])
        plt.title(f'{band}-band TOD Cuts: '+r'$\alpha$ (K)'+ f' (S{session}, Feed{feed})')
        plt.legend()
        if self.settings['plotting']['savefigs']:
            plt.savefig(f'./figures/TODcuts/session{session}_feed{feed}_alpha.png')
            plt.savefig(f'./figures/TODcuts/pdf/session{session}_feed{feed}_alpha.pdf')

        plt.figure(4) # fknee
        plt.hist(fknee['all'], bins=np.linspace(limits['fknee'][0],limits['fknee'][1],nbins), histtype='step', color='C1')
        plt.hist(fknee['good'], bins=np.linspace(limits['fknee'][0],limits['fknee'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Feed {feed}', color='C1')
        plt.xlabel(r'$f_{\rm{knee}}$' + ' (Hz)')
        plt.xlim(limits['fknee'])
        plt.title(f'{band}-band TOD Cuts: '+r'$f_{\rm{knee}}$'+ f' (S{session}, Feed{feed})')
        plt.legend()
        if self.settings['plotting']['savefigs']:
            plt.savefig(f'./figures/TODcuts/session{session}_feed{feed}_fknee.png')
            plt.savefig(f'./figures/TODcuts/pdf/session{session}_feed{feed}_fknee.pdf')

        plt.figure(5) # bandwidth
        plt.hist(bandwidth['all'], bins=np.linspace(limits['bandwidth'][0],limits['bandwidth'][1],nbins), histtype='step', color='C1')
        plt.hist(bandwidth['good'], bins=np.linspace(limits['bandwidth'][0],limits['bandwidth'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Feed {feed}', color='C1')
        plt.xlabel(r'$\Delta\nu_{\rm{eff}}$'+ ' (GHz)')
        plt.xlim(limits['bandwidth'])
        plt.title(f'{band}-band TOD Cuts: '+r'$\Delta\nu_{\rm{eff}}$'+ f' (S{session}, Feed{feed})')
        plt.legend()
        if self.settings['plotting']['savefigs']:
            plt.savefig(f'./figures/TODcuts/session{session}_feed{feed}_bandwidth.png')
            plt.savefig(f'./figures/TODcuts/pdf/session{session}_feed{feed}_bandwidth.pdf')

        plt.figure(6) # average frequency
        plt.hist(avefreq['all'], bins=np.linspace(limits['avefreq'][0],limits['avefreq'][1], 96), histtype='step', color='C1')
        plt.hist(avefreq['good'], bins=np.linspace(limits['avefreq'][0],limits['avefreq'][1],96), histtype='stepfilled', alpha=0.3, label=f'S{session}, Feed {feed}', color='C1')
        plt.xlabel(r'$\nu_{\rm{eff}}$'+ ' (GHz)')
        plt.xlim(limits['avefreq'])
        plt.title(f'{band}-band TOD Cuts: '+r'$\nu_{\rm{eff}}$'+ f' (S{session}, Feed{feed})')
        plt.legend()
        if self.settings['plotting']['savefigs']:
            plt.savefig(f'./figures/TODcuts/session{session}_feed{feed}_averagefrequency.png')
            plt.savefig(f'./figures/TODcuts/pdf/session{session}_feed{feed}_averagefrequency.pdf')

        plt.figure(6) # average frequency
        plt.hist(rms['all'], bins=np.linspace(limits['rms'][0],limits['rms'][1], nbins), histtype='step', color='C1')
        plt.hist(rms['good'], bins=np.linspace(limits['rms'][0],limits['rms'][1],nbins), histtype='stepfilled', alpha=0.3, label=f'S{session}, Feed {feed}', color='C1')
        plt.xlabel('RMS'+ ' (K)')
        plt.xlim(limits['rms'])
        plt.title(f'{band}-band TOD Cuts: '+'RMS'+ f' (S{session}, Feed{feed})')
        plt.legend()
        if self.settings['plotting']['savefigs']:
            plt.savefig(f'./figures/TODcuts/session{session}_feed{feed}_rms.png')
            plt.savefig(f'./figures/TODcuts/pdf/session{session}_feed{feed}_rms.pdf')

        if self.settings['plotting']['showfigs']:
            plt.show()

        plt.close('all')

        return




    def spectralRFI(self, session, bank):
        ''' Visualizes the kurtosis and coefficient of variation for a given
        bank in a given session '''

        fig_dimensions = [7,10]

        # Open the data
        SpectralRFI = self.read_spectralRFI(session, bank)

        def aspect(extent, height, width):
            x_extent = np.abs(extent[1]-extent[0])
            y_extent = np.abs(extent[3]-extent[2])
            raw_aspect = y_extent/x_extent
            desired_aspect = height/width
            imshow_aspect = desired_aspect/raw_aspect/2
            return imshow_aspect

        from matplotlib import gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        extent = [np.nanmin(SpectralRFI['scan_number'])-0.5, np.nanmax(SpectralRFI['scan_number'])+0.5, SpectralRFI['extent'][1], SpectralRFI['extent'][0]]
        imshow_aspect = aspect(extent, fig_dimensions[1], fig_dimensions[0])

        for pol in [0,1]:

            fig, axs = plt.subplots(nrows=2, figsize=(fig_dimensions[0],fig_dimensions[1]))

            ax0 = axs[0]
            ax1 = axs[1]
            ax0.set_ylabel(r'$\nu_{c}$'+' (GHz)')
            ax0.set_title('Kurtosis' )
            im0 = ax0.imshow(SpectralRFI['kurtosis'][pol,:,:].transpose(), vmin=0, vmax=np.nanmedian(SpectralRFI['kurtosis'][pol,:,:])*5, aspect=imshow_aspect, extent=extent)

            im1 = ax1.imshow(SpectralRFI['variation_coefficient'][pol,:,:].transpose(), vmin=0, vmax=np.nanmedian(SpectralRFI['variation_coefficient'][pol,:,:])*7.5, aspect=imshow_aspect, extent=extent)
            ax1.set_xlabel('Scan Number')
            ax1.set_ylabel(r'$\nu_{c}$'+' (GHz)')
            ax1.set_title('Coefficient of Variation' )

            fig.colorbar(im0, ax=ax0,pad=0.03)
            fig.colorbar(im1, ax=ax1,pad=0.03)
            plt.text(0.45, 0.93, f'Session {session}, Bank {bank}, Pol {pol}', fontsize=14, ha='center',  transform=plt.gcf().transFigure)

            if self.settings['plotting']['savefigs']:
                import os
                if not os.path.exists('./figures/rfi/pdf'):
                    os.makedirs('./figures/rfi/pdf')
                plt.savefig(f'./figures/rfi/session{session}_bank{bank}_pol{pol}.png')
                plt.savefig(f'./figures/rfi/pdf/session{session}_bank{bank}_pol{pol}.pdf')
                print(f"Saved ./figures/rfi/session{session}_bank{bank}_pol{pol}.png")

            if self.settings['plotting']['showfigs']:
                plt.show()
            else:
                plt.close(1)

        return





    def calibrated_data(self, settings):
        ''' Plot an image of calibrated data for all scans of a given session, all banks '''

        # Config
        fig_dimensions = [8,4]

        # List of banks to plot
        available_banks = {'1': ['B', 'C', 'D', 'F', 'G', 'H'],
                           '2': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                           '3': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                           '4': ['B', 'C', 'D']}

        # For every bank, show the calibrated data
        banks_to_plot = available_banks[f"{self.settings['data']['session']}"]

        # Calculate an aspect ratio that meets the figure standards
        def aspect(extent, subplot_height, subplot_width, banks_to_plot):
            x_extent = duration
            y_extent = np.nanmax(custom.frequencies['extent']) - np.nanmin(custom.frequencies['extent'])
            raw_aspect = y_extent/x_extent
            desired_aspect = subplot_height/subplot_width
            imshow_aspect = desired_aspect/raw_aspect/len(banks_to_plot)
            return imshow_aspect

        # For every scan
        from tqdm import tqdm
        from matplotlib import gridspec
        for scan in tqdm(np.unique(self.data.field('SCAN'))):

            # For both polarisations
            for pol in [0,1]:

                # Start figure
                subplot_height = fig_dimensions[1]*len(banks_to_plot)
                subplot_width = fig_dimensions[0]
                plt.figure(1, constrained_layout=True, figsize=(subplot_width,subplot_height))
                gs = gridspec.GridSpec(len(banks_to_plot),2, width_ratios=[1,0.03])
                gs.update(wspace=0.07, hspace=0)

                # Iterate through the chosen banks
                for i, bank in enumerate(banks_to_plot):

                    # Initialise custom object
                    settings['data']['bank'] = bank
                    custom = IO(settings)

                    # Get calibrated data
                    from processing import Processing
                    cal = Processing.diode(custom)
                    scan_caldata = cal[f'{pol}'][cal['scan']==scan, :]
                    duration = np.shape(scan_caldata)[0]*0.1/60 # scan duration in minutes

                    # Plot figure
                    ax = plt.subplot(gs[i,0])
                    if i == 0:
                        # Get scan info
                        source, procedure = Processing.scaninfo(custom, scan)
                        names = {'OnOff:PSWITCHOFF:TPWCAL': 'Off',
                                 'OnOff:PSWITCHON:TPWCAL' : 'On',
                                 'RALongMap:NONE:TPWCAL'  : 'Mapping',
                                 'Nod:NONE:TPWCAL'        : 'Nod',
                                 'Unknown': 'Unknown'}
                        procedure = names[procedure]
                        ax.set_title(f"Session {custom.settings['data']['session']}, Scan {scan}, Pol {pol}: {source} {procedure}" )
                    extent = [0, duration, custom.frequencies['extent'][1], custom.frequencies['extent'][0]]
                    im = ax.imshow(scan_caldata.T, vmin=20, vmax=50, extent=extent, origin='upper', aspect=aspect(extent, subplot_height, subplot_width, banks_to_plot))
                    ax.set_ylabel(fr'Bank {bank}: $\nu_c$ (GHz)')
                    if i == len(banks_to_plot)-1:
                        ax.set_xlabel('Time (min)')
                    else:
                        ax.set_xticks([])

                ax = plt.subplot(gs[:,1])
                cb = plt.colorbar(im, ax)
                cb.ax.set_title(r'$T_{\rm sys}$' + ' (K)')


                if self.settings['plotting']['savefigs']:
                    import os
                    if not os.path.exists('./figures/caldata/pdf'):
                        os.makedirs('./figures/caldata/pdf')
                    plt.savefig(f"./figures/caldata/session{self.settings['data']['session']}_scan{scan}_pol{pol}.png")
                    plt.savefig(f"./figures/caldata/pdf/session{self.settings['data']['session']}_scan{scan}_pol{pol}.pdf")
                    print(f"Saved ./figures/caldata/session{self.settings['data']['session']}_scan{scan}_pol{pol}.png")

                if self.settings['plotting']['showfigs']:
                    plt.show()
                else:
                    plt.close(1)

        return




    def obstype(self):
        ''' Plots the schedule of a session '''

        # Choose colours
        colours = {'daisy_center': 'b',
                   '3C147':        'r',
                   '0530+1331':    'g'}

        # Print object and procedure type in each scan
        numbers, procedures, objects = [], [], []
        for scan in np.unique(self.data.field('SCAN')):
            numbers.append(scan)
            procedures.append(np.unique(self.data.field('OBSMODE')[self.data.field('SCAN')==scan])[0])
            objects.append(np.unique(self.data.field('OBJECT') [self.data.field('SCAN')==scan])[0])
        numbers = np.array(numbers)

        # Define plotting styles: custom name then alpha

        if self.settings['data']['band'] == 'C':
            style = {'OnOff:PSWITCHOFF:TPWCAL': ['Off', 0.3],
                     'OnOff:PSWITCHON:TPWCAL' : ['On',  0.7],
                     'RALongMap:NONE:TPWCAL'  : ['Mapping', 0.3],
                     'Unknown': ['Unknown', 0.1]}

        elif self.settings['data']['band'] == 'Ku':
            style = {'Nod:NONE:TPWCAL'      : ['Nod', 0.3],
                     'RALongMap:NONE:TPWCAL': ['Mapping', 0.3]}

        # Unpack plotting settings
        alphas     = [style[w][1] for w in procedures]
        for key in style.keys(): # Replace every instance
            procedures = [w.replace(key, style[key][0]) for w in procedures]

        # Plot scans
        fig, ax = plt.subplots(figsize=(3.2*1.3, 8*1.3*5))
        plt.title(f"Session {self.settings['data']['session']} Observation Types")
        plt.xlim([0,1])
        plt.ylim([0, np.nanmax(numbers)+1])
        plt.ylabel('Scan Number')

        for start, end, object, procedure, alpha in zip(numbers, numbers+1, objects, procedures, alphas):
            if procedure !='Unknown':
                ax.axhspan(start, end, 0, 1,  alpha=alpha, color=colours[object], lw=0, zorder=0, label=f'{object}: {procedure}')

        for position in np.arange(np.nanmax(numbers)+1):
            plt.plot([0,1], [position,position],'k-', alpha=0.5, linewidth=0.5)

        plt.xticks([])
        allticks = np.arange(np.nanmax(numbers)+1)
        plt.yticks(allticks+0.5, allticks)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        if self.settings['plotting']['savefigs']:
            plt.savefig(f"./figures/obstype_session{self.settings['data']['session']}.png")
            plt.savefig(f"./figures/obstype_session{self.settings['data']['session']}.pdf")

        if self.settings['plotting']['showfigs']:
            plt.show()


        return




    def schedule(self):
        ''' Plots the schedule of a session '''

        # Calculate sidereal time of object
        from astropy.coordinates import SkyCoord, EarthLocation, AltAz
        from astropy.time import Time
        from astropy import units as u
        from coordinates import Coordinates


        # Create date range
        def datetime_linspace(start, end):
            ''' Create a range of dates separated by one minute '''
            import pandas as pd
            start = start[:-5]+'00.00' # floor start to the nearest minute
            start = pd.Timestamp(start) # timestamps are in nanoseconds
            end = pd.Timestamp(end)
            nminutes = np.floor((end.value-start.value)/(60*1e9))+1 # number of one minute steps
            dates = np.arange(nminutes)*60*1e9 + start.value
            dates = pd.to_datetime(dates)
            return dates

        # Get date range
        dates = datetime_linspace(start=self.data.field('DATE-OBS')[0], end=self.data.field('DATE-OBS')[-1])

        # Calculate LST
        def LST_wrapper(gbt_lon, dates):
            ''' Calls the LST function in the right format '''
            LST = []
            for date in dates:
                YY = str(int(str(date).split('-')[0])-2000)
                MM = str(date).split('-')[1]
                DD = str(date).split('-')[2].split(' ')[0]
                hh = str(date).split(' ')[1].split(':')[0]
                mm = str(date).split(' ')[1].split(':')[1]
                LST.append(Coordinates.lst_calculator(lon=gbt_lon, utc_time=f'{MM}{DD}{YY} {hh}{mm}'))
            return LST

        gbt_lon = -79.83983
        LST = LST_wrapper(gbt_lon, dates)

        # Calculate elevation of objects of interest (target, 3C147 and 0531+...)
        def elevation_calculator(dates, source_name):
            ''' Get elevation of the source at a given date as seen from the GBT '''
            # Calibrator Coordinates
            catalogue_galactic = {'daisy_center': [192.41, -11.51],
                                  '3C147':        [161.68636686748, +10.29833172063],
                                  '0530+1331':    [191.36766147, -11.01194840]}
            calibrator_coords = catalogue_galactic[source_name]
            # Set location
            observing_location = EarthLocation(lat=38.43312*u.deg, lon=-79.83983*u.deg, height=8.245950E+02*u.m)
            # Get elevation of object
            coords = SkyCoord(l=calibrator_coords[0]*u.degree,b=calibrator_coords[1]*u.degree, frame='galactic')
            # Get elevation
            times = Time(dates, scale='utc')
            coordsaltaz = coords.transform_to(AltAz(obstime=times,location=observing_location))
            elevation = coordsaltaz.alt.degree
            return elevation

        # Plot elevations in a loop
        fig, ax = plt.subplots(figsize=(8,2.7))#(figsize=(8*1.3,3.2*1.3))
        plt.title(f"Session {self.settings['data']['session']} Schedule")
        plt.ylim([10,90])
        plt.xlim([np.nanmin(LST), np.nanmax(LST)])
        plt.xlabel('LST (h)')
        plt.ylabel('Elevation (deg)')
        colours = {'daisy_center': 'b',
                   '3C147':        'r',
                   '0530+1331':    'g'}

        for source in ['daisy_center', '3C147', '0530+1331']:

            # Find the boundaries of each observation on the source
            def find_boundaries(source):
                instances = np.where(self.data.field('OBJECT')==source)[0]
                boundaries_start = instances[np.where(np.r_[True, instances[1:] > instances[:-1]+1])[0]]  # find changes
                # Now that we know where new names start, find the previous instance and mark it as an end
                boundaries_end = instances[np.where(np.r_[True, instances[1:] > instances[:-1]+1])[0]-1]
                # Sort the arrays
                boundaries_start = np.sort(boundaries_start)
                boundaries_end = np.sort(boundaries_end)
                # Add the first and last instances here
                boundaries_start = np.insert(boundaries_start, 0, instances[0]) # also concatenate the first instance
                boundaries_end = np.insert(boundaries_end, -1, instances[-1]) # also concatenate the last instance
                # Remove any repeated items
                boundaries_start = np.unique(boundaries_start)
                boundaries_end = np.unique(boundaries_end)
                return boundaries_start, boundaries_end

            boundaries_start, boundaries_end = find_boundaries(source)

            # Find the LSTs of those boundaries
            def LST_wrapper2(gbt_lon, dates):
                ''' Calls the LST function in the right format '''
                LST = []
                for date in dates:
                    YY = str(int(str(date).split('-')[0])-2000)
                    MM = str(date).split('-')[1]
                    DD = str(date).split('-')[2].split('T')[0]
                    hh = str(date).split('T')[1].split(':')[0]
                    mm = str(date).split('T')[1].split(':')[1]
                    LST.append(Coordinates.lst_calculator(lon=gbt_lon, utc_time=f'{MM}{DD}{YY} {hh}{mm}'))
                return LST

            boundary_LSTs_start = LST_wrapper2(gbt_lon, self.data.field('DATE-OBS')[boundaries_start])
            boundary_LSTs_end = LST_wrapper2(gbt_lon, self.data.field('DATE-OBS')[boundaries_end])

            # Plot transparent boxes
            for start, end in zip(boundary_LSTs_start, boundary_LSTs_end):
                ax.axvspan(start, end, -90, 90, alpha=0.3, color=colours[source], lw=0, zorder=0)

            # Plot elevation profile
            elevation = elevation_calculator(dates, source_name=source)
            plt.plot(LST, elevation, colours[source], linewidth=1.5, label=source, zorder=1)

        plt.legend(loc='upper right')

        if self.settings['plotting']['savefigs']:
            plt.savefig(f"./figures/schedule_session{self.settings['data']['session']}.png")
            plt.savefig(f"./figures/schedule_session{self.settings['data']['session']}.pdf")

        if self.settings['plotting']['showfigs']:
            plt.show()

        return









    def scan(self):
        ''' Plot a single scan '''
        for scan in [self.settings['data']['scan']]:
            scan_data = self.data.field('DATA')[ np.where(self.data.field('SCAN') == scan)[0] ]
            cal_data  = self.data.field('CAL') [ np.where(self.data.field('SCAN') == scan)[0] ]

            frequency_mean_data = np.nanmean(scan_data,axis=1)
            frequency_mean_data = frequency_mean_data.reshape(int(len(frequency_mean_data)/4),4)
            pol_mean_data = np.nanmean(frequency_mean_data,axis=1)

            time = np.arange(len(pol_mean_data))*0.1/60
            plt.plot(time, pol_mean_data)
            plt.title(f"Scan number {scan}")
            plt.xlabel(f'Time (min)')
            plt.show()




    def plot_data_simple(x_axis_name, y_axis_name, xlabel, ylabel, title, savename=None, show=False):
        ''' Function to plot a column of data against another '''

        plt.plot(self.data.field(f'{x_axis_name}'), self.data.field(f'{y_axis_name}'),',')
        plt.title(title + f' [{session_full_name}]')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if savename is not None:
            plt.savefig(savename+'.png')
            plt.savefig(savename+'.pdf')

        if show:
            plt.show()
        return



    def calculate_cdelt(band, daisy_radius, nyquist):
        ''' Returns Nyquist pixel size in armin '''

        # Pixelization: nyquist is pixels per FWHM beamwidt
        x_extent = daisy_radius*2 # deg
        y_extent = daisy_radius*2 # deg

        # Define write modes
        if self.settings['data']['band'] == 'C':
            beam_FWHM_arcmin = 2.54
        elif self.settings['data']['band'] == 'Ku':
            beam_FWHM_arcmin = 0.90

        # Get Nyquist spacing
        nxpix = np.round(x_extent/(beam_FWHM_arcmin/60.)*nyquist) # nyquist is the number of pix/beam
        nypix = np.round(y_extent/(beam_FWHM_arcmin/60.)*nyquist) # nyquist is the number of pix/beam
        cdelt = beam_FWHM_arcmin/nyquist # Nyquist pixel size in arcmin

        return cdelt, nxpix



    def plot_heatmap(array1, array2, label1, label2, res=50, log10=False, title=None, savename=None, xlim=None, ylim=None):
        ''' Plots a heatmap of two variables '''

        import matplotlib.pyplot as plt

        # Set resolution
        res = np.round(res)
        res = np.int_(res)

        def discard_common_nans(array1, array2):
            array1nonans = array1[(~np.isnan(array1) & ~np.isnan(array2))]
            array2nonans = array2[(~np.isnan(array1) & ~np.isnan(array2))]
            return array1nonans, array2nonans

        def discard_outside_limit(array1, array2):
            array1withinlimit = array1[(array1>xlim[0]) & (array1<xlim[1]) & (array2>ylim[0]) & (array2<ylim[1])]
            array2withinlimit = array2[(array1>xlim[0]) & (array1<xlim[1]) & (array2>ylim[0]) & (array2<ylim[1])]
            return array1withinlimit, array2withinlimit

        # Discard NaNs and discard anything outside the x and y limits
        if xlim is not None and ylim is not None:
            array1, array2 = discard_outside_limit(array1, array2)
        array1nonans, array2nonans = discard_common_nans(array1, array2)

        heatmap, xedges, yedges = np.histogram2d(array1nonans, array2nonans, bins=res)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


        if log10:
            plt.imshow(np.log10(heatmap.T), extent=extent, origin='lower', zorder = 4)
        else:
            plt.imshow(heatmap.T, extent=extent, origin='lower', zorder = 4)

        plt.xlabel(label1)
        plt.ylabel(label2)
        if title is not None:
            plt.title(title+' [log10]')
        plt.colorbar()

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if savename is not None:
            plt.savefig(f'{savename}', dpi=600, bbox_inches='tight', pad_inches=0)

        plt.axis('square')

        plt.show()
        plt.close()
        return



    def plot_hits(l_Ku, b_Ku, l_C, b_C, pix_per_beam):
        ''' Make a hit map '''

        # Basic parameters
        central_coords = [192.41, -11.51]
        rad_Ku = 0.4*1.1 # to set x and y limits
        rad_C = 1.5*1.1 # to set x and y limits

        # Calculate optimum gridding
        cdelt_C, npix_C = calculate_cdelt(band='C', daisy_radius=1.5, nyquist=pix_per_beam)
        cdelt_Ku, npix_Ku = calculate_cdelt(band='Ku', daisy_radius=0.4, nyquist=pix_per_beam)
        print(f'Using {cdelt_C:.2f} arcmin pixels for C ({npix_C:.0f} pix), and {cdelt_Ku:.2f} arcmin pixels for Ku ({npix_Ku:.0f} pix).')

        # Plot the heatmaps
        plot_heatmap(array1=l_C, array2=b_C, label1='Galactic Longitude (deg)', label2='Galactic Latitude (deg)', title=f'C-Band Hits: {pix_per_beam} pix/beam', res=npix_C, log10=True,     savename=f'./figures/cband_hits_{pix_per_beam}.png', xlim=[central_coords[0]-rad_C, central_coords[0]+rad_C], ylim=[central_coords[1]-rad_C, central_coords[1]+rad_C])
        plot_heatmap(array1=l_Ku, array2=b_Ku, label1='Galactic Longitude (deg)', label2='Galactic Latitude (deg)', title=f'Ku-Band Hits: {pix_per_beam} pix/beam', res=npix_Ku, log10=True, savename=f'./figures/kuband_hits_{pix_per_beam}.png', xlim=[central_coords[0]-rad_Ku, central_coords[0]+rad_Ku], ylim=[central_coords[1]-rad_Ku, central_coords[1]+rad_Ku])

        return
