import numpy as np

def elevation(settings):
    from IO import IO
    import matplotlib.pyplot as plt
    GBT = IO(settings)
    for feed in [1,2]:
        for session in [1,2,3]:
            plt.figure(1,figsize=(7,4))
            data = GBT.read_tod(name='firstKu', session=session, object='daisy_center', feed=feed)
            print(data['scan_number'])
            plt.title(f'Ku Feed {feed}')
            plt.plot(data['elevation'], data['tod_raw'],',', label=f'Session {session}')
            plt.legend()
            plt.xlabel('Elevation (deg)')
            plt.ylabel('Brightness Temperature (K)')
            plt.xlim([15,70])
        plt.savefig(f'./figures/elevation/Ku_feed{feed}.png')
        plt.savefig(f'./figures/elevation/Ku_feed{feed}.pdf')
        plt.show()
        plt.close()
    return


def elevationfit(settings):
    from IO import IO
    from processing import Processing
    import matplotlib.pyplot as plt
    GBT = IO(settings)
    process = Processing(settings)

    for feed in [1,2]:
        for scan in [130]:
            for session in [3]:
                plt.figure(1,figsize=(7,4))
                data = GBT.read_tod(name='firstKu', session=session, object='daisy_center', feed=feed)
                plt.title(f'Ku Feed {feed}')
                el = data['elevation'][data['scan_number']==scan]
                bt = data['tod_raw'][data['scan_number']==scan]

                # def Atmospherenoise(elevation, A, C, noise):
                #     ''' 1/f noise model '''
                #     return np.divide( A, np.sin(np.deg2rad(elevation)) ) + C + np.random.normal(0,noise, size=np.size(elevation))
                #
                # sim = Atmospherenoise(el, A=7, C=18.52413119747375, noise=0.1)
                #
                # bt = sim
                plt.plot(el, bt, '.', label=f'Session {session}', alpha=0.5)
                lmfit_params, uncertainties = process.AtmosphericFit(data=bt, elevation=el)

                def Atmosphere(elevation, A, C):
                    ''' 1/f noise model '''
                    return np.divide( A, np.sin(np.deg2rad(elevation)) ) + C


                xspace = np.linspace(np.nanmin(el), np.nanmax(el), 200)
                model = Atmosphere(xspace, *lmfit_params)
                plt.plot(xspace, model, 'k--')

                datamodel = Atmosphere(el, *lmfit_params)
                print(lmfit_params)

                plt.plot(el, bt-datamodel+np.nanmean(bt), '.', label='Subtracted')

                lmfit_params, uncertainties = process.AtmosphericFit(data=bt-datamodel, elevation=el)
                print(lmfit_params)


                plt.legend()
                plt.xlabel('Elevation (deg)')
                plt.ylabel('Brightness Temperature (K)')
            # plt.savefig(f'./figures/elevation/Ku_feed{feed}.png')
            # plt.savefig(f'./figures/elevation/Ku_feed{feed}.pdf')
            plt.show()
            plt.close()
    return


def update_all_spectralRFI(settings, band=None):
    available_banks = {'1': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       '2': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       '3': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       '4': ['A', 'B', 'C', 'D']}
    from IO import IO
    sessions_dict = {'C':  [4],
                     'Ku': [1,2,3]}
    if band is not None:
        sessions = sessions_dict[f'{band}']
    else:
        sessions = [4,3,2,1]
    for session in sessions:
        for bank in available_banks[f'{session}']:
            settings['data']['session'] = session
            settings['data']['bank'] = bank
            gbt = IO(settings)
            print(f'Session {session} Bank {bank} below')
            gbt.update_spectralRFI(session,bank)
    return


def update_all_noisestats(settings, nchannels, band=None):

    available_banks = {'1': ['B', 'C', 'D', 'F', 'G', 'H'],
                       '2': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       '3': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       '4': ['B', 'C', 'D']}

    from processing import Processing
    sessions_dict = {'C':  [4],
                     'Ku': [1,2,3]}
    if band is not None:
        sessions = sessions_dict[f'{band}']
    else:
        sessions = [4,3,2,1]

    for session in sessions:
        for bank in available_banks[f'{session}']:
            settings['data']['session'] = session
            settings['data']['bank'] = bank
            settings['flagging']['dataset_nbins'] = nchannels
            gbt = Processing(settings)
            print(f'Session {session} Bank {bank} below')
            gbt.noisestats(session,bank,nchannels)
    return


def plot_all_spectralRFI(settings):
    from tqdm import tqdm
    available_banks = {'1': ['B', 'C', 'D', 'F', 'G', 'H'],
                       '2': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       '3': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       '4': ['B', 'C', 'D']}
    from plotting import Plotting
    for session in [4,3,2,1]:
        print(f'Plotting session {session}')
        for bank in tqdm(available_banks[f'{session}']):
            settings['data']['session'] = session
            settings['data']['bank'] = bank
            gbt = Plotting(settings)
            gbt.spectralRFI(session=session, bank=bank)
    return


def plot_all_noisestats(settings, nchannels, object=None, band=None):
    from tqdm import tqdm
    available_banks = {'1': ['B', 'C', 'D', 'F', 'G', 'H'],
                       '2': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       '3': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                       '4': ['B', 'C', 'D']}
    sessions_dict = {'C':  [4],
                     'Ku': [1,2,3]}
    if band is not None:
        sessions = sessions_dict[f'{band}']
    else:
        sessions = [4,3,2,1]
    from plotting import Plotting
    for session in sessions:
        print(f'Plotting session {session}')
        for bank in tqdm(available_banks[f'{session}']):
            settings['data']['session'] = session
            settings['data']['bank'] = bank
            settings['flagging']['dataset_nbins'] = nchannels
            gbt = Plotting(settings)
            gbt.noisestats(session=session, bank=bank, nchannels=nchannels, object=object)
    return


def plot_all_caldata(settings, band=None):
    sessions_dict = {'C':  [4],
                     'Ku': [1,2,3]}
    if band is not None:
        sessions = sessions_dict[f'{band}']
    else:
        sessions = [4,3,2,1]
    from plotting import Plotting
    for session in sessions:
        settings['data']['session'] = session
        plot = Plotting(settings)
        plot.calibrated_data(settings)


def plot_datacuts(settings,nchannels=None):
    from plotting import Plotting
    for band in ['C','Ku']:
        plot = Plotting(settings)
        if nchannels is None:
            nchannels = settings['flagging']['dataset_nbins']
        plot.noisestat_distribution(band, nchannels=nchannels, object='daisy_center')



def maketods(settings,session,median_filter_length,name):
    from processing import Processing
    for session in [session]:
        gbt = Processing(settings)
        gbt.todmaker(name=name, session=session, median_filter_length=median_filter_length, create_destriping_dataset=True)



def todweights(settings,session,median_filter_length,name):
    from processing import Processing
    for session in [session]:
        gbt = Processing(settings)
        for feed in [1]:
            gbt.tod_weights(name=name, session=session, feed=feed, object='daisy_center', bad_scan_ranges=None, weight_by_whitenoise=False)


#
#
# # Schedule plot
# if settings['plotting']['schedule']:
#     for session in np.arange(4):
#         plot.schedule(datadir=settings['dir']['data'], session=settings['data']['session'], saveplot=True)
#
#
# # Hit maps
# if 'hits' in plots:
#     # Read Full Galactic Coordinates
#     l_Ku, b_Ku = read_galactic_coordinates(datadir, band='Ku', object_name='daisy_center')
#     l_C,  b_C  = read_galactic_coordinates(datadir, band='C', object_name='daisy_center')
#
#     # Plot hits maps
#     for pix_per_beam in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
#         plot_hits(l_Ku, b_Ku, l_C, b_C, pix_per_beam)
#
# # One to one plots
# if 'simple_plots' in plots:
#     plot_data('LST','DURATION', 'LST (s)', 'Duration', 'Duration', savename=None, show=True)
#     plot_data('LST', 'HUMIDITY', 'LST (s)', 'Humidity', 'Humidity', savename=None, show=True)
#     plot_data('LST', 'TAMBIENT', 'LST (s)', 'Temperature (K)', 'Weather', savename=None, show=True)
#     plot_data('AZIMUTH', 'ELEVATIO', 'Azimuth (deg)', 'Elevation (deg)', 'AZ-EL', savename=None, show=True)
#     plot_data('LST', 'PRESSURE', 'LST (s)', 'Pressure (mmHg)', 'Pressure', savename=None, show=True)
