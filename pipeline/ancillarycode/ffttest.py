import numpy as np
import matplotlib.pyplot as plt

# Generate white noise and do an FFT
std = 20
npoints = 5000

def prep_fft(data, data_frequency_Hz):
    ''' Perform FFT and return power spectrum
    A constant data recording rate is assumed '''
    from numpy.fft import fft
    # Number of sample points
    N = len(data)
    # Sample spacing in seconds
    T = 1./data_frequency_Hz
    # Create FFT axes
    yf = fft(data)
    power = 1/N * np.abs(yf[0:N//2])**2
    freq = np.linspace(0.0, 1.0/(2.0*T), N//2)
    return freq[1:], power[1:] # we are not interested in the first zero frequency bin


randomdata = np.random.normal(0,std,npoints)
freq, power = prep_fft(data=randomdata, data_frequency_Hz=10)
plt.loglog(freq,power)
plt.loglog([np.nanmin(freq),np.nanmax(freq)],[std**2, std**2],'k--')
plt.loglog([np.nanmin(freq),np.nanmax(freq)],[np.mean(power), np.mean(power)],'r--')
plt.xlim([np.nanmin(freq),np.nanmax(freq)])
plt.show()
