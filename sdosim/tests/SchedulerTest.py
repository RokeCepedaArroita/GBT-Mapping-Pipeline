import numpy as np
from matplotlib import pyplot
from comancpipeline.Tools import Coordinates
import h5py
import healpy as hp
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy import units as u
import copy
from tqdm import tqdm
import pysla
from concurrent import futures
from scipy.interpolate import interp1d

import Scheduler

if __name__ == "__main__":

    scheduler = Scheduler.Scheduler(Time('2019-01-01T00:00:00.000',format='isot'),
                                    SkyCoord(30*u.deg,0*u.deg,frame='galactic').icrs,
                                    0,30,
                                    nRepointings=60,
                                    nDays = 2,
                                    source_lha0=-3,
                                    minElevation=30,
                                    span_ha = 2)

    scheduler.plot_elvlst()
    pyplot.show()
