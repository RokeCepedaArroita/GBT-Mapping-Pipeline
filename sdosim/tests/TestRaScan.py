import numpy as np
from matplotlib import pyplot
from comancpipeline.Tools import Coordinates
import h5py
import healpy as hp
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import astropy.units as u
from Observations import *
from Observatory import * 
from ReceiverModes import *
from Mapping import Mapper, MapperHPX
from tqdm import tqdm
from profile import profile 
tsys = 20
sample_rate = 50

from concurrent import futures
from multiprocessing import Pool, Process, RawArray, shared_memory, managers, Queue

gl, gb = 30,0
target = Scheduler(SkyCoord(gl*u.deg,0*u.deg,frame='galactic').icrs)
schedule = RaScans(target,ra_distance=1, dec_step=0.1, dec_distance=1, slew_speed=0.1)


schedule.define_scans(50)
