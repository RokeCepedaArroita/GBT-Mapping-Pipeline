''' GBT Pipeline '''

import numpy as np
import matplotlib.pyplot as plt
from config import settings
from wrapper import *
from IO import IO
from plotting import Plotting
from processing import Processing


# Read and extract the raw TODs in firstKu TOD_Session1_feed1_firstKu_FULL_DES.h5

#maketods(settings, name='thesis_2petals', session=1, median_filter_length=120)
#maketods(settings, name='thesis_2petals', session=2, median_filter_length=120)
#maketods(settings, name='thesis_2petals', session=3, median_filter_length=120)
#maketods(settings, name='thesis_2petals', session=4, median_filter_length=240)
todweights(settings, name='thesis_2petals', session=4, median_filter_length=240)


#GBT = Plotting(settings)
#GBT.schedule()





















# GBT = IO(settings)
# GBT.merge_filelists(band='C', name='firstC')
#

asdasdasdasdasd
# Create TODs

GBT = Processing(settings)
for session in [4]:
    for feed in [1,2]:
        GBT.tod_weights(name='whitenoiseKu', session=session, feed=feed, object='daisy_center')
        #GBT.todmaker(name='firstC', session=session)




qweqweqweqweqweqwe


GBT = IO(settings)
GBT.merge_filelists(band='Ku', name='whitenoiseKu')


qweqweqweqweqweqweqwe


# Create TODs

GBT = Processing(settings)
for session in [4]:
    for feed in [1,2]:
        #GBT.tod_weights(name='whitenoiseKu', session=session, feed=feed, object='daisy_center')
        GBT.todmaker(name='firstC', session=session)




qweqweqweqweqweqwe


# Merge filelists

GBT = IO(settings)
GBT.merge_filelists(band='Ku', name='firstKu')



asdasdasdasdasqweqwe




#
# plot_datacuts(settings,nchannels=128)

#plot_all_noisestats(settings, nchannels=4, object='daisy_center', band='Ku')
# plot_datacuts(settings)

# GBT = Processing(settings)
# for session in [4]:
#     # GBT.todmaker(name='final', session=session)
#     GBT.tod_weights(name='final', session=session, object='daisy_center')
