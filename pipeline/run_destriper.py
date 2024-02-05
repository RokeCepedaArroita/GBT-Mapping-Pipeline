import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Ellipse

from scipy.interpolate import interp1d
from astropy.io import fits
from astropy import wcs

import h5py

from tqdm import tqdm
import click
import ast
import os

from comancpipeline.Tools import ParserClass,binFuncs
from comancpipeline.MapMaking import DataReader, Destriper

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.argument('filename')#, help='Level 1 hdf5 file')
@click.option('--options', cls=PythonLiteralOption, default="{}")
def call_level1_destripe(filename, options):
    level1_destripe(filename, options)

def level1_destripe(filename,options):

    """Plot hit maps for feeds

    Arguments:

    filename: the name of the COMAP Level-1 file

    """
    # Get the inputs:
    parameters = ParserClass.Parser(filename)
    title = parameters['Inputs']['title']
    for k1,v1 in options.items():
        if len(options.keys()) == 0:
            break
        for k2, v2 in v1.items():
            parameters[k1][k2] = v2
    title = parameters['Inputs']['title']

    # Read in all the data
    if parameters['Inputs']['filelist'][-3:] == '.h5':
        filelist = [parameters['Inputs']['filelist']]
    elif parameters['Inputs']['filelist'][-4:] == '.txt':
        with open(parameters['Inputs']['filelist']) as f:
            content = f.readlines()
        filelist = [x.strip() for x in content]
    else:
        print('Your filelist in the .ini file does not end on h5 or txt for a filelist. Check and try again.')


    data = DataReader.ReadDataLevel2(filelist,parameters,**parameters['ReadData'])
    # print(np.max(data.offset_residuals.offsetpixels))
    # print(data.offset_residuals.Noffsets)
    # pyplot.plot(data.offset_residuals())
    # pyplot.show()
    # pyplot.imshow(data.naive.get_map(),vmin=-1+33,vmax=1+33)
    # pyplot.show()
    offsetMap, offsets = Destriper.Destriper(parameters, data)

    ###
    # Write out the offsets
    ###

    # ????

    ###
    # Write out the maps
    ###
    naive = data.naive.get_map()
    offmap= offsetMap.get_map()
    hits = data.naive.get_hits()
    variance = data.naive.get_cov()

    des = naive-offmap
    des[hits == 0] = np.nan
    clean_map = naive-offmap


    hdu = fits.PrimaryHDU(des,header=data.naive.wcs.to_header())
    cov = fits.ImageHDU(variance,name='Covariance',header=data.naive.wcs.to_header())
    hits = fits.ImageHDU(hits,name='Hits',header=data.naive.wcs.to_header())
    naive = fits.ImageHDU(naive,name='Naive',header=data.naive.wcs.to_header())

    hdul = fits.HDUList([hdu,cov,hits,naive])
    if not os.path.exists(parameters['Inputs']['maps_directory']):
        os.makedirs(parameters['Inputs']['maps_directory'])
    fname = '{}/{}.fits'.format(parameters['Inputs']['maps_directory'],
                                               parameters['Inputs']['title'])

    hdul.writeto(fname,overwrite=True)

if __name__ == "__main__":
    call_level1_destripe()
