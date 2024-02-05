import numpy as np



# # Tried but failed to modify the fits file
# def add2fits(fitdir, field, data, hdu_number=0):
#     ''' Adds new data to the fits file '''
#
#     from astropy.io import fits
#     hdul = fits.open(fitdir)
#     hdul[hdu_number].header[field] = data
#     print('Overwritting data to add new field...')
#     hdul.writeto(fitdir, overwrite=True)
#     print(f'Field {field} with value {data} has been added to hdu number {hdu_number}!')
#
#     print(hdul[hdu_number].header)
#
#
#     return
#
# # add2fits(fitdir, field='ROKETEST', data='IT WORKS', hdu_number=2)
#
#
# # Now open and print
# data = openfits(fitdir,hdu_number=2, verbose=True)
# print(header.field('ROKECE'))
# asdasdasd
