from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from numpy import get_include
from Cython.Build import cythonize
import os

try:
    slalib_path = os.environ['SLALIB_LIBS']
except KeyError:
    slalib_path = '/star/lib' # default path of Manchester machines
    print('Warning: No SLALIB_LIBS environment variable set, assuming: {}'.format(slalib_path))


pysla = Extension(name = 'sdosim.Tools.pysla', 
                  sources = ['sdosim/Tools/pysla.f90'],
                  libraries=['sla'],
                  library_dirs =['{}'.format(slalib_path)],
                  f2py_options = [],
                  extra_f90_compile_args=['-L{}'.format(slalib_path),'-L{}/libsla.so'.format(slalib_path)])

binFuncs = Extension(name='sdosim.Tools.binFuncs',
                     include_dirs=[get_include()],
                     sources=['sdosim/Tools/binFuncs.pyx'])
#extensions = [binFuncs]
#setup(ext_modules=cythonize(extensions))

config = {'name':'sdosim',
          'version':'0.1dev',
          'packages':['sdosim',
                      'sdosim.Observations',
                      'sdosim.ReceiverModels',
                      'sdosim.Scheduler',
                      'sdosim.WriteModes',
                      'sdosim.Tools'],
          'ext_modules':cythonize([binFuncs,pysla])}



setup(**config)
