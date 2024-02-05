# Functions and globals for shared memory pool
# Import * on this module

import numpy as np
from multiprocessing import Pool, Process, RawArray, shared_memory, managers, Queue, Lock
import copy

var_dict = {}
shm_data = {'shm_list':{}}
shm_lock = Lock()


def create_shared_memory(rcvrmodes):
    """
    Creates a shared memory dataset using the .data attribute is reciever/write mode modules
    """
    
    # Save the big datasets to global variables, then delete them from the objects
    for rcvrmode in rcvrmodes:
        if hasattr(rcvrmode,'data'):
            X_shape  = rcvrmode.data.shape
            X_type   = rcvrmode.data.dtype
            var_dict['shape_{}'.format(rcvrmode.__name__)] = X_shape
            var_dict[ 'type_{}'.format(rcvrmode.__name__)] = X_type

            shm = shared_memory.SharedMemory(create=True,size=rcvrmode.data.nbytes,name=rcvrmode.__name__)
            np_array = np.ndarray(rcvrmode.data.shape, dtype=rcvrmode.data.dtype, buffer=shm.buf)
            np_array[:] = rcvrmode.data[:]
            var_dict['data_{}'.format(rcvrmode.__name__)] = shm.name
            rcvrmode.data = None
            shm_data['shm_list'][rcvrmode.__name__] = shm

def clear_shared_memory(mode):
    """
    Clears a shared memory dataset (e.g. set to zero) that is in the global
    memory dictionary (var_dict)
    """
    
    X_shape = var_dict['shape_{}'.format(mode)]
    X_type  = var_dict[ 'type_{}'.format(mode)]
    shm  = shared_memory.SharedMemory(name=mode)
    data = np.ndarray(X_shape, dtype=X_type, buffer=shm.buf) 
    data *= 0
    shm.close()

def get_shared_memory(name):
    """
    Returns copy of a shared memory block stored in var_dict
    """

    X_shape = var_dict['shape_{}'.format(name)]
    X_type  = var_dict[ 'type_{}'.format(name)]
    existing_shm  = shared_memory.SharedMemory(name=shm_data['shm_list'][name].name)
    mdata = copy.copy(np.ndarray(X_shape, dtype=X_type, buffer=existing_shm.buf))
    existing_shm.close()
    return mdata
