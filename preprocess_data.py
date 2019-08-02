## import necessary packages
import numpy as np
from numpy.linalg import svd
import h5py as h5
from sca_functions import *
import matplotlib.pyplot as pl
import B.general_admin as ga
import os, sys
import multiprocessing
from functools import partial
from contextlib import contextmanager
import pickle4reducer
import multiprocessing as mp
from itertools import repeat
from multiprocessing import RawArray
from operator import mul
from functools import reduce
from multiprocessing import Pool
import sys
import time



def reindex_data(data_ntk, ntk = [0,1,2]):
    # reshapes data
    return data_ntk.transpose(ntk)


def construct_adiff(j, data_ntk):
    # data_ntk = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    n, t, k = data_ntk.shape
    jtk = data_ntk[j]
    mat_nt2 = np.zeros([n, t**2])
    print(j)
    for i in range(n):
        itk = data_ntk[i]
        # construct TxT covariance matrix
        ai = itk.dot(jtk.T)/k
        # construct small cdiff and reshape into 1x(T^2)
        ai_diff = (ai - ai.T).reshape([1, t**2])
        mat_nt2[i] = ai_diff
    return mat_nt2.dot(mat_nt2.T)


def demean_data(data_ntk):
    data_mean = np.expand_dims(data_ntk.mean(2),2)
    return data_ntk - data_mean


if __name__ == "__main__":
    # i = sys.argv[1]
    base_dir = '/nrs/ahrens/Virginia_nrs/25_LS_wG/190703_HuCH2B_gCaMP7F_8dpf_forG/'
    experiment = 'exp0/'
    folder_name = base_dir + experiment
    dirs = ga.get_subfolders(folder_name, make_plot_folder=False)
    os.chdir(dirs['sca'])
    x = np.load(dirs['dff'] + 'dff_mean_full.npy') # T by N
    ncomp = 6 # number of SCA components
    t = 50 #
    pre = 100 # number of frames to discard from the start of the recording
    t0 = time.time()
    data_tkn = x[pre:][:((x.shape[0]-pre)//t)*t,:].reshape([-1,t, x.shape[1]])
    data_ntk = reindex_data(data_tkn, ntk = [2,1,0]) # reindex data if necessary
    n,t,k = data_ntk.shape
    print('n: {0}, t: {1}, k: {2}'.format(n,t,k))
    print('removing mean')
    data_ntk_demeaned = demean_data(data_ntk)
    np.save('data_ntk_demeaned',data_ntk_demeaned)
