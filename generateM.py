## import necessary packages
import numpy as np
from numpy.linalg import svd
import h5py as h5
from sca_functions import *
import matplotlib.pyplot as pl
import B.general_admin as ga
import os, sys, time
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



def reindex_data(data_ntk, ntk = [0,1,2]):
    # reshapes data
    return data_ntk.transpose(ntk)


def construct_adiff(j, fname):
    t0 = time.time()
    data_ntk = np.load(fname)
    print('time to load: {0}'.format((time.time()-t0)))
    # print(fname)s
    n, t, k = data_ntk.shape
    jtk = data_ntk[j]
    n = 100
    mat_nt2 = np.zeros([n, t**2])
    print(j)
    for i in range(n):
        if np.mod(i, 5) == 0:
            print('n: {0}'.format(i))
        itk = data_ntk[i]
        # construct TxT covariance matrix
        ai = itk.dot(jtk.T)/k
        # construct small cdiff and reshape into 1x(T^2)
        ai_diff = (ai - ai.T).reshape([1, t**2])
        mat_nt2[i] = ai_diff
    return mat_nt2.dot(mat_nt2.T)


if __name__ == "__main__":
    base_dir = '/nrs/ahrens/Virginia_nrs/25_LS_wG/190703_HuCH2B_gCaMP7F_8dpf_forG/'
    experiment = 'exp0/'
    folder_name = base_dir + experiment
    dirs = ga.get_subfolders(folder_name, make_plot_folder=False)
    os.chdir(dirs['sca'])
    fname = dirs['sca'] + 'data_ntk_demeaned.npy'

    t0 = time.time()
    with mp.Pool(50) as p:
        # M = p.starmap(construct_adiff, zip(np.arange(n), repeat(fname)))
        M = p.map(partial(construct_adiff, fname=fname), np.arange(100))

    t1 = time.time()-t0
    print('\nM constructed...')
    print(len(M))
    np.save('M', M)
    M = np.array(M).sum(0)
    print(M.shape)
    print('total time: {0}'.format((time.time()-t0)))



    #
    # print('\nM constructed and saved...')
    # u, s, vh = svd(M, full_matrices=False)
    # print('\nsvd on M computed...')
    #
    # np.save('proj_nk_t_k', proj_nn_t_k)
    # np.save('usvh', usvh)
    # np.save('sc', sc)
    # print(x.shape)
