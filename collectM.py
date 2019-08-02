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
from glob import glob



def load_data(file):
    return np.load(file)


if __name__ == "__main__":
    base_dir = '/nrs/ahrens/Virginia_nrs/25_LS_wG/190703_HuCH2B_gCaMP7F_8dpf_forG/'
    experiment = 'exp0/'
    folder_name = base_dir + experiment
    dirs = ga.get_subfolders(folder_name, make_plot_folder=False)
    os.chdir(dirs['M'])
    files = sorted(glob(dirs['sca'] + 'n*.npy'))
    print(files[0])
    M = load_data(files[0])
    # np.save('M.npy', M)
    for ind, file in enumerate(files[1:]):
        print(ind)
        M = M + load_data(file)

    np.save(dirs['M'] + 'M_shape.npy', 1)

    # np.save(dirs['sca'] + 'M_shape.npy', M.shape)
    np.save('M.npy', M)


    # print('\nM constructed...')
    # # M = construct_M(data_ntk_demeaned)
    # np.save('M', M)
    # M = np.array(M).sum(0)
    # print(M.shape)
    #
    # print('\nM constructed and saved...')
    # u, s, vh = svd(M, full_matrices=False)
    # print('\nsvd on M computed...')
    #
    # np.save('proj_nk_t_k', proj_nn_t_k)
    # np.save('usvh', usvh)
    # np.save('sc', sc)
    # print(x.shape)
