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


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def reindex_data(data_ntk, ntk = [0,1,2]):
    # reshapes data
    return data_ntk.transpose(ntk)

def construct_adiff(j, data_ntk):
    n, t, k = data_ntk.shape
    jtk = data_ntk[j]
    mat_nt2 = np.zeros([n, t**2])
    for i in range(n):
        itk = data_ntk[i]
        # construct TxT covariance matrix
        ai = itk.dot(jtk.T)/k
        # construct small cdiff and reshape into 1x(T^2)
        ai_diff = (ai - ai.T).reshape([1, t**2])
        mat_nt2[i] = ai_diff
    return mat_nt2.dot(mat_nt2.T)



def construct_M(data_ntk):
    n, t, k = data_ntk.shape
    # create matrix to store results
    M = np.zeros([n, n])
    for j in range(n):
        if np.mod(j, 5000):
            print(j)
        jtk = data_ntk[j]
        # create matrix to hold temporary result
        mat_nt2 = np.zeros([n, t**2])
        for i in range(n):
            itk = data_ntk[i]
            # construct TxT covariance matrix
            ai = itk.dot(jtk.T)/k
            # construct small cdiff and reshape into 1x(T^2)
            ai_diff = (ai - ai.T).reshape([1, t**2])
            mat_nt2[i] = ai_diff
        M += mat_nt2.dot(mat_nt2.T)
    return M

def project_data(data_ntk, u):
    n, t, k = data_ntk.shape
    n, nk = u.shape
    proj_nn_t_k = u.T.dot(data_ntk.reshape([n, t*k])).reshape([nk, t,k])
    return proj_nn_t_k

def demean_data(data_ntk):
    data_mean = np.expand_dims(data_ntk.mean(2),2)
    return data_ntk - data_mean


def plot_data(data, title = ''):
    fig = pl.figure(figsize = (15,15))
    x,y = data.shape
    pl.title(title)
    pl.imshow(data, aspect = y/x)
    pl.colorbar()
    pl.savefig(title, transparent = True)
    pl.show()


if __name__ == "__main__":

    ncomp = 6
    ctx = mp.get_context()
    ctx.reducer = pickle4reducer.Pickle4Reducer()

    # load data
    # filepath = './churchland.h5'
    # data_ntk = h5.File(filepath, 'r')['d']['value'][()]
    # print('data dimension: {0}'.format(data_ntk.shape))
    # filepath = './toy_for_v.h5'
    # data_ntk = h5.File(filepath, 'r')['data'][()]
    # print('reindexing...')

    base_dir = '/nrs/ahrens/Virginia_nrs/25_LS_wG/190703_HuCH2B_gCaMP7F_8dpf_forG/'

    experiment = 'exp0/'
    folder_name = base_dir + experiment
    dirs = ga.get_subfolders(folder_name, make_plot_folder=False)
    os.chdir('/groups/ahrens/home/ruttenv/code/utils/SCA/')

    x = np.load(dirs['dff'] + 'dff_mean_full.npy')
    k = 50
    pre = 100
    data_tkn = x[pre:][:((x.shape[0]-pre)//k)*k,:].reshape([k,-1, x.shape[1]])

    data_ntk = reindex_data(data_tkn, ntk = [2,1,0]) # reindex data if necessary
    n,t,k = data_ntk.shape
    print('n: {0}, t: {1}, k: {2}'.format(n,t,k))

    print('removing mean')
    data_ntk_demeaned = demean_data(data_ntk)

    print('constructing M...')
    # with poolcontext(processes=100) as pool:
    #     M = pool.map(partial(construct_adiff, data_ntk=data_ntk_demeaned), np.arange(n))

    with mp.Pool(50) as p:
        M = p.map(partial(construct_adiff, data_ntk=data_ntk_demeaned), np.arange(n))

    print('\nM constructed...')
    # M = construct_M(data_ntk_demeaned)
    np.save('M', M)
    M = np.array(M).sum(0)
    print(M.shape)

    print('\nM constructed and saved...')
    u, s, vh = svd(M, full_matrices=False)
    print('\nsvd on M computed...')

    # plot_data(u, title = 'u')

    usvh = {}
    usvh['u'] = u
    usvh['s'] = s
    usvh['vh'] = vh
    usvh['ncomp'] = ncomp

    proj_nn_t_k = project_data(data_ntk, u[:,:ncomp])
    print('projected data dimension: {0}'.format(proj_nn_t_k.shape))

    proj_nn_t_k_demeaned = demean_data(proj_nn_t_k)
    print('computing SC components')
    sc = kron_factorise(proj_nn_t_k_demeaned, ntk = [0,1,2], spc_num = 5, kr_num = 5)

    np.save('proj_nk_t_k', proj_nn_t_k)
    np.save('usvh', usvh)
    np.save('sc', sc)
