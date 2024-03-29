## import necessary packages
import numpy as np
from numpy.linalg import svd
import h5py as h5
from sca_functions import *
import matplotlib.pyplot as pl

def reindex_data(data_ntk, ntk = [0,1,2]):
    # reshapes data
    return data_ntk.transpose(ntk)

def construct_M(data_ntk):
    n, t, k = data_ntk.shape
    # create matrix to store results
    M = np.zeros([n, n])
    for j in range(n):
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

    # load data
    filepath = './churchland.h5'
    data_ntk = h5.File(filepath, 'r')['d']['value'][()]
    print('data dimension: {0}'.format(data_ntk.shape))
    filepath = './toy_for_v.h5'
    data_ntk = h5.File(filepath, 'r')['data'][()]
    print('reindexing...')


    data_ntk = reindex_data(data_ntk, ntk = [0,1,2]) # reindex data if necessary

    print('removing mean')
    data_ntk_demeaned = demean_data(data_ntk)

    print('constructing M...')
    M = construct_M(data_ntk_demeaned)
    u, s, vh = svd(M, full_matrices=False)
    plot_data(u, title = 'u')

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
