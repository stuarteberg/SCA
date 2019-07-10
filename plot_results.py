import numpy as np
import matplotlib.pyplot as pl
import h5py as h5
from sca_functions import *

def plot_data(data, title = ''):
    fig = pl.figure(figsize = (15,15))
    x,y = data.shape

    pl.title(title)
    pl.imshow(data, aspect = y/x)
    pl.colorbar()
    pl.savefig(title, transparent = True)
    pl.show()

def order_mat(r_nt):
    import numpy as np
    ord_ = list(np.argsort(np.array([np.argwhere(i ==i.max())[0] for i in r_nt]).squeeze()))
    return r_nt[ord_],ord_

def proj_back_n(data_ncomp_t,u_n_nk):
    return u_n_nk.dot(data_ncomp_t)


if __name__ == "__main__":

    # load data
    filepath_data = './churchland.h5'
    data_ntk = h5.File(filepath_data, 'r')['d']['value'][()]

    filepath_proj = './proj_nk_t_k.npy'
    proj_nk_t_k = np.load(filepath_proj)
    proj_nk_tk_avg = proj_nk_t_k.mean(2)

    filepath_usvh = './usvh.npy'
    usvh = np.load(filepath_usvh).item()

    filepath_sc = './sc.npy'
    sc = np.load(filepath_sc).item()

    ncomp = usvh['ncomp']
    print(ncomp)
    u = usvh['u'][:,:ncomp]
    ## back project
    sc_umat_back_proj = [proj_back_n(data,u) for data in sc['umats']]

    ## order by peak firing time
    umats_ordered = np.array([order_mat(umat)[0] for umat in  sc['umats']])

    umats_bp_ordered = np.array([order_mat(umat_bp)[0] for umat_bp in sc_umat_back_proj])

    proj_ordered, order = order_mat(proj_nk_tk_avg)
    data_avg_ordered, order = order_mat(data_ntk.mean(2))

    tile = tile_im(umats_ordered[0:8],4)
    tile_bp = tile_im(umats_bp_ordered[0:8],4)

    ###################### plotting functions #############################
    title = 'singular vectors (u0)'
    plot_data(usvh['u'], title = title)
    title = 'singular values'
    fig = pl.figure(figsize = (21,3))
    pl.plot(usvh['s'], 'o')
    pl.title(title)
    pl.show()

    title = 'sc umat'
    plot_data(umats_ordered[0], title = title)

    title = 'sc umat back projected through u'
    plot_data(umats_bp_ordered[0], title = title)

    # title = 'umats ordered'
    # plot_data(tile, title = title)
    #
    # title = 'umats bp ordered'
    # plot_data(tile_bp, title = title)

    title = 'average of the projected data (ordered)'
    plot_data(proj_ordered, title = title)

    title = 'trial average data'
    plot_data(order_mat(data_avg_ordered)[0], title = title)
