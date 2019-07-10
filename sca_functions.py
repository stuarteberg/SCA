import numpy as np

def convolve_im(data_n_t, umats):
    from scipy.signal import fftconvolve
    ts = []
    for u in umats:
        ts.append(fftconvolve(data_n_t,u[:,::-1],axes = 1, mode = 'same').mean(axis = 0))
    return np.array(ts)



def order_hippo(r_nt):
    import numpy as np
    ord_ = list(np.argsort(np.array([np.argwhere(i ==i.max())[0] for i in r_nt]).squeeze()))
    return r_nt[ord_],ord_



## kronecker functions
def pc_project_data_nt(X_n_t,nt = [0,1], n_comp = 100):
    from sklearn.decomposition import PCA
    import numpy as np
    ndim,tdim = nt
    X_n_t = np.transpose(X_n_t,[ndim,tdim])
    print('calculating spatial principle components')
    X_npc_t, pc_n, pca = applyPCA_nt(X_n_t, n_comp) # get spatial pcs but don't apply
    return X_npc_t, pc_n, pca



def applyPCA_nt(X_features_trials, num):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num)
    pca.fit(X_features_trials.T)
    X_projected = pca.components_.dot(X_features_trials)
    return X_projected, pca.components_, pca

def data_nt_2_trials(data_nt,tk = 10):
    n, tt = data_nt.shape
    k = tt//tk
    return data_nt[:,:tk*k].reshape([n,k, tk])

def reshape_pcs(pc_n, x,y):
    return pc_n.reshape([-1,x,y])

def kron_factorise(X_npc_tpc_k, ntk = [0,1,2], spc_num = 5, kr_num = 5):
    ndim, tdim, kdim = ntk
    # reshape to standard form
    X_npc_tpc_k = np.transpose(X_npc_tpc_k,[ndim,tdim,kdim])
    n,t,k = X_npc_tpc_k.shape
    # get cdiff an kronecker factors
    kron = False
    if kron ==False:
        cdiff, cplus = get_cdiff_ntk(X_npc_tpc_k, ntk = [0,1,2], kron = kron, kr_num = kr_num)
    else:
        cdiff, cplus, eigsS, eigsT, krs = get_cdiff_ntk(X_npc_tpc_k, ntk = [0,1,2], kron = True, kr_num = kr_num)
    # get PSCs
    umats, u, s, ratio = get_comp_from_cdiff(cdiff, n, t, num = spc_num)
    umats_plus, u_plus, s_plus, ratio_plus = get_comp_from_cdiff(cplus, n, t, num = spc_num)

    sc = {}
    sc['umats'] = umats
    sc['u'] = u
    sc['ratio'] = ratio
    sc['cdiff'] = cdiff

    sc['umatsp'] = umats_plus
    sc['up'] = u_plus
    sc['ratiop'] = ratio_plus
    sc['cplus'] = cplus

    if kron ==True:
        kr = {}
        kr['s'] = eigsS
        kr['t'] = eigsT
        kr['ratio'] = krs

        return kr, sc
    else:
        return sc



def get_cdiff_ntk(X_npc_tpc_k, ntk = [0,1,2], kron = True, kr_num = 5):
    import numpy as np
    from numpy.linalg import svd, norm
    ndim, tdim, kdim = ntk
    X_npc_tpc_k = np.transpose(X_npc_tpc_k,[ndim,tdim,kdim])
    n_comp, t_comp, k = X_npc_tpc_k.shape
    print(n_comp, t_comp,k)
    vectorized = X_npc_tpc_k.reshape([n_comp*t_comp, k])
    vectorized_demeaned = vectorized - np.expand_dims(vectorized.mean(1),1)
    c1 = vectorized_demeaned.dot(vectorized_demeaned.T)/k
    c2 = c1.reshape([n_comp, t_comp, n_comp, t_comp])
    c3 = np.swapaxes(c2, 0, 2)
    c3 = c3.reshape([n_comp*t_comp,n_comp*t_comp])
    cdiff = c1-c3
    cplus = (c1+c3)
    if kron:
        ck = np.swapaxes(c2, 1, 2) # covariance for kronecker
        ck = ck.reshape([n_comp**2,t_comp**2])
        uc, cs, vhc = svd(ck, full_matrices=False)
        ucovs = np.swapaxes(uc.reshape([n_comp,n_comp,-1]),0,2)
        vcovs = vhc.reshape([-1,t_comp,t_comp])
        vmax = kr_num
        eigsT = [get_eigs(v) for ind, v in enumerate(vcovs[:vmax])] # tuples of both eigvals and vectors
        eigsS = [get_eigs(v) for ind, v in enumerate(ucovs[:vmax])] # tuples of both eigvals and vectors
        return cdiff, cplus, eigsS, eigsT, cs/cs.sum()
    else:
        return cdiff, cplus


def get_comp_from_cdiff(cdiff, n_comp, t_comp, num = 5):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=num, n_iter=7, random_state=42).fit(cdiff)
    vmax = svd.components_.shape[0]
    cdiff_u = [svd.components_[i].reshape([n_comp, t_comp]) for i in range(vmax)]
    return cdiff_u, svd.components_.T, svd.singular_values_, svd.explained_variance_ratio_

def proj_back_n(data_ncomp_t,pcn):
    return pcn.T.dot(data_ncomp_t)

def proj_back_nt(data_ncomp_t,pcn, pct):
    return pcn.T.dot(data_ncomp_t).dot(pct)

def get_eigs(A):
    import numpy as np
    from numpy.linalg import eig
    import numpy.linalg as la
    eval, evec = la.eig(A)
    ord = np.argsort(eval)[::-1]
    return (eval[ord],evec[:,ord])

def getPCmaps(data_nt, list_inds_zyx):
    import numpy as np
    from scipy.linalg import svd
    N, T = data_nt.shape
    u, s, vh = svd(data_nt.T, full_matrices=False)
    pc = {}
    pc['u'] = u
    pc['vh'] = vhsha
    pc['s'] = s
    vol = fill_vol(vh0, list_inds_zyx)
    return vol, pc


def applyRasterMap(data_nt, n_X = 30):
    import numpy as np
    from rastermap import Rastermap
    model = Rastermap(n_components=1, n_X=30, nPC=200, init='pca')
    model = model.fit(data_nt)
    isort_full = np.argsort(model.embedding[:,0])
    return isort_full


def tile_im(ims,h):
    # first dimension needs to be time
    import numpy as np
    ndims =ims.shape
    if len(ndims) ==3:
        w = int(np.ceil(ndims[0]/h))
        z =np.zeros([h*ndims[1],w*ndims[2]])
        ind = 0
        for i in range(h):
            for j in range(w):
                if ind<ndims[0]:
                    z[i*ndims[1]:(i+1)*ndims[1],j*ndims[2]:(j+1)*ndims[2]] = ims[ind]
                    ind +=1
    if len(ndims) ==4:
        w = int(np.ceil(ndims[1]/h))
        print(w)
        z =np.zeros([ndims[0],h*ndims[2],w*ndims[3]])
        ind = 0
        for i in range(h):
            for j in range(w):
                if ind<ndims[1]:
                    z[:, i*ndims[2]:(i+1)*ndims[2],j*ndims[3]:(j+1)*ndims[3]] = ims[:, ind]
                    ind +=1
    return z
