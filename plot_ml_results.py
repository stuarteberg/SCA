import matplotlib.pyplot as pl
import numpy as np
import h5py as h5

def load_data(path, dset = 'data'):
    try:
        X = h5.File(path, 'r')[dset][()]
    except:
        X = h5.File(path, 'r')['data']['X'][()]
    # print(X.keys())
    print(X.shape)
    return X

def order_mat(r_nt):
    import numpy as np
    ord_ = list(np.argsort(np.array([np.argwhere(i ==i.max())[0] for i in r_nt]).squeeze()))
    return r_nt[ord_],ord_

if __name__ == "__main__":

    path = './results/toy_data.h5'
    path = './results/test_data.hdf5'
    # ['data']['X']
    X = load_data(path)

    fig = pl.figure(figsize = (15,15))
    pl.title("ginny's data!", y = 1.05)
    # mat, ord = order_mat(X[:,:,5])
    mat = X[:,:,23]
    pl.imshow(mat)
    pl.show()
