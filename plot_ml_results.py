

import numpy as np

def load_data(path):
    with open(path, mode='rb') as file: # b is important -> binary
        X = file.read()
    X = np.fromfile(path, dtype="float32")
    print(X[0])

    print(X.shape)
    return X



if __name__ == "__main__":

    path = './results/toy_data.bin'
    n = 50
    k = 500
    X = load_data(path)
    X = X.reshape([n, -1, k])
