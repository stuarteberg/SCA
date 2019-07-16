import numpy as np
import matplotlib.pyplot as pl
from scipy.linalg import solve_lyapunov
from numpy.linalg import inv
import h5py as h5

def construct_chain(n, weight):
    ## build the 'A' matrix of a 'chain network';
    ## the weights decrease (bell-shape) as you go down to avoid to avoid build-up of variance down the chain (in the stochastic network)
    A = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == j+1:
                A[i,j] = 1 + weight*np.exp(-(i/.5)**2)
    A -= np.eye(n)
    return A

def construct_reversible(C):
    ## constructs the A matrix of a time-reversible system which achieves a desired spatial covariance i.e.: S = 0
    return -0.5 * inv(C)

def construct_noise(n, factor):
    n_sub = n//2
    ell_noise =  np.zeros([n, n]) #cholesky factor of the input noise; just a diagonal matrix of 1 (for the first subsys) and sqrt(factor) for the second subsys
    ell_noise[:n_sub, :n_sub] = np.diag(np.ones(n_sub))
    ell_noise[-n_sub:, -n_sub:] = np.diag(np.ones(n_sub)*factor)
    ell_noise = np.sqrt(ell_noise)
    return ell_noise

def simulate_dynamics(n, k, duration, dt, sampling_dt, a, c, ell_noise):
    n_bins = int(duration/dt)
    sample_every = int(sampling_dt/dt)
    accu = [] # accumulator
    ell = np.linalg.cholesky(c)
    x0 = ell @ np.random.randn(n, k) # draw initial conditions (NxK) from the stationary distribution already

    for t in range(n_bins):
        noise = np.sqrt(dt/tau) * ell_noise @ np.random.randn(n, k) # draw the noise
        if t == 0:
            print('starting... \nt = 0')
            x = x0 + dt/tau * a @ x0 + noise
        else:
            x = x + dt/tau * a @ x + noise

        if np.mod(t, sample_every) ==0:
            accu.append(x)
        if t == n_bins:
            print('finished')
    return np.array(accu).transpose([1,0,2])


if __name__ == '__main__':

    k = 500 # number of trials
    n_sub = 25 # number of neurons in each sub-pop
    n = 2 * n_sub # total number of neurons
    dt = 1e-4 # simulation time step
    sampling_dt = 2e-3 # sampling resolution

    duration = 0.5 # trial duration
    tau = 20 * 1e-3 # membrane time constant

    # fig = pl.figure()
    # pl.imshow(A)
    # pl.show()

    factor = 3.0 # sequential (chain) sub-system; 0.2 is a good number, too :D
    weight = 0.2
    a_seq = construct_chain(n_sub, weight)
    # fig = pl.figure()
    # pl.imshow(a_seq)
    # pl.show()
    c_seq = solve_lyapunov(a_seq, -np.eye(n_sub))# find the controllability Gramian, ie stationary covariance matrix when the input is pure white noise -- which it will be

    # reversible sub-system that matches the Gramian, but scaled to speed up the fluctuations which would otherwise be super slow
    a_rev = factor * construct_reversible(c_seq)

    z = np.zeros([n_sub, n_sub])
    a = np.vstack([np.hstack([a_seq, z]), np.hstack([z, a_rev])])
    # fig = pl.figure()
    # pl.title('connectivity matrix')
    # pl.imshow(a, cmap = 'gray')
    # pl.show()

    ell_noise = construct_noise(n, factor)
    # fig = pl.figure()
    # pl.title('noise')
    # pl.imshow(ell_noise @ ell_noise.T, cmap = 'gray')
    # pl.colorbar()
    # pl.show()
    # overall Gramian
    c = solve_lyapunov(a, -ell_noise @ ell_noise.T)

    print(np.all(np.linalg.eigvals(c) > 0))
    x = simulate_dynamics(n, k, duration, dt, sampling_dt, a, c, ell_noise)
    print(x.shape)

    path = "/Users/virginiarutten/Desktop/SCA/results/test_data.hdf5"
    with h5.File(path, 'w') as f:
        dset = f.create_dataset("data", data=x)
