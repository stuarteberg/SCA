import numpy as np
import matplotlib.pyplot as pl
from scipy.linalg import solve_lyapunov
from numpy.linalg import inv

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
    ell_noise[:n_sub, :n_sub] = 1
    ell_noise[-n_sub:, -n_sub:] = factor
    ell_noise = np.sqrt(ell_noise)
    return ell_noise

def simulate_dynamics(n, k, duration, dt, sampling_dt, a, c, ell_noise):

    n_bins = duration/dt
    sample_every = sampling_dt/dt
    accu = [] # accumulator

    ell = np.linalg.cholesky(c)
    x0 = ell @ np.rand.randn(n, k) # draw initial conditions (NxK) from the stationary distribution already

    for t in range(n_bins):
        noise = np.sqrt(dt/tau) * ell_noise * np.rand.randn(n, k) # draw the noise
        x = x + dt/tau*a + noise

        if np.mod(t, sample_every):
            accu.append(x)
        if t == n_bins:
            print('finished')
    return accu


if __name__ == '__main__':

    k = 500 # number of trials
    n_sub = 25 # number of neurons in each sub-pop
    n = 2 * n_sub # total number of neurons
    dt = 1e-4 # simulation time step
    sampling_dt = 2e-3 # spatial sampling resolution

    duration = 0.5 # trial duration
    tau = 20*1e-3 # membrane time constant

    # fig = pl.figure()
    # pl.imshow(A)
    # pl.show()

    factor = 3.0 # sequential (chain) sub-system; 0.2 is a good number, too :D
    weight = 0.2
    a_seq = construct_chain(n_sub, weight)
    c_seq = solve_lyapunov(a_seq, -np.eye(n_sub))# find the controllability Gramian, ie stationary covariance matrix when the input is pure white noise -- which it will be

    # reversible sub-system that matches the Gramian, but scaled to speed up the fluctuations which would otherwise be super slow
    a_rev = factor * construct_reversible(c_seq)

    z = np.zeros([n_sub, n_sub])
    a = np.vstack([np.hstack([a_seq, z]), np.hstack([a_rev, z])])

    ell_noise = construct_noise(n, factor)

    # overall Gramian
    c = solve_lyapunov(a, -ell_noise @ ell_noise.T)

    print(np.all(np.linalg.eigvals(c) > 0))
    x = simulate_dynamics(n, k, duration, dt, sampling_dt, a, c, ell_noise)
