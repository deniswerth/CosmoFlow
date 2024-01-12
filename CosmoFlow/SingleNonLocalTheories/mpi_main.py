from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import time

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

start_time = time.time()

from squeezed import *


# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("RANK = ", rank)

#-----------

n = 50
kappa = np.logspace(-1, 0, n)

#-----------


def recombine(K):
    Result = []
    L = max(len(K[i]) for i in range(len(K)))
    for i in range(L):
        for j in range(len(K)):
            try:
                Result.append(K[j][i])
            except Exception:
                pass
    Result = np.asarray(Result)
    return Result


count = n // size  # number of catchments for each process to analyze
remainder = n % size  # extra catchments if n is not a multiple of size

if rank < remainder:  # processes with rank < remainder analyze one extra catchment
    start = rank * (count + 1)  # index of first catchment to analyze
    stop = start + count + 1  # index of last catchment to analyze
else:
    start = rank * count + remainder
    stop = start + count


local_kappa = kappa[rank:n:size] #kappa[start:stop] # get the portion of the array to be analyzed by each rank
local_results = np.empty(local_kappa.shape)  # create result array
local_results[:local_kappa.shape[0]] = local_kappa  # write parameter values to result array
local_results[:] = squeezed(local_results[:])  # run the function for each parameter set and rank

# send results to rank 0
if rank > 0:
    comm.Send(local_results, dest = 0, tag = 14)  # send results to process 0
else:
    final_results = []
    final_results.append(local_results)
    #final_results = np.copy(local_results)  # initialize final results with results from process 0
    for i in range(1, size):  # determine the size of the array to be received from each process
        if i < remainder:
            rank_size = count + 1
        else:
            rank_size = count
        tmp = np.empty(rank_size)  # create empty array to receive results
        comm.Recv(tmp, source = i, tag = 14)  # receive results from the process
        #final_results = np.concatenate((final_results, tmp))  # add the received results to the final results
        final_results.append(tmp)
    final_results = recombine(final_results)
    Shape = final_results
    print("results")
    print(Shape)
    print("The code took", time.time() - start_time, "sec to run")
    np.save("kappa.npy", kappa)
    np.save("Shape", Shape)
    plt.semilogx(kappa, Shape/kappa)
    plt.show()


