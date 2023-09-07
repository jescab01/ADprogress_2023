import pickle

import pandas as pd
from mpi4py import MPI
import numpy as np
from braak_parallel import *

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 4 python mpi_braak.py
"""

name = "BraakStages"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

## Define param combinations
# Common simulation requirements
reps = 100
modes = ["ab_pos", "ab_ant", "ab_rand", "tau_rand", "rand", "fixed"]

        # maxHe, mCie, mCee, maxTAU2SC, rho, HAdamrate, ABinh, TAUinh
params = [[mode, r] for mode in modes for r in range(reps)]


params = np.asarray(params, dtype=object)
n = params.shape[0]


## Distribution of task load in ranks
count = n // size  # number of catchments for each process to analyze
remainder = n % size  # extra catchments if n is not a multiple of size

if rank < remainder:  # processes with rank < remainder analyze one extra catchment
    start = rank * (count + 1)  # index of first catchment to analyze
    stop = start + count + 1  # index of last catchment to analyze
else:
    start = rank * count + remainder
    stop = start + count


local_params = params[start:stop, :]  # get the portion of the array to be analyzed by each rank

local_results = braak_parallel(local_params)  # run the function for each parameter set and rank

print("Rank %i ha finalizado las simulaciones" % rank)

if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0

    print("Rank %i ha enviado los datos" % rank)

else:  ## MASTER PROCESS _receive, merge and save results
    final_results = np.copy(local_results)  # initialize final results with results from process 0
    for i in range(1, size):  # determine the size of the array to be received from each process

        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if tmp is not None:  # Sometimes temp is a Nonetype wo/ apparent cause
            final_results = np.vstack((final_results, tmp))  # add the received results to the final results

    print("Main ha recogido los datos; ahora guardar.")

    fResults_df = pd.DataFrame(final_results, columns=["mode", "r",
                                                       "time", "fpeak", "relpow_alpha",
                                                       "avgrate_pos", "avgrate_ant", "avgfc_pos", "avgfc_ant",
                                                       "rI", "rII", "rIII", "rIV", "rV"])

    ## Save results
    ## Folder structure - Local
    if "Jesus CabreraAlvarez" in os.getcwd():
        wd = os.getcwd()

        main_folder = wd + "\\" + "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)
        specific_folder = main_folder + "\\PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")

        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

        fResults_df.to_csv(specific_folder + "/results.csv", index=False)

        # with open(specific_folder + "/results.pkl", "wb") as f:
        #     pickle.dump(final_results, f)
        #     f.close()
        # np.save(specific_folder + "/results.pkl", final_results, allow_pickle=True)


    ## Folder structure - CLUSTER
    else:
        main_folder = "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)

        os.chdir(main_folder)

        specific_folder = "PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

        os.chdir(specific_folder)

        fResults_df.to_csv("results.csv", index=False)

        # np.save("results.pkl", final_results, allow_pickle=True)

        # with open("results.pkl", "wb") as f:
        #     pickle.dump(final_results, f)
        #     f.close()
