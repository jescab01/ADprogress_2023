import pickle

import pandas as pd
from mpi4py import MPI
import numpy as np
from adpgCirc_parallel import *

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 4 python mpi_adpgCirc.py
"""

name = "ADpgCirc_vCC"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

## Define param combinations
# Common simulation requirements
maxHe_vals = np.arange(0, 2, 0.02)
mCie_vals = np.arange(0, 30, 0.25)
mCee_vals = np.arange(0, 100, 1)
maxTAU2SC_vals = np.arange(0, 0.9, 0.009)
# HAdamrate_vals = np.arange(0, 50, 0.5)
# rho_vals = np.logspace(-1, 3, 50)
coupling_vals = np.arange(0, 120, 1)
speed_vals = np.arange(0.5, 25, 1)

        # g,   s, maxHe, mCie, mCee, maxTAU2SC, rho, HAdamrate, ABinh, TAUinh
params = [[25, 20, maxHe, 20.5, 80, 0.3, 50, 5, 0.4, 1.8] for maxHe in maxHe_vals] + \
         [[25, 20, 0.35, mCie, 80, 0.3, 50, 5, 0.4, 1.8] for mCie in mCie_vals] + \
         [[25, 20, 0.35, mCie, 80, 0.3, 50, 5, 0, 1.8] for mCie in mCie_vals] + \
         [[25, 20, 0.35, mCie, 80, 0.3, 50, 5, 0.4, 0] for mCie in mCie_vals] + \
         [[25, 20, 0.35, 20.5, mCee, 0.3, 50, 5, 0.4, 1.8] for mCee in mCee_vals] + \
         [[25, 20, 0.35, 20.5, 80, maxTAU2SC, 100, 5, 0.4, 1.8] for maxTAU2SC in maxTAU2SC_vals] + \
         [[g, 20, 0.35, 20.5, 80, 0.3, 50, 5, 0.4, 1.8] for g in coupling_vals] + \
         [[25, s, 0.35, 20.5, 80, 0.3, 50, 5, 0.4, 1.8] for s in speed_vals]

# TAU2SC_vals = np.logspace(-8, 1, 100)  # orig = 5e-2
# params = [[0.35, 20.5, 75, 0.8, 100, 5, 0.4, 1.8, TAU2SC] for TAU2SC in TAU2SC_vals] + \
#          [[0.35, 20.5, 75, 0.5, 100, 5, 0.4, 1.8, TAU2SC] for TAU2SC in TAU2SC_vals] + \
#          [[0.35, 20.5, 75, 0.3, 100, 5, 0.4, 1.8, TAU2SC] for TAU2SC in TAU2SC_vals]

# coupling_vals = np.arange(0, 120, 1)
# speed_vals = np.arange(0.5, 25, 1)
# params = [[g, 3.9, 0.35, 20.5, 75, 0.3, 100, 5, 0.4, 1.8] for g in coupling_vals] + \
#          [[50, s, 0.35, 20.5, 75, 0.3, 100, 5, 0.4, 1.8] for s in speed_vals]

# HAdamrate_vals = np.logspace(-1, 3, 100)
# params = [[25, 20, 0.35, 20.5, 75, 0.3, 50, HAdam, 0.4, 1.8] for HAdam in HAdamrate_vals]

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

local_results = adpgCirc_parallel(local_params)  # run the function for each parameter set and rank

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

    fResults_df = pd.DataFrame(final_results, columns=["g", "s", "maxHe", "minCie", "minCee", "maxTAU2SC", "rho",
                                                       "HAdamrate", "cABinh", "cTAUinh",
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
