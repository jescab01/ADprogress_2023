
import time
import numpy as np
import pandas as pd
from mpi4py import MPI

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 2 python FrequencyChartsMPI.py
"""

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
    ctb_folderOLD = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD\\"

    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTplot, FFTpeaks

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    ctb_folder = wd + "CTB_data3/"
    ctb_folderOLD = wd + "CTB_dataOLD/"

    import sys
    sys.path.append(wd)
    from toolbox.fft import FFTplot, FFTpeaks


## 0. DEFINE JOB and prepare PARALELIZATION
name = "FreqCharts4.0_wNoise"
emp_subj = "NEMOS_035"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

modes = [mode1 + "&" + mode2 for mode1 in ["classical"] for mode2 in ["fixed"]]
n_rep = 3


# 3d PSE
# He_vals = np.arange(1.5, 10, 0.25)
# Hi_vals = np.arange(5, 35, 0.25)
# p_vals = np.linspace(0, 2, 60)
# Hset = [[mode, r, 0.22*p, He, Hi, 10, 20, 108, 33.75, "exp_pHeHi"] for p in p_vals for He in He_vals for Hi in Hi_vals for mode in modes for r in range(n_rep)]


# He_vals = np.arange(1.5, 10, 0.25)
# Hi_vals = np.arange(5, 35, 0.25)
# Hset = [[mode, r, 0.22, He, Hi, 10, 20, 108, 33.75, "exp_HeHi"] for He in He_vals for Hi in Hi_vals for mode in modes for r in range(n_rep)]
#
# He_vals = np.arange(1.5, 10, 0.25)
# p_vals = np.linspace(0, 2, 60)
# pHeset = [[mode, r, 0.22*p, He, 22, 10, 20, 108, 33.75, "exp_pHe"] for He in He_vals for p in p_vals for mode in modes for r in range(n_rep)]
#
# Hi_vals = np.arange(5, 35, 0.25)
# p_vals = np.linspace(0, 2, 60)
# pHiset = [[mode, r, 0.22*p, 3.25, Hi, 10, 20, 108, 33.75, "exp_pHi"] for Hi in Hi_vals for p in p_vals for mode in modes for r in range(n_rep)]


# taue_vals = np.arange(2, 40, 0.5)
# taui_vals = np.arange(2, 40, 0.5)
# tauset = [[mode, r, 0.22, 3.25, 22, taue, taui, 108, 33.75, "exp_tau"] for taue in taue_vals for taui in taui_vals for mode in modes for r in range(n_rep)]
#
# Cee_vals = np.arange(25, 150, 1)
# Cie_vals = np.arange(5, 60, 0.5)
# Cset = [[mode, r, 3.25, 22, 10, 20, Cee, Cie, "exp_C"] for Cee in Cee_vals for Cie in Cie_vals for mode in modes for r in range(n_rep)]

He_vals = np.arange(1.5, 10, 0.25)
Cie_vals = np.arange(5, 60, 0.5)
set1 = [[mode, r, 0.22, He, 22, 10, 20, 108, Cie, "exp1"] for He in He_vals for Cie in Cie_vals for mode in modes for r in range(n_rep)]

Cee_vals = np.linspace(0, 2, 60)
p_vals = np.linspace(0, 2, 60)
set2 = [[mode, r, 0.22*p, 3.25, 22, 10, 20, 108*CC, 33.75, "exp2"] for p in p_vals for CC in Cee_vals for mode in modes for r in range(n_rep)]

Cie_vals = np.linspace(0, 2, 60)
p_vals = np.linspace(0, 2, 60)
set3 = [[mode, r, 0.22*p, 3.25, 22, 10, 20, 108, 33.75*CC, "exp3"] for p in p_vals for CC in Cie_vals for mode in modes for r in range(n_rep)]

CC_vals = np.linspace(0, 2, 60)
set4 = [[mode, r, 0.22, 3.25, 22, 10, 20, 108*CC1, 33.75*CC2, "exp4"] for CC2 in CC_vals for CC1 in CC_vals for mode in modes for r in range(n_rep)]
        # mode, reps, He, Hi, taue, taui, Cee, Cie

params = set1 + set2 + set3 + set4

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


## 1. COMMON SIMULATION SETUP

tic0 = time.time()

samplingFreq = 1000  #Hz
simLength = 20000  # ms - relatively long simulation to be able to check for power distribution
transient = 12000  # seconds to exclude from timeseries due to initial transient

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.EulerDeterministic(dt=1000/samplingFreq)

conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
conn.weights = conn.scaled_weights(mode="tract")

# Subset of 2 nodes is enough
conn.weights = conn.weights[:2][:, :2]
conn.tract_lengths = conn.tract_lengths[:2][:, :2]
conn.region_labels = conn.region_labels[:2]

mon = (monitors.Raw(),)


def computeFrequencyCharts(params_):

    result = []
    for ii, set in enumerate(params_):

        tic = time.time()
        print("Rank %i out of %i  ::  %i/%i " % (rank, size, ii + 1, len(params_)))

        print(set)
        mode, r, p, He, Hi, taue, taui, Cee, Cie, exploring = set

        m = JansenRit1995(He=np.array([He]), Hi=np.array([Hi]),
                          tau_e=np.array([taue]), tau_i=np.array([taui]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([Cee]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([Cie]),
                          p=np.array([p]), sigma=np.array([0.022]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

        # Coupling function
        coup = coupling.SigmoidalJansenRit(a=np.array([0]), cmax=np.array([0.005]),  midpoint=np.array([6]), r=np.array([0.56]))


        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup,  integrator=integrator, monitors=mon)
        sim.configure()

        output = sim.run(simulation_length=simLength)
        print("Simulation time: %0.2f / %0.2f sec" % (time.time() - tic, time.time() - tic0,))
        # Extract data cutting initial transient; PSP in pyramidal cells as exc_input - inh_input
        pspPyr = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T  # PSPs activity as recorded in MEEG
        ratePyr = m.e0 / (1 + np.exp(m.r * (m.v0 - (pspPyr))))  # Firing rate in pyramidal cells

        raw_time = output[0][0][transient:]
        regionLabels = conn.region_labels

        peaks, _, band_modules, ffts_abs, freqs, ffts_norm = \
            FFTpeaks(pspPyr, simLength, transient, samplingFreq, curves=True, freq_range=[1, 30], norm=True)


        result.append([mode, r, p, m.He[0], m.Hi[0], m.tau_e[0], m.tau_i[0], Cee, Cie, exploring] +
                      list(peaks) + list(band_modules) +
                      [np.average(pspPyr[0]), np.average(ratePyr[0]),
                       sum(ffts_norm[0, (12 < freqs)]) / sum(ffts_norm[0]),
                       sum(ffts_norm[0, (8 < freqs) & (freqs < 12)]) / sum(ffts_norm[0]),
                       sum(ffts_norm[0, (4 < freqs) & (freqs < 8)]) / sum(ffts_norm[0]),
                       sum(ffts_norm[0, (2 < freqs) & (freqs < 4)]) / sum(ffts_norm[0])])

    return np.asarray(result, dtype=object)


local_results = computeFrequencyCharts(local_params)  # run the function for each parameter set and rank


if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0

else:  ## MASTER PROCESS _receive, merge and save results
    final_results = np.copy(local_results)  # initialize final results with results from process 0
    for i in range(1, size):  # determine the size of the array to be received from each process

        # if i < remainder:
        #     rank_size = count + 1
        # else:
        #     rank_size = count
        # tmp = np.empty((rank_size, final_results.shape[1]))  # create empty array to receive results

        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if tmp is not None:  # Sometimes temp is a Nonetype wo/ apparent cause
            # print(final_results.shape)
            # print(tmp.shape)  # debugging
            # print(i)

            final_results = np.vstack((final_results, tmp))  # add the received results to the final results


    fResults_df = pd.DataFrame(final_results, columns=["mode", "rep", "p", "He", "Hi", "taue", "taui", "Cee", "Cie", "exp",
                                                       "roi1_Hz", "roi2_Hz", "roi1_auc", "roi2_auc", "roi1_meanS", "roi1_meanFR",
                                                       "roi1_aucBeta","roi1_aucAlpha","roi1_aucTheta","roi1_aucDelta"])

    ## Save resutls
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
