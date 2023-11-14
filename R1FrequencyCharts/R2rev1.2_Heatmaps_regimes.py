
import time
import numpy as np

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRit1995


## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    # from toolbox.fft import multitapper, FFTpeaks, PSDplot, PSD
    # from toolbox.fc import PLV
    # from toolbox.signals import epochingTool
    from toolbox.mixes import timeseries_spectra

data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"





# SIMULATION PARAMETERS   ###
subj, g, s, sigma, p = "HC-fam", 0, 20, 0.022, 0.22
simLength, transient, samplingFreq = 6 * 1000, 2 * 1000, 1000  # ms, ms, Hz


regimesCip = [("Slow limit cycle - Theta oscillation", 40),
              ("Fast limit cycle - Alpha oscillation", 33.75),
              ("Fixed point - Noisy low power oscillation", 20)]





# STRUCTURAL CONNECTIVITY      #########################################
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
conn.weights = conn.scaled_weights(mode="tract")

## TO USE JUST CB
SC_dmn_idx = [0, 1]
conn.weights = conn.weights[:, SC_dmn_idx][SC_dmn_idx]
conn.tract_lengths = conn.tract_lengths[:, SC_dmn_idx][SC_dmn_idx]
conn.region_labels = conn.region_labels[SC_dmn_idx]



# COUPLING FUNCTION   #########################################
coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]), r=np.array([0.56]))
conn.speed = np.array([s])

#   MORE   ######
# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(Hz)
integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([0])))
# integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)


for title, Cip in regimesCip:

    # NEURAL MASS MODEL    #########################################################
    m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([Cip]),
                          p=np.array([p]), sigma=np.array([sigma]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

    print("Simulating ...")

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where:
    # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    pspPyr = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T
    raw_time = output[0][0][transient:]
    regionLabels = conn.region_labels

    timeseries_spectra(pspPyr, simLength, transient, regionLabels,  folder="figures/", title=title, mode="html", width=1000, height=300)
    timeseries_spectra(pspPyr, simLength, transient, regionLabels, folder="figures/", title=title, mode="svg", width=1000, height=300)


