
import os
import time
import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats
from mne import filter
import pingouin as pg
import statsmodels.api as sm
from itertools import combinations
import pickle

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline

from tvb.simulator.lab import *
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from tvb.simulator.models.JansenRit_WilsonCowan import JansenRit_WilsonCowan

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys

    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTpeaks
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order
    from toolbox.littlebrains import addpial

## Folder structure - CLUSTER
elif "t192" in os.getcwd():
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"
    import sys

    sys.path.append(wd)
    from toolbox.fft import FFTpeaks
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order
    from toolbox.littlebrains import addpial

## Folder structure - CLUSTER BRIGIT
else:
    wd = "/mnt/lustre/home/jescab01/"
    data_folder = wd + "ADprogress_data/"
    import sys

    sys.path.append(wd)
    from toolbox.fft import FFTpeaks
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order
    from toolbox.littlebrains import addpial



class CircularADpgModel_vCC:
    # Spread model variables following Alexandersen 2022.
    # M is an arbitrary unit of concentration

    def __init__(self, initConn, AB_initMap, TAU_initMAp, ABt_initMap, TAUt_initMap,
                 AB_initdam, TAU_initdam, HA_initdam,
                 init_He, init_Cee, init_Cie, rho=0.001, toxicSynergy=2,
                 prodAB=2, clearAB=2, transAB2t=2, clearABt=1.5,
                 prodTAU=2, clearTAU=2, transTAU2t=2, clearTAUt=2.66,
                 AB_damrate=1, TAU_damrate=1, TAU_dam2SC=0.2, HA_damrate=1, maxHAdam=2, maxTAU2SCdam=0.2,
                 cABexc=0.8, cABinh=0.4, cTAUexc=1.8, cTAUinh=1.8):

        self.rho = {"label": "rho", "value": np.array([rho]),
                    "doc": "effective diffusion constant (cm/year)"}

        self.prodAB = {"label": ["k0", "a0"], "value": np.array([prodAB]), "doc": "production rate for a-beta (M/year)"}
        self.clearAB = {"label": ["k1", "a1"], "value": np.array([clearAB]),
                        "doc": "clearance rate for a-beta (1/M*year)"}
        self.transAB2t = {"label": ["k2", "a2"], "value": np.array([transAB2t]),
                          "doc": "transformation of a-beta into its toxic variant (M/year)"}
        self.clearABt = {"label": ["k1t", "a1t"], "value": np.array([clearABt]),
                         "doc": "clearance rate for toxic a-beta (1/M*year)"}

        self.prodTAU = {"label": ["k3", "b0"], "value": np.array([prodTAU]),
                        "doc": "production rate for p-tau (M/year)"}
        self.clearTAU = {"label": ["k4", "b1"], "value": np.array([clearTAU]),
                         "doc": "clearance rate for p-tau (1/M*year)"}
        self.transTAU2t = {"label": ["k5", "b2"], "value": np.array([transTAU2t]),
                           "doc": "transformation of p-tau into its toxic variant (M/year)"} \
            if len(np.array([transTAU2t]).shape) == 1 else \
            {"label": ["k5", "b2"], "value": np.array([transTAU2t]).squeeze(),
             "doc": "transformation of p-tau into its toxic variant (M/year)"}

        self.clearTAUt = {"label": ["k4t", "b1t"], "value": np.array([clearTAUt]),
                          "doc": "clearance rate for toxic p-tau (1/M*year)"}

        self.toxicSynergy = {"label": ["k6", "b3"], "value": np.array([toxicSynergy]),
                             "doc": "synergistic effect between toxic a-beta and toxic p-tau production (1/M^2*year)"} \
            if len(np.array([toxicSynergy]).shape) == 1 else \
            {"label": ["k5", "b2"], "value": np.array([toxicSynergy]).squeeze(),
             "doc": "transformation of p-tau into its toxic variant (M/year)"}

        self.AB_initMap = {"label": "", "value": AB_initMap, "doc": "mapping of initial roi concentration of AB"}
        self.TAU_initMap = {"label": "", "value": TAU_initMAp, "doc": "mapping of initial roi concentration of TAU"}

        self.ABt_initMap = {"label": "", "value": ABt_initMap,
                            "doc": "mapping of initial roi concentration of AB toxic"}
        self.TAUt_initMap = {"label": "", "value": TAUt_initMap,
                             "doc": "mapping of initial roi concentration of TAU toxic"}

        AB_initdam = AB_initdam if type(AB_initdam) == list else [AB_initdam for roi in initConn.region_labels]
        self.AB_initdam = {"label": "q(AB)", "value": AB_initdam, "doc": "initial damage/impact variable of AB"}
        self.AB_damrate = {"label": "k(AB)", "value": np.array([AB_damrate]),
                           "doc": "rate of damage/impact for AB (M/year)"}

        TAU_initdam = TAU_initdam if type(TAU_initdam) == list else [TAU_initdam for roi in initConn.region_labels]
        self.TAU_initdam = {"label": "q(TAU)", "value": TAU_initdam,
                            "doc": "initial damage/impact of hyperphosphorilated TAU"}
        self.TAU_damrate = {"label": "K(TAU)", "value": np.array([TAU_damrate]),
                            "doc": "rate of damage/impact for hyperphosphorilated TAU (M/year)"}
        self.TAU_dam2SC = {"label": "gamma", "value": np.array([TAU_dam2SC]),
                           "doc": "constant for the damage of structural connectivity by hpTAU (cm/year)"}
        self.maxTAU2SCdam = {"label": "gamma", "value": np.array([maxTAU2SCdam]),
                           "doc": "maximum damage of structural connectivity by hpTAU (cm/year)"}

        HA_initdam = HA_initdam if type(HA_initdam) == list else [HA_initdam for roi in initConn.region_labels]
        self.HA_initdam = {"label": "q(POW)", "value": HA_initdam, "doc": "initial damage/impact variable of POWER"}
        self.maxHAdam = {"label": "q(POW)", "value": np.array([maxHAdam]),
                          "doc": "max damage/impact variable of POWER"}
        self.HA_damrate = {"label": "", "value": np.array([HA_damrate]),
                            "doc": "rate of damage/impact for AB (M/year)"}

        self.initConn = {"label": "SC", "value": initConn, "orig_weights":initConn.weights.copy(), "doc": "Initial state for structural connectivity"}


        init_He = init_He if type(init_He) == list else [init_He for roi in initConn.region_labels]
        self.init_He = {"label": "He", "value": init_He, "range": [2.6, 9.75],
                        "doc": "Initial state for excitatory PSP amplitud. def3.25"}
        init_Cee = init_Cee if type(init_Cee) == list else [init_Cee for roi in initConn.region_labels]
        self.init_Cee = {"label": "Cee", "value": init_Cee, "range": [54, 162],
                        "doc": "Initial state for average synaptic contacts between exc interneurons and pyramidals. def108"}
        init_Cie = init_Cie if type(init_Cie) == list else [init_Cie for roi in initConn.region_labels]
        self.init_Cie = {"label": "Hi", "value": init_Cie, "range": [15, 50],
                        "doc": "Initial state for average synaptic contacts between inh interneurons and pyramidals. def33.75"}

        self.cABexc = {"label": "c_beta", "value": np.array([cABexc]),
                       "doc": "constant for the effect of AB on excitation"}
        self.cABinh = {"label": "c_beta2", "value": np.array([cABinh]),
                       "doc": "constant for the effect of AB on inhibition"}
        self.cTAUexc = {"label": "c_tau", "value": np.array([cTAUexc]),
                     "doc": "constant for the effect of pTau on delays"}
        self.cTAUinh = {"label": "c_tau", "value": np.array([cTAUinh]),
                     "doc": "constant for the effect of pTau on delays"}

    def run(self, time, dt, sim=False, sim_dt=1):

        ## 1. Initiate state variables
        state_variables = np.asarray([self.AB_initMap["value"],
                                      self.ABt_initMap["value"],
                                      self.TAU_initMap["value"],
                                      self.TAUt_initMap["value"],

                                      self.AB_initdam["value"],
                                      self.TAU_initdam["value"],

                                      self.init_He["value"],
                                      self.init_Cee["value"],
                                      self.init_Cie["value"],

                                      self.HA_initdam["value"]])

        weights = self.initConn["value"].weights

        evolution_sv = [state_variables.copy()]

        print("Simulating protein spread  . for %0.2fts (dt=%0.2f)   _simulate: %s" % (time, dt, sim))

        if (type(sim_dt) == int) | (type(sim_dt) == float):
            tsel = np.arange(0, time, sim_dt)
        else:
            tsel = sim_dt

        if (sim) and (0 in tsel):
            subj, g, s, sigma, simLength, transient = sim
            pspPyr, raw_time, ratePyr, spectra, plv_sim, plv_r, reqtime = \
                simulate_v3(subj, self.initConn["value"], weights, g, s, sigma=sigma, sv=state_variables[6:-1],
                            t=simLength, trans=transient)

            baseline_activity = np.average(ratePyr, axis=1)  # spectra[1]
            evolution_net = [[weights, spectra, plv_sim, np.average(ratePyr, axis=1), pspPyr]]
            print("   . ts%0.2f/%0.2f  _  SIMULATION REQUIRED %0.2f seconds  -  rPLV(%0.2f)" % (
                0, time, reqtime, plv_r))

        else:
            evolution_net = [[weights]]
            print("   . ts%0.2f/%0.2f" % (0, time), end="\r")

        ## 2. loop over time
        for t in np.arange(dt, time, dt):

            ## HyperActivity damage (attracting TAUt and generating more AB)
            dActivity = np.average(ratePyr, axis=1) - baseline_activity  # spectra[1] / baseline_power

            deriv = self.dfun(state_variables, self.Laplacian(weights), dActivity)

            state_variables = state_variables + dt * deriv

            ## Update weights by damage function
            TAUdam = state_variables[5]
            dWeights = - self.TAU_dam2SC["value"] * \
                       (np.tile(TAUdam, (len(TAUdam), 1)).transpose() + np.tile(TAUdam, (len(TAUdam), 1))) * \
                       (weights - self.initConn["orig_weights"] * (1 - self.maxTAU2SCdam["value"]))
                        ## Current weights - 70% of the initial weights

            weights = weights + dWeights
            weights[weights < 0] = 0  # weights cannot be negative

            if sim and (t in tsel):
                subj, g, s, sigma, simLength, transient = sim
                pspPyr, raw_time, ratePyr, spectra, plv_sim, plv_r, reqtime = \
                    simulate_v3(subj, self.initConn["value"], weights, g, s, sigma=sigma, sv=state_variables[6:-1],
                                t=simLength, trans=transient)

                evolution_net.append([weights, spectra, plv_sim, np.average(ratePyr, axis=1), pspPyr])
                evolution_sv.append(state_variables)
                print("   . ts%0.2f/%0.2f  _  SIMULATION REQUIRED %0.2f seconds  -  rPLV(%0.2f)" % (
                    t, time, reqtime, plv_r))
            else:
                evolution_sv.append(state_variables.copy())
                evolution_net.append([weights])
                print("   . ts%0.2f/%0.2f" % (t, time), end="\r")

        return [np.arange(0, time, dt), evolution_sv, evolution_net]

    def Laplacian(self, weights):
        # Weighted adjacency, Diagonal and Laplacian matrices
        Wij = np.divide(weights, np.square(self.initConn["value"].tract_lengths),
                        where=np.square(self.initConn["value"].tract_lengths) != 0,
                        # Where to compute division; else out
                        out=np.zeros_like(weights))  # array allocation
        Dii = np.eye(len(Wij)) * np.sum(Wij, axis=0)
        Lij = (Dii - Wij)

        return Lij

    def dfun(self, state_variables, Lij, dHA):
        # Here we want to model the spread of proteinopathies.
        # Approach without activity dependent spread/generation. Following Alexandersen 2022.

        AB = state_variables[0]
        ABt = state_variables[1]
        TAU = state_variables[2]
        TAUt = state_variables[3]

        ABdam = state_variables[4]
        TAUdam = state_variables[5]

        He_ = state_variables[6]
        Cee_ = state_variables[7]
        Cie_ = state_variables[8]

        HAdam = state_variables[-1]

        # Unpack heterogeneous rho
        [rho_AB, rho_ABt, rho_TAU, rho_TAUt] = self.rho["value"][0] \
            if len(self.rho["value"].shape) == 2 else self.rho["value"].repeat(4)

        # Derivatives
        ###  Amyloid-beta
        dAB = -rho_AB * np.sum(Lij * AB, axis=1) + self.prodAB["value"] * (1 + HAdam) - self.clearAB["value"] * AB - \
              self.transAB2t["value"] * AB * ABt
        dABt = -rho_ABt * np.sum(Lij * ABt, axis=1) - self.clearABt["value"] * ABt + self.transAB2t[
            "value"] * AB * ABt

        ###  (hyperphosphorilated) Tau
        dTAU = -rho_TAU * np.sum(Lij * TAU, axis=1) + self.prodTAU["value"] - self.clearTAU["value"] * TAU - \
               self.transTAU2t["value"] * TAU * TAUt - self.toxicSynergy["value"] * ABt * TAU * TAUt
        dTAUt = -rho_TAUt * np.sum((Lij * (1 + HAdam)).transpose() * TAUt, axis=1) - self.clearTAUt["value"] * TAUt + \
                self.transTAU2t["value"] * TAU * TAUt + self.toxicSynergy["value"] * ABt * TAU * TAUt

        dABdam = self.AB_damrate["value"] * ABt * (1 - ABdam)
        dTAUdam = self.TAU_damrate["value"] * TAUt * (1 - TAUdam)

        ## Hyperactivity impact
        dHAdam = self.HA_damrate["value"] * dHA  # * (self.maxHAdam["value"] - HAdam)

        ## (He) PSP amplitude transfer - Impact on Glutamate reuptake
        dHe = self.cABexc["value"] * ABdam * (self.init_He["range"][1] - He_)

        ## INTRA-CONNECTIVITY transfers: a(exc), b(inh)
        dCee = - self.cTAUexc["value"] * TAUdam * (Cee_ - self.init_Cee["range"][0])

        dCie = - self.cABinh["value"] * ABdam * (Cie_ - self.init_Cie["range"][0]) \
               - self.cTAUinh["value"] * TAUdam * (Cie_ - self.init_Cie["range"][0])

        derivative = np.array([dAB, dABt, dTAU, dTAUt, dABdam, dTAUdam, dHe, dCee, dCie, dHAdam])

        return derivative


def simulate_v3(subj, conn, weights, g, s, sigma=0.022, sv=None, t=10, trans=1):

    # Prepare simulation parameters
    simLength = t * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = trans * 1000  # ms

    tic = time.time()

    # STRUCTURAL CONNECTIVITY      #########################################
    conn.weights = weights
    conn.speed = np.array([s])

    #  Load FC labels, transform to SC format; check if match SC.
    FClabs = list(np.loadtxt(data_folder + "FCavg_matrices/" + subj + "_roi_labels.txt", dtype=str))
    FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    FC_cortex_idx = [FClabs.index(roi) for roi in conn.region_labels[conn.cortical]]  # find indexes in FClabs that matches cortical_rois

    # load SC labels.
    SClabs = list(conn.region_labels)
    SC_cortex_idx = [SClabs.index(roi) for roi in conn.region_labels[conn.cortical]]

    #   NEURAL MASS MODEL  &  COUPLING FUNCTION   #########################################################
    ## For CircularModel_vH.
    if len(sv) == 2:
        m = JansenRit1995(He=np.array(sv[0]), Hi=np.array(sv[1]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([0.1085]), sigma=np.array([sigma]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

    ## For CircularModel_vCC.
    elif len(sv) == 3:
        # Parameters from Stefanovski 2019.
        m = JansenRit1995(He=np.array(sv[0]), Hi=np.array([22]),
                          tau_e=np.array([10]), tau_i=np.array([20]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array(sv[1]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array(sv[2]),
                          p=np.array([0.1085]), sigma=np.array([sigma]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

    ## For original model with taus
    else:
        # Parameters from Stefanovski 2019.
        m = JansenRit1995(He=np.array(sv[0]), Hi=np.array(sv[1]),
                          tau_e=np.array(sv[2]), tau_i=np.array([sv[3]]),
                          c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                          c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                          p=np.array([0.1085]), sigma=np.array([sigma]),
                          e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))

    # OTHER PARAMETERS   ###
    # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
    # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
    integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

    mon = (monitors.Raw(),)

    # print("Simulating %s (%is)  ||  PARAMS: g%i sigma%0.2f" % (model, simLength / 1000, g, 0.022))

    # Run simulation
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
    sim.configure()
    output = sim.run(simulation_length=simLength)

    # Extract data: "output[a][b][:,0,:,0].T" where: a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
    raw_time = output[0][0][transient:]
    # Getting postsynaptic potential activity (as it is measured by MEEG):
    # https://psychology.nottingham.ac.uk/staff/mxs/MScCognNeurosciNeuroimaging/handouts/n_C84EBM_1_meg2008b.pdf
    pspPyr = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T  # PSPs activity as recorded in MEEG
    ratePyr = m.e0 / (1 + np.exp(m.r * (m.v0 - (pspPyr))))  # Firing rate in pyramidal cells

    spectra = FFTpeaks(pspPyr, simLength, transient, samplingFreq, curves=True, freq_range=[1, 30])


    # PLOTs :: Signals and spectra
    # timeseries_spectra(raw_data[:], simLength, transient, regionLabels, mode="inline", freqRange=[2, 40], opacity=1)

    bands = [["3-alpha"], [(8, 12)]]
    # bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

    for b in range(len(bands[0])):
        (lowcut, highcut) = bands[1][b]

        # Band-pass filtering
        filterSignals = filter.filter_data(pspPyr, samplingFreq, lowcut, highcut, verbose=False)

        # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
        efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals", verbose=False)

        # Obtain Analytical signal
        efPhase = list()
        efEnvelope = list()
        for i in range(len(efSignals)):
            analyticalSignal = scipy.signal.hilbert(efSignals[i])
            # Get instantaneous phase and amplitude envelope by channel
            efPhase.append(np.angle(analyticalSignal))
            efEnvelope.append(np.abs(analyticalSignal))

        # Check point
        # from toolbox import timeseriesPlot, plotConversions
        # regionLabels = conn.region_labels
        # timeseriesPlot(raw_data, raw_time, regionLabels)
        # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

        # CONNECTIVITY MEASURES
        ## PLV and plot
        plv_sim = PLV(efPhase, verbose=False)
        plv_sim_cx = plv_sim[:, SC_cortex_idx][SC_cortex_idx]
        # Load empirical data to make simple comparisons
        plv_emp = \
            np.loadtxt(data_folder + "FCavg_matrices/" + subj + "_" + bands[0][b] + "_plv_avg.txt", delimiter=',')[:,
            FC_cortex_idx][
                FC_cortex_idx]

        # Comparisons
        t1 = np.zeros(shape=(2, len(plv_emp) ** 2 // 2 - len(plv_emp) // 2))
        t1[0, :] = plv_sim_cx[np.triu_indices(len(plv_emp), 1)]
        t1[1, :] = plv_emp[np.triu_indices(len(plv_emp), 1)]
        plv_r = np.corrcoef(t1)[0, 1]

    return pspPyr, raw_time, ratePyr, spectra, plv_sim, plv_r, time.time() - tic


def circApproach(out_circ, conn, title, surrogates=None, folder="figures"):

    fig = make_subplots(rows=3, cols=4, horizontal_spacing=0.1,
                        specs=[[{}, {}, {}, {}],
                               [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {}],
                               [{}, {"secondary_y": True}, {"type": "surface"}, {}]])

    # 1. HEATMAPS visualization
    print("1. HEATMAPS")
    fig = heatmaps_viz(fig, out_circ, pos=[(1, 1), (1, 3), (1, 2), (1, 4)], mode=title)

    # 2. MOLECULAR & PARAMs visualization
    print("2. MOLECULAR and PARAMS")
    fig = molecular_viz(fig, out_circ, conn, pos=[(2, 1), (3, 1)], mode=title)

    # 3. SPECTRAL visualization
    print("3. SPECTRAL")
    fig, abs_ffts, norm_ffts = spectral_viz(fig, out_circ, pos=[(2, 2), (3, 2)])

    # 4. FC & RATE visualization
    print("4. FC & RATE")
    fig, edges_color, nodes_color, hovertext3d = FCplusrate_viz(fig, out_circ, conn, pos=[(2, 3), (3, 3)], surrogates=surrogates)

    # 5. FRAMES
    print("5. FRAMES")
    shortout_0 = [simpack for i, simpack in enumerate(out_circ[0]) if len(out_circ[2][i]) > 1]
    shortout_1 = [simpack for i, simpack in enumerate(out_circ[1]) if len(out_circ[2][i]) > 1]

    # add references
    fig.add_trace(go.Scatter(x=[shortout_0[0], shortout_0[0]], y=[-0.05, 0.5], mode="lines", legendgroup="timeref",
                   line=dict(color="black", width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[shortout_0[0], shortout_0[0]], y=[0, 12], mode="lines", legendgroup="timeref",
                   line=dict(color="black", width=1), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[shortout_0[0], shortout_0[0]], y=[-0.05, 1], mode="lines", legendgroup="timeref",
                   line=dict(color="black", width=1), showlegend=False), row=2, col=3)

    if "vCC" in title:
        fig.update(frames=[go.Frame(data=[

            go.Scatter(x=shortout_1[i][3], y=shortout_1[i][1]),

            go.Scatter(y=norm_ffts[i]),
            go.Scatter(y=abs_ffts[i]),

            go.Scatter3d(line=dict(color=edges_color[i])),
            go.Scatter3d(hovertext=hovertext3d[i], marker=dict(color=nodes_color[i])),

            go.Scatter(x=[shortout_0[i], shortout_0[i]]), # ref lines
            go.Scatter(x=[shortout_0[i], shortout_0[i]]),
            go.Scatter(x=[shortout_0[i], shortout_0[i]])],
            # TRACES: [0-11]Heatmaps(noVaría); [12-20]State vars; [21]ProtProp;
            # [22-28]Spectra up & down; [29-32]Rate and FC; [33-35] Brain3d; [36-38] references.
            traces=[21,   27, 28,   33, 34,  36, 37, 38], name=str(t)) for i, t in enumerate(shortout_0)])

    elif "vH" in title:
        fig.update(frames=[go.Frame(data=[

            go.Scatter(x=shortout_1[i][3], y=shortout_1[i][1]),

            go.Scatter(y=norm_ffts[i]),
            go.Scatter(y=abs_ffts[i]),

            go.Scatter3d(line=dict(color=edges_color[i])),
            go.Scatter3d(hovertext=hovertext3d[i], marker=dict(color=nodes_color[i])),

            go.Scatter(x=[shortout_0[i], shortout_0[i]]),  # ref lines
            go.Scatter(x=[shortout_0[i], shortout_0[i]]),
            go.Scatter(x=[shortout_0[i], shortout_0[i]])],
            # TRACES: [0-11]Heatmaps; [12-19]State vars; [20]ProtProp;
            # [21-27]Spectra up & down; [28-31]Rate and FC; [32-34] Brain3d; [35-37] references.
            traces=[20,   26, 27,    32, 33,   35, 36, 37], name=str(t)) for i, t in enumerate(shortout_0)])

    # CONTROLS : Add sliders and buttons
    fig.update_layout(
        template="plotly_white",
        legend=dict(yanchor="bottom", y=-0.1, x=0.775, tracegroupgap=0, groupclick="toggleitem"),

        # Heatmaps
        xaxis1=dict(title="He"), xaxis3=dict(title="He"), xaxis2=dict(title="p (input)"),
        xaxis4=dict(title="p (input)"),
        yaxis1=dict(title="Cie"), yaxis3=dict(title="Cie"), yaxis2=dict(title="Cee"), yaxis4=dict(title="Cee"),
        coloraxis1=dict(colorbar_title="Hz", colorbar_x=0.42, colorbar_y=0.86, colorbar_len=0.28, colorbar_thickness=7, colorscale="Cividis"),
        coloraxis2=dict(colorbar_title="Hz", colorbar_x=0.94, colorbar_y=0.86, colorbar_len=0.28, colorbar_thickness=7),

        yaxis5=dict(title="State variables"),
        yaxis8=dict(title="Relative power"), yaxis7=dict(range=[0, 14], title="Frequency (Hz)"),
        yaxis9=dict(title="Averaged FC", range=[0, 1]), yaxis10=dict(title="Firing rate"),
        yaxis12=dict(title="ABt", range=[-0.05, 0.4]), yaxis13=dict(title="Norm. power", range=[0, np.max(np.array(norm_ffts))]),
        yaxis14=dict(title="Abs. power", range=[0, np.max(np.array(abs_ffts))]),

        xaxis5=dict(title="Time (years)"), xaxis6=dict(title="Time (years)"), xaxis7=dict(title="Time (years)"),
        xaxis9=dict(title="TAUt", range=[-0.05, 0.4]), xaxis10=dict(title="Frequency (Hz)"),

        scene1=dict(xaxis=dict(title="Sagital axis<br>(R-L)"), yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                    zaxis=dict(title="Horizontal axis<br>(sup-inf)"), camera=dict(eye=dict(x=0.75, y=0.75, z=0.75))),
        sliders=[dict(
            steps=[
                dict(method='animate', args=[[str(t)], dict(mode="immediate", frame=dict(duration=250, redraw=True,
                                                                                         easing="cubic-in-out"),
                                                            transition=dict(duration=0))], label=str(t)) for i, t
                in enumerate(shortout_0)],
            transition=dict(duration=0), x=0.1, xanchor="left", y=-0.1,
            currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
            len=0.52, tickcolor="white")],
        updatemenus=[dict(type="buttons", showactive=False, y=-0.15, x=0, xanchor="left",
                          buttons=[
                              dict(label="Play", method="animate",
                                   args=[None,
                                         dict(frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                              transition=dict(duration=0),
                                              fromcurrent=True, mode='immediate')]),
                              dict(label="Pause", method="animate",
                                   args=[[None],
                                         dict(frame=dict(duration=250, redraw=False, easing="cubic-in-out"),
                                              transition=dict(duration=0),
                                              mode="immediate")])])])

    pio.write_html(fig, file=folder + "/circApproach_" + title + ".html", auto_open=True, auto_play=False)


def heatmaps_viz(fig, out_circ, pos, mode):
    """
    Plot constructor. Pass the data and it constucts certain plots in the desired positions.
    12 traces.
    :param fig:
    :param out_prop:
    :param pos: Figures position. 0-3 [Heatmaps], 4 [Prot. concentration; damage; params]
    :param mode:
    :return:
    """

    timepoints = out_circ[0]

    avg_weights = np.array([np.average(t_list[0]) for t_list in out_circ[2]])
    avg_weights = avg_weights / max(avg_weights) * 0.22

    if "circ" in mode:

        _, ABt, _, TAUt, ABdam, TAUdam, He, Hi, taue, taui, HAdam = np.average(out_circ[1], axis=2).transpose()

        ## 1. HEATMAPS
        main_folder = "E:\LCCN_Local\PycharmProjects\\brainModels\FrequencyChart\\data\\"
        simulations_tag = "PSEmpi_FreqCharts4.0-m02d01y2023-t13h.49m.33s"  # Tag cluster job
        # mode = "classical" if mode=="classic" else mode
        df = pd.read_csv(main_folder + simulations_tag + "/results.csv")

        df_avg = df.groupby(['mode', "p", 'He', 'Hi', 'taue', 'taui', 'Cee', 'Cie', 'exp']).mean().reset_index()
        cmax_freq, cmin_freq = max(df_avg["roi1_Hz"].values), min(df_avg["roi1_Hz"].values)
        cmax_pow, cmin_pow = max(df_avg["roi1_auc"].values), min(df_avg["roi1_auc"].values)
        cmax_fr, cmin_fr = max(df_avg["roi1_meanFR"].values), min(df_avg["roi1_meanFR"].values)

        Hchart_df = df.loc[(df["mode"] == "classical" + "&fixed") & (df["exp"] == "exp_H")]
        # 1.1 He-Hi POWER
        fig.add_trace(go.Heatmap(z=Hchart_df.roi1_meanFR, x=Hchart_df.He, y=Hchart_df.Hi, coloraxis="coloraxis1"), row=pos[0][0], col=pos[0][1])

        hovertext = ["   <b>t%0.2f</b><br>He = %0.2f  |  Hi = %0.2f<br>Firing rate (Hz) %0.4f"
                     % (timepoints[ii], He[ii], Hi[ii], Hchart_df["roi1_meanFR"].iloc[
            np.argsort(np.abs(Hchart_df["He"] - He[ii]) + np.abs(Hchart_df["Hi"] - Hi[ii])).values[0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=He, y=Hi, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext, name="HeHi-power",
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7), row=pos[0][0], col=pos[0][1])

        fig.add_trace(go.Scatter(x=[He[0]], y=[Hi[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext[0], name="HeHi-power",
                                 line=dict(color="red", width=4)), row=pos[0][0], col=pos[0][1])  # add initial point

        # 1.2 He-Hi FREQUENCY
        fig.add_trace(go.Heatmap(z=Hchart_df.roi1_Hz, x=Hchart_df.He, y=Hchart_df.Hi, coloraxis="coloraxis2"), row=pos[1][0], col=pos[1][1])

        hovertext = ["   <b>t%0.2f</b><br>He = %0.2f  |  Hi = %0.2f<br>Frequency (Hz) %0.4f"
                     % (timepoints[ii], He[ii], Hi[ii],
                        Hchart_df["roi1_Hz"].iloc[
                            np.argsort(np.abs(Hchart_df["He"] - He[ii]) + np.abs(Hchart_df["Hi"] - Hi[ii])).values[0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=He, y=Hi, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext, name="HeHi-frequency",
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7),
                      row=pos[1][0], col=pos[1][1])

        fig.add_trace(go.Scatter(x=[He[0]], y=[Hi[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext[0], name="HeHi-power",
                                 line=dict(color="red", width=4)),
                      row=pos[1][0], col=pos[1][1])  # add initial point


        tauchart_df = df.loc[(df["mode"] == "classical" + "&fixed") & (df["exp"] == "exp_tau")]
        # 1.3 Taue-Taui POWER
        fig.add_trace(go.Heatmap(z=tauchart_df.roi1_meanFR, x=tauchart_df.taue, y=tauchart_df.taui, coloraxis="coloraxis1"),
                      row=pos[2][0], col=pos[2][1])

        hovertext = ["   <b>t%0.2f</b><br>tau_e = %0.2f  |  tau_i = %0.2f<br>Firing rate (Hz) %0.4f"
                     % (timepoints[ii], taue[ii], taui[ii],
                        tauchart_df["roi1_meanFR"].iloc[
                            np.argsort(
                                np.abs(tauchart_df["taue"] - taue[ii]) + np.abs(tauchart_df["taui"] - taui[ii])).values[
                                0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=taue, y=taui, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext,
                                 line=dict(color=px.colors.sequential.BuPu[3], width=3), opacity=0.7),
                      row=pos[2][0], col=pos[2][1])

        fig.add_trace(go.Scatter(x=[taue[0]], y=[taui[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext[0],
                                 line=dict(color="red", width=4)),
                      row=pos[2][0], col=pos[2][1])


        # 1.4 Taue-Taui FREQUENCY
        fig.add_trace(go.Heatmap(z=tauchart_df.roi1_Hz, x=tauchart_df.taue, y=tauchart_df.taui, coloraxis="coloraxis2"),
                      row=pos[3][0], col=pos[3][1])

        hovertext = ["   <b>t%0.2f</b><br>tau_e = %0.2f  |  tau_i = %0.2f<br>Frequency (Hz) %0.4f"
                     % (timepoints[ii], taue[ii], taui[ii], tauchart_df["roi1_Hz"].iloc[
            np.argsort(np.abs(tauchart_df["taue"] - taue[ii]) + np.abs(tauchart_df["taui"] - taui[ii])).values[0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=taue, y=taui, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext,
                                 line=dict(color=px.colors.sequential.BuPu[3], width=3), opacity=0.7),
                      row=pos[3][0], col=pos[3][1])

        fig.add_trace(go.Scatter(x=[taue[0]], y=[taui[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext[0],
                                 line=dict(color="red", width=4)), row=pos[3][0], col=pos[3][1])

    elif "vCC" in mode:

        _, ABt, _, TAUt, ABdam, TAUdam, He, Cee, Cie, HAdam = np.average(out_circ[1], axis=2).transpose()

        ## 1. HEATMAPS
        main_folder = "E:\LCCN_Local\PycharmProjects\\brainModels\FrequencyChart\\data\\"
        simulations_tag = "PSEmpi_FreqCharts4.0-m02d01y2023-t13h.49m.33s"  # Tag cluster job
        # mode = "classical" if mode=="classic" else mode
        df = pd.read_csv(main_folder + simulations_tag + "/results.csv")

        df_avg = df.groupby(['mode', "p", 'He', 'Hi', 'taue', 'taui', 'Cee', 'Cie', 'exp']).mean().reset_index()
        cmax_freq, cmin_freq = max(df_avg["roi1_Hz"].values), min(df_avg["roi1_Hz"].values)
        cmax_pow, cmin_pow = max(df_avg["roi1_auc"].values), min(df_avg["roi1_auc"].values)
        cmax_fr, cmin_fr = max(df_avg["roi1_meanFR"].values), min(df_avg["roi1_meanFR"].values)


        chart1_df = df.loc[(df["mode"] == "classical" + "&fixed") & (df["exp"] == "exp_1")]
        ### 1.1 He - Cie RATE
        fig.add_trace(go.Heatmap(z=chart1_df.roi1_meanFR, x=chart1_df.He, y=chart1_df.Cie, coloraxis="coloraxis1"), row=pos[0][0], col=pos[0][1])

        hovertext1 = ["   <b>t%0.2f</b><br>He = %0.2f  |  Cie = %0.2f<br>Firing rate (Hz) %0.4f"
                     % (timepoints[ii], He[ii], Cie[ii],
                        chart1_df["roi1_meanFR"].iloc[np.argsort(np.abs(chart1_df["He"] - He[ii]) + np.abs(chart1_df["Cie"] - Cie[ii])).values[0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=He, y=Cie, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext1,
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7), row=pos[0][0], col=pos[0][1])

        fig.add_trace(go.Scatter(x=[He[0]], y=[Cie[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext1[0],
                                 line=dict(color="red", width=4)),
                      row=pos[0][0], col=pos[0][1])  # add initial point

        ### 1.2 He - Cie FREQUENCY
        fig.add_trace(go.Heatmap(z=chart1_df.roi1_Hz, x=chart1_df.He, y=chart1_df.Cie, coloraxis="coloraxis2"), row=pos[1][0], col=pos[1][1])

        hovertext2 = ["   <b>t%0.2f</b><br>He = %0.2f  |  Cie = %0.2f<br>Frequency (dB) %0.4f"
                     % (timepoints[ii], He[ii], Cie[ii],
                        chart1_df["roi1_Hz"].iloc[np.argsort(np.abs(chart1_df["He"] - He[ii]) + np.abs(chart1_df["Cie"] - Cie[ii])).values[0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=He, y=Cie, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext2,
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7), row=pos[1][0], col=pos[1][1])

        fig.add_trace(go.Scatter(x=[He[0]], y=[Cie[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext2[0],
                                 line=dict(color="red", width=4)),
                      row=pos[1][0], col=pos[1][1])  # add initial point


        chart2_df = df.loc[(df["mode"] == "classical" + "&fixed") & (df["exp"] == "exp_2")]
        ### 1.3 Cee|Cie - input (p)  POWER
        fig.add_trace(go.Heatmap(z=chart2_df.roi1_meanFR, x=chart2_df.p, y=chart2_df.Cee, coloraxis="coloraxis1"), row=pos[2][0], col=pos[2][1])

        hovertext3 = ["   <b>t%0.2f</b><br>Cee = %0.2f  |  lrc/p = %0.2f<br>Firing rate (Hz) %0.4f"
                     % (timepoints[ii], Cee[ii], avg_weights[ii],
                        chart2_df["roi1_meanFR"].iloc[
                            np.argsort(np.abs(chart2_df["Cee"] - Cee[ii]) + np.abs(chart2_df["p"] - avg_weights[ii])).values[0]])
                     for ii in range(len(Cee))]

        fig.add_trace(go.Scatter(x=avg_weights, y=Cee, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext3,
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7),
                      row=pos[2][0], col=pos[2][1])

        fig.add_trace(go.Scatter(x=[avg_weights[0]], y=[Cee[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext3[0],
                                 line=dict(color="red", width=4)),
                        row=pos[2][0], col=pos[2][1])  # add initial point

        ### 1.4 Cee|Cie - input (p)  FREQUENCY
        fig.add_trace(go.Heatmap(z=chart2_df.roi1_Hz, x=chart2_df.p, y=chart2_df.Cee, coloraxis="coloraxis2"), row=pos[3][0], col=pos[3][1])

        hovertext4 = ["   <b>t%0.2f</b><br>Cee = %0.2f  |  lrc/p = %0.2f<br>Frequency (Hz) %0.4f"
                     % (timepoints[ii], Cee[ii], avg_weights[ii],
                        chart2_df["roi1_Hz"].iloc[
                            np.argsort(np.abs(chart2_df["Cee"] - Cee[ii]) + np.abs(chart2_df["p"] - avg_weights[ii])).values[0]])
                     for ii in range(len(Cee))]

        fig.add_trace(go.Scatter(x=avg_weights, y=Cee, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext4,
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7),
                      row=pos[3][0], col=pos[3][1])

        fig.add_trace(go.Scatter(x=[avg_weights[0]], y=[Cee[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext4[0],
                                 line=dict(color="red", width=4)),
                      row=pos[3][0], col=pos[3][1])  # add initial point

    elif "vH" in mode:

        _, ABt, _, TAUt, ABdam, TAUdam, He, Hi, HAdam = np.average(out_circ[1], axis=2).transpose()

        ## 1. HEATMAPS
        main_folder = "E:\LCCN_Local\PycharmProjects\\ADprogress\FrequencyCharts\\data\\"
        simulations_tag = "PSEmpi_FreqCharts4.0-m03d08y2023-t08h.24m.10s"  # Tag cluster job
        # mode = "classical" if mode=="classic" else mode
        df = pd.read_csv(main_folder + simulations_tag + "/results.csv")

        df_avg = df.groupby(['mode', "p", 'He', 'Hi', 'taue', 'taui', 'Cee', 'Cie', 'exp']).mean().reset_index()
        cmax_freq, cmin_freq = max(df_avg["roi1_Hz"].values), min(df_avg["roi1_Hz"].values)
        cmax_pow, cmin_pow = max(df_avg["roi1_auc"].values), min(df_avg["roi1_auc"].values)
        cmax_fr, cmin_fr = max(df_avg["roi1_meanFR"].values), min(df_avg["roi1_meanFR"].values)


        chart1_df = df.loc[(df["mode"] == "classical" + "&fixed") & (df["exp"] == "exp_HeHi")]
        ### 1.1 He - Hi RATE
        fig.add_trace(go.Heatmap(z=chart1_df.roi1_meanFR, x=chart1_df.He, y=chart1_df.Hi, coloraxis="coloraxis1"), row=pos[0][0], col=pos[0][1])

        hovertext1 = ["   <b>t%0.2f</b><br>He = %0.2f  |  Cie = %0.2f<br>Firing rate (Hz) %0.4f"
                     % (timepoints[ii], He[ii], Hi[ii],
                        chart1_df["roi1_meanFR"].iloc[np.argsort(np.abs(chart1_df["He"] - He[ii]) + np.abs(chart1_df["Hi"] - Hi[ii])).values[0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=He, y=Hi, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext1,
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7), row=pos[0][0], col=pos[0][1])

        fig.add_trace(go.Scatter(x=[He[0]], y=[Hi[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext1[0],
                                 line=dict(color="red", width=4)),
                      row=pos[0][0], col=pos[0][1])  # add initial point

        ### 1.2 He - Cie FREQUENCY
        fig.add_trace(go.Heatmap(z=chart1_df.roi1_Hz, x=chart1_df.He, y=chart1_df.Hi, coloraxis="coloraxis2"), row=pos[1][0], col=pos[1][1])

        hovertext2 = ["   <b>t%0.2f</b><br>He = %0.2f  |  Hi = %0.2f<br>Frequency (dB) %0.4f"
                     % (timepoints[ii], He[ii], Hi[ii],
                        chart1_df["roi1_Hz"].iloc[np.argsort(np.abs(chart1_df["He"] - He[ii]) + np.abs(chart1_df["Hi"] -Hi[ii])).values[0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=He, y=Hi, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext2,
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7), row=pos[1][0], col=pos[1][1])

        fig.add_trace(go.Scatter(x=[He[0]], y=[Hi[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext2[0],
                                 line=dict(color="red", width=4)),
                      row=pos[1][0], col=pos[1][1])  # add initial point



        ### 1.3 He - input (p)  FRate
        chart2_df = df.loc[(df["mode"] == "classical" + "&fixed") & (df["exp"] == "exp_pHe")]
        fig.add_trace(go.Heatmap(z=chart2_df.roi1_meanFR, x=chart2_df.p, y=chart2_df.He, coloraxis="coloraxis1"),
                      row=pos[2][0], col=pos[2][1])

        hovertext3 = ["   <b>t%0.2f</b><br>He = %0.2f  |  lrc/p = %0.2f<br>Firing rate (Hz) %0.4f"
                     % (timepoints[ii], He[ii], avg_weights[ii],
                        chart2_df["roi1_meanFR"].iloc[
                            np.argsort(np.abs(chart2_df["Cee"] - He[ii]) + np.abs(chart2_df["p"] - avg_weights[ii])).values[0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=avg_weights, y=He, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext3,
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7),
                      row=pos[2][0], col=pos[2][1])

        fig.add_trace(go.Scatter(x=[avg_weights[0]], y=[He[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext3[0],
                                 line=dict(color="red", width=4)),
                        row=pos[2][0], col=pos[2][1])  # add initial point

        ### 1.4 Cee|Cie - input (p)  FREQUENCY
        fig.add_trace(go.Heatmap(z=chart2_df.roi1_Hz, x=chart2_df.p, y=chart2_df.He, coloraxis="coloraxis2"), row=pos[3][0], col=pos[3][1])

        hovertext4 = ["   <b>t%0.2f</b><br>Cee = %0.2f  |  lrc/p = %0.2f<br>Frequency (Hz) %0.4f"
                     % (timepoints[ii], He[ii], avg_weights[ii],
                        chart2_df["roi1_Hz"].iloc[
                            np.argsort(np.abs(chart2_df["Cee"] - He[ii]) + np.abs(chart2_df["p"] - avg_weights[ii])).values[0]])
                     for ii in range(len(He))]

        fig.add_trace(go.Scatter(x=avg_weights, y=He, mode="lines+markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext4,
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7),
                      row=pos[3][0], col=pos[3][1])

        fig.add_trace(go.Scatter(x=[avg_weights[0]], y=[He[0]], mode="markers", showlegend=False, hoverinfo="text",
                                 hovertext=hovertext4[0],
                                 line=dict(color="red", width=4)),
                      row=pos[3][0], col=pos[3][1])  # add initial point

    fig.update_layout(coloraxis1=dict(cmin=0, cmax=cmax_fr), coloraxis2=dict(cmin=0, cmax=15))

    return fig


def molecular_viz(fig, out_circ, conn, pos, mode):
    """
    Plot constructor. Pass the data and it constucts certain plots in the desired positions.

    :param fig:
    :param out_prop:
    :param pos:
    :param mode:
    :return:
    """

    timepoints = out_circ[0]

    avg_weights = np.array([np.average(t_list[0]) for t_list in out_circ[2]])
    avg_weights = avg_weights / max(avg_weights) * 0.22

    # 1. Plot parameters and other curves
    if "circ" in mode:

        _, ABt, _, TAUt, ABdam, TAUdam, He, Hi, taue, taui, HAdam = np.average(out_circ[1], axis=2).transpose()


        # 2. STATE VARIABLES
        cmap_p, cmap_s = px.colors.qualitative.Pastel, px.colors.qualitative.Pastel2
        # 2.1 Concentrations of toxic proteins
        for i, pair in enumerate([[ABt, "ABt"], [TAUt, "TAUt"]]):
            trace, name = pair
            fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="M",
                                     line=dict(width=3, color=cmap_s[i])), row=pos[0][0], col=pos[0][1])

        # 2.2 Damage
        for i, pair in enumerate([[ABdam, "ABdam"], [TAUdam, "TAUdam"], [HAdam, "HAdam"]]):
            trace, name = pair
            fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="dam",
                                     line=dict(width=2, color=cmap_s[i]), visible="legendonly"), row=pos[0][0], col=pos[0][1])

        # 2.3 NMM parameters
        for i, pair in enumerate([[He, "He"], [Hi, "Hi"], [taue, "taue"], [avg_weights, "w"]]):
            trace, name = pair
            fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="nmm",
                                     line=dict(width=3, dash="dash", color=cmap_p[i])), secondary_y=True, row=pos[0][0], col=pos[0][1])

    elif "vCC" in mode:

        _, ABt, _, TAUt, ABdam, TAUdam, He, Cee, Cie, HAdam = np.average(out_circ[1], axis=2).transpose()

        # 2. STATE VARIABLES
        cmap_p, cmap_s = px.colors.qualitative.Pastel, px.colors.qualitative.Pastel2
        # 2.1 Concentrations of toxic proteins
        for i, pair in enumerate([[ABt, "ABt"], [TAUt, "TAUt"]]):
            trace, name = pair
            fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="M",
                                     line=dict(width=3, color=cmap_s[i])), row=pos[0][0], col=pos[0][1])

        # 2.2 Damage
        for i, pair in enumerate([[ABdam, "ABdam"], [TAUdam, "TAUdam"], [HAdam, "HAdam"]]):
            trace, name = pair
            fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="dam",
                                     line=dict(width=2, color=cmap_s[i]), visible="legendonly"), row=pos[0][0], col=pos[0][1])

        # 2.3 NMM parameters
        for i, pair in enumerate([[He, "He"], [Cee, "Cee"], [Cie, "Cie"], [avg_weights, "w"]]):
            trace, name = pair
            fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="nmm",
                                     line=dict(width=3, dash="dash", color=cmap_p[i])), secondary_y=True, row=pos[0][0], col=pos[0][1])

    elif "vH" in mode:

        _, ABt, _, TAUt, ABdam, TAUdam, He, Hi, HAdam = np.average(out_circ[1], axis=2).transpose()

        # 2. STATE VARIABLES
        cmap_p, cmap_s = px.colors.qualitative.Pastel, px.colors.qualitative.Pastel2
        # 2.1 Concentrations of toxic proteins
        for i, pair in enumerate([[ABt, "ABt"], [TAUt, "TAUt"]]):
            trace, name = pair
            fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="M",
                                     line=dict(width=3, color=cmap_s[i])), row=pos[0][0], col=pos[0][1])

        # 2.2 Damage
        for i, pair in enumerate([[ABdam, "ABdam"], [TAUdam, "TAUdam"], [HAdam, "HAdam"]]):
            trace, name = pair
            fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="dam",
                                     line=dict(width=2, color=cmap_s[i]), visible="legendonly"), row=pos[0][0], col=pos[0][1])

        # 2.3 NMM parameters
        for i, pair in enumerate([[He, "He"], [Hi, "Hi"], [avg_weights, "w"]]):
            trace, name = pair
            fig.add_trace(go.Scatter(x=timepoints, y=trace, name=name, legendgroup="nmm",
                                     line=dict(width=3, dash="dash", color=cmap_p[i])), secondary_y=True, row=pos[0][0], col=pos[0][1])

    # 2. Plot first trace of dyn Molecular
    _, ABt, _, TAUt = out_circ[1][0][:4]

    fig.add_trace(go.Scatter(x=TAUt, y=ABt, mode="markers", showlegend=False, hovertext=conn.region_labels,
                             marker=dict(opacity=0.7, colorscale='Portland', color=np.random.rand(len(ABt)))),
                  row=pos[1][0], col=pos[1][1])

    return fig


def spectral_viz(fig, out_circ, pos):
    """

    :param fig:
    :param out_circ:
    :param pos: Just one position.
    :return:
    """
    # Gather simulation data and process spectra to extract relatives
    timepoints, peak_freqs, band_powers, abs_ffts, norm_ffts = [], [], [], [], []

    for i, simpack in enumerate(out_circ[2]):
        if len(simpack) > 1:
            timepoints.append(out_circ[0][i])
            peak_freqs.append(np.average(simpack[1][0]))

            # Calculate rel_power per bands
            ffts, freqs = simpack[1][3:]

            # AVG and normalize
            avgfft = np.average(ffts, axis=0)
            abs_ffts.append(avgfft)

            # normavg_fft = avg_fft / sum(avg_fft)
            normavg_fft = (avgfft - min(avgfft)) / (max(avgfft)-min(avgfft))

            pow_beta = sum(normavg_fft[(12 < freqs)])/sum(normavg_fft)
            pow_alpha = sum(normavg_fft[(8 < freqs) & (freqs < 12)])/sum(normavg_fft)
            pow_theta = sum(normavg_fft[(4 < freqs) & (freqs < 8)])/sum(normavg_fft)
            pow_delta = sum(normavg_fft[(2 < freqs) & (freqs < 4)])/sum(normavg_fft)

            band_powers.append([pow_beta, pow_alpha, pow_theta, pow_delta])
            norm_ffts.append(normavg_fft)

    band_powers = np.asarray(band_powers)

    fig.add_trace(go.Scatter(x=timepoints, y=peak_freqs, name="avgFreqPeak", legendgroup="netsim",
                             line=dict(width=2, color="mediumvioletred"), opacity=0.8), row=pos[0][0], col=pos[0][1])

    fig.add_trace(go.Scatter(x=timepoints, y=band_powers[:, 0], name="avgPowBeta (12-30Hz)", legendgroup="netsim",
                             line=dict(dash="dot", width=1, color="gold"), opacity=0.8), secondary_y=True, row=pos[0][0], col=pos[0][1])
    fig.add_trace(go.Scatter(x=timepoints, y=band_powers[:, 1], name="avgPowAlpha (8-12Hz)", legendgroup="netsim",
                             line=dict(dash="dot", width=3, color="darkorange"), opacity=0.8), secondary_y=True, row=pos[0][0], col=pos[0][1])
    fig.add_trace(go.Scatter(x=timepoints, y=band_powers[:, 2], name="avgPowTheta (4-8Hz)", legendgroup="netsim",
                             line=dict(dash="dot", width=2, color="palevioletred"), opacity=0.8), secondary_y=True, row=pos[0][0], col=pos[0][1])
    fig.add_trace(go.Scatter(x=timepoints, y=band_powers[:, 3], name="avgPowDelta (2-4Hz)", legendgroup="netsim",
                             line=dict(dash="dot", width=1, color="mediumorchid"), opacity=0.8), secondary_y=True, row=pos[0][0], col=pos[0][1])

    # Add init for dynamical one
    fig.add_trace(go.Scatter(x=out_circ[2][0][1][4], y=norm_ffts[0], name="normSpectra", legendgroup="netsim",
                             line=dict(width=3, color="lightgray"), opacity=0.8), row=pos[1][0], col=pos[1][1])

    fig.add_trace(go.Scatter(x=out_circ[2][0][1][4], y=abs_ffts[0], name="absSpectra", legendgroup="netsim",
                             line=dict(dash="dash", width=2, color="dimgray"), opacity=0.8), secondary_y=True, row=pos[1][0], col=pos[1][1])

    return fig, abs_ffts, norm_ffts


def FCplusrate_viz(fig, out_circ, conn, pos, surrogates, threshold=0.05):


    # 1_v2. Static lines for posterior, anterior and rate
    # Posterior as Isthmus cingulate to posterior
    posterior_rois = ['ctx-lh-cuneus', 'ctx-lh-inferiorparietal', 'ctx-lh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                      'ctx-lh-lingual', 'ctx-lh-pericalcarine', 'ctx-lh-precuneus', 'ctx-lh-superiorparietal', 'ctx-lh-supramarginal',

                      'ctx-rh-cuneus',  'ctx-rh-inferiorparietal', 'ctx-rh-isthmuscingulate', 'ctx-rh-lateraloccipital',
                      'ctx-rh-lingual', 'ctx-rh-pericalcarine',  'ctx-rh-precuneus', 'ctx-rh-superiorparietal', 'ctx-rh-supramarginal']

    anterior_rois = [
       'ctx-lh-caudalanteriorcingulate', 'ctx-lh-caudalmiddlefrontal',
       'ctx-lh-lateralorbitofrontal',
       'ctx-lh-medialorbitofrontal',
       'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis',
       'ctx-lh-parstriangularis', 'ctx-lh-precentral',
       'ctx-lh-rostralanteriorcingulate', 'ctx-lh-rostralmiddlefrontal',
       'ctx-lh-superiorfrontal','ctx-lh-frontalpole',

       'ctx-rh-caudalanteriorcingulate','ctx-rh-caudalmiddlefrontal',
       'ctx-rh-lateralorbitofrontal',
       'ctx-rh-medialorbitofrontal',
       'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis',
       'ctx-rh-parstriangularis', 'ctx-rh-precentral',
        'ctx-rh-rostralanteriorcingulate', 'ctx-rh-rostralmiddlefrontal',
       'ctx-rh-superiorfrontal', 'ctx-rh-frontalpole']

    # # 1_v1. Static lines for posterior, anterior and rate
    # posterior_rois = ['ctx-lh-cuneus', 'ctx-lh-inferiorparietal', 'ctx-lh-isthmuscingulate', 'ctx-lh-lateraloccipital',
    #                   'ctx-lh-lingual', 'ctx-lh-pericalcarine','ctx-lh-postcentral', 'ctx-lh-posteriorcingulate',
    #                   'ctx-lh-precuneus', 'ctx-lh-superiorparietal', 'ctx-lh-supramarginal',
    #
    #                   'ctx-rh-cuneus',  'ctx-rh-inferiorparietal', 'ctx-rh-isthmuscingulate', 'ctx-rh-lateraloccipital',
    #                   'ctx-rh-lingual', 'ctx-rh-pericalcarine', 'ctx-rh-postcentral', 'ctx-rh-posteriorcingulate',
    #                   'ctx-rh-precuneus', 'ctx-rh-superiorparietal', 'ctx-rh-supramarginal']
    #
    # anterior_rois = [
    #    'ctx-lh-caudalanteriorcingulate', 'ctx-lh-caudalmiddlefrontal',
    #    'ctx-lh-lateralorbitofrontal',
    #    'ctx-lh-medialorbitofrontal',
    #    'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis',
    #    'ctx-lh-parstriangularis', 'ctx-lh-precentral',
    #    'ctx-lh-rostralanteriorcingulate', 'ctx-lh-rostralmiddlefrontal',
    #    'ctx-lh-superiorfrontal','ctx-lh-frontalpole',
    #
    #    'ctx-rh-caudalanteriorcingulate','ctx-rh-caudalmiddlefrontal',
    #    'ctx-rh-lateralorbitofrontal',
    #    'ctx-rh-medialorbitofrontal',
    #    'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis',
    #    'ctx-rh-parstriangularis', 'ctx-rh-precentral',
    #     'ctx-rh-rostralanteriorcingulate', 'ctx-rh-rostralmiddlefrontal',
    #    'ctx-rh-superiorfrontal', 'ctx-rh-frontalpole']

    posterior_ids = np.array([i for i, roi in enumerate(conn.region_labels) if roi in posterior_rois])
    anterior_ids = [i for i, roi in enumerate(conn.region_labels) if roi in anterior_rois]

    timepoints, posterior_fc, anterior_fc, ant_avgrate, post_avgrate = [], [], [], [], []
    for i, simpack in enumerate(out_circ[2]):
        if len(simpack) > 1:
            timepoints.append(out_circ[0][i])

            post_avgrate.append(np.average(out_circ[2][i][3][posterior_ids]))
            ant_avgrate.append(np.average(out_circ[2][i][3][anterior_ids]))
            posterior_fc.append(np.average(out_circ[2][i][2][:, posterior_ids][posterior_ids]))
            anterior_fc.append(np.average(out_circ[2][i][2][:, anterior_ids][anterior_ids]))

    # Static lines
    fig.add_trace(go.Scatter(x=timepoints, y=post_avgrate, name="avgRate Posterior", legendgroup="fc",
                             line=dict(width=2, color="firebrick", dash="dot"), opacity=0.8), secondary_y=True, row=pos[0][0], col=pos[0][1])

    fig.add_trace(go.Scatter(x=timepoints, y=ant_avgrate, name="avgRate Anterior", legendgroup="fc",
                             line=dict(width=2, color="cornflowerblue", dash="dot"), opacity=0.8), secondary_y=True, row=pos[0][0], col=pos[0][1])

    fig.add_trace(go.Scatter(x=timepoints, y=posterior_fc, name="avgFC Posterior", legendgroup="fc",
                             line=dict(width=3, color="firebrick"), opacity=0.8), row=pos[0][0], col=pos[0][1])

    fig.add_trace(go.Scatter(x=timepoints, y=anterior_fc, name="avgFC Anterior", legendgroup="fc",
                             line=dict(width=2, color="cornflowerblue"), opacity=0.8), row=pos[0][0], col=pos[0][1])

    # 2. Go for the 3D plot
    regionLabels = conn.region_labels
    weights = conn.weights
    centres = conn.centres

    # Edges trace
    ## Filter edges to show: remove low connected nodes via thresholding
    edges_ids = list(combinations([i for i, roi in enumerate(regionLabels)], r=2))
    edges_ids = [(i, j) for i, j in edges_ids if weights[i, j] > threshold]

    ## Define [start, end, None] per coordinate and connection
    edges_x = [elem for sublist in [[centres[i, 0]] + [centres[j, 0]] + [None] for i, j in edges_ids] for elem in
               sublist]
    edges_y = [elem for sublist in [[centres[i, 1]] + [centres[j, 1]] + [None] for i, j in edges_ids] for elem in
               sublist]
    edges_z = [elem for sublist in [[centres[i, 2]] + [centres[j, 2]] + [None] for i, j in edges_ids] for elem in
               sublist]

    ## Define color per connection based on FC changes
    increaseFC = [out_circ[2][i][2] - out_circ[2][0][2] for i, t in enumerate(out_circ[0]) if len(out_circ[2][i]) > 1]

    if surrogates is not None:
        surrogates = np.array(surrogates)
        increaseFC_norm = [[(increaseFC[ii][i, j] + 1) / 2
                            if pg.ttest(x=surrogates[:, i, j], y=increaseFC[ii][i, j])["p-val"].values < 0.05 / len(edges_ids)
                            else (0.9 + 1) / 2 for i, j in edges_ids] for ii, inc in enumerate(increaseFC)]
    else:
        increaseFC_norm = [[
            (increaseFC[ii][i, j] + 1) / 2 for i, j in edges_ids] for ii, inc in enumerate(increaseFC)]

    edges_color = [[px.colors.sample_colorscale("Jet", e)[0] for e in inc for i in range(3)] for inc in increaseFC_norm]

    fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip",
                               line=dict(color=edges_color[0], width=3), opacity=0.6, name="Edges"), row=pos[1][0], col=pos[1][1])

    # Nodes trace
    ## Define size per degree
    degree, size_range = np.sum(weights, axis=1), [8, 25]
    size = ((degree - np.min(degree)) * (size_range[1] - size_range[0]) / (np.max(degree) - np.min(degree))) + size_range[0]

    ## Define color per hyperactivity
    # increasePOW = [out_circ[1][i][2] - out_circ[1][0][2] for i, t in enumerate(out_circ[0])]
    # increasePOW_norm = [[((increasePOW[ii][i] - np.min(increasePOW)) / (np.max(increasePOW) - np.min(increasePOW)))
    #                      for i, roi in enumerate(regionLabels)] for ii, t in enumerate(out_circ[0])]
    dActivity = [out_circ[2][i][3] - out_circ[2][0][3] for i, t in enumerate(out_circ[0]) if len(out_circ[2][i]) > 1]

    dActivity_norm = [[((dA[i] - np.min(dActivity)) / (np.max(dActivity) - np.min(dActivity)))
                         for i, roi in enumerate(regionLabels)] for ii, dA in enumerate(dActivity)]

    nodes_color = [[px.colors.sample_colorscale("Jet", e)[0] for e in dA] for dA in dActivity_norm]

    # Create text labels per ROI
    hovertext3d = [["<b>" + roi + "</b>" +
                    "<br>Power (dB) " + str(round(out_circ[2][ii][1][1][i], 5)) +
                    "<br>Frequency (Hz) " + str(round(out_circ[2][ii][1][0][i], 5)) +
                    "<br><br>Firing Rate increase (Hz) " + str(round(out_circ[2][ii][3][i] - out_circ[2][0][3][i], 5))
                    for i, roi in enumerate(regionLabels)] for ii, t in enumerate(out_circ[0]) if len(out_circ[2][ii]) > 1]

    fig.add_trace(go.Scatter3d(x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], hoverinfo="text",
                               hovertext=hovertext3d[0], mode="markers", name="Nodes",
                               marker=dict(size=size, color=nodes_color[0], opacity=1,
                                           line=dict(color="gray", width=2))), row=pos[1][0], col=pos[1][1])

    fig = addpial(fig, mode="surface", opacity=0.1, row=pos[1][0], col=pos[1][1])

    return fig, edges_color, nodes_color, hovertext3d


def paramtraj_in3D(out_circ, mode, PSE3d_tag, folder="figures", auto_open=True, param_info=None):

    if "vH" in mode:
        main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\FrequencyCharts\data\\'
        df = pd.read_csv(main_folder + PSE3d_tag + "/results.csv")

        df = df.groupby(["p", "He", "Hi"]).mean().reset_index()

        # Fixed point simulations to None: rio1_Hz, roi1_auc, roi1_meanFR
        df["roi1_Hz"].loc[df["roi1_auc"] < 1e-6] = None
        df["roi1_meanFR"].loc[df["roi1_auc"] < 1e-6] = None
        df["roi1_auc"].loc[df["roi1_auc"] < 1e-6] = None


        He_PSEvals, Hi_PSEvals, p_PSEvals = \
            sorted(set(df.He)), sorted(set(df.Hi)), sorted(set(df.p))

        init_p = p_PSEvals[np.argmin(abs(np.array(p_PSEvals) - 0.22))]
        init_Hi, init_He = 22, 3.25

        # define the combination of params for each timestep
        weights_traj = np.array([np.average(out[0]) for out in out_circ[2]])
        weights_traj = weights_traj / max(weights_traj) * 0.22

        params_traj = np.average(np.array(out_circ[1]), axis=2)[:, 6:-1]

        # Associate each param value in trajectory with a PSE value
        assocPSEvals_inTraj = pd.DataFrame(
            np.array([(i, out_circ[0][i],
                       He_PSEvals[np.argmin(abs(params[0] - He_PSEvals))],
                       Hi_PSEvals[np.argmin(abs(params[1] - Hi_PSEvals))],
                       p_PSEvals[np.argmin(abs(weights_traj[i] - p_PSEvals))])
                      for i, params in enumerate(params_traj)]), columns=["i", "t", "He", "Hi", "p"])

        # Tuple combination of PSE values per timepoint
        setsPSEvals_inTraj = set([(He_PSEvals[np.argmin(abs(params[0] - He_PSEvals))],
                                Hi_PSEvals[np.argmin(abs(params[1] - Hi_PSEvals))],
                                p_PSEvals[np.argmin(abs(weights_traj[i] - p_PSEvals))])
                               for i, params in enumerate(params_traj)])

        # Define the temporal frames per parameter combination
        minmaxt_inSets = \
            pd.DataFrame(np.array(
                [(set + (np.min(assocPSEvals_inTraj["t"].loc[(assocPSEvals_inTraj["He"] == set[0]) & (assocPSEvals_inTraj["Hi"] == set[1]) & (assocPSEvals_inTraj["p"] == set[2])].values),
                   np.max(assocPSEvals_inTraj["t"].loc[(assocPSEvals_inTraj["He"]==set[0]) & (assocPSEvals_inTraj["Hi"]==set[1]) &
                                                         (assocPSEvals_inTraj["p"]==set[2])].values)))
             for set in setsPSEvals_inTraj]), columns=["He", "Hi", "p", "tmin", "tmax"])


        minmaxt_inSets = minmaxt_inSets.sort_values(["tmin"])

        df["tmin"], df["tmax"] = None, None
        for i, row in minmaxt_inSets.iterrows():
            df["tmin"].loc[(df["He"]==row.He) & (df["Hi"]==row.Hi) & (df["p"]==row.p)] = row.tmin
            df["tmax"].loc[(df["He"]==row.He) & (df["Hi"]==row.Hi) & (df["p"]==row.p)] = row.tmax


        ## PLOTTING: animation over p
        if mode == "full":

            df = df.iloc[:, :-2].copy()
            # df["freq"].loc[df["freq"] == 0] = None

            fig = make_subplots(rows=3, cols=3,
                                specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]], shared_xaxes=True,
                                row_titles=["Frequency (Hz)", "meanFR (Hz)", "Power (dB)"])

            # 1. column 1: p axis
            dfsub = df.loc[df["p"] == init_p].dropna()

            # 1.1  Freq
            fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub.roi1_Hz, zmin=min(df.roi1_Hz), zmax=max(df.roi1_Hz),
                                     colorbar=dict(len=0.3, y=0.9, thickness=15)), row=1, col=1)
            # 1.2 Firing Rate
            fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub.roi1_meanFR, colorscale="Cividis", zmin=min(df.roi1_meanFR),
                                     zmax=max(df.roi1_meanFR), colorbar=dict(len=0.3, y=0.5, thickness=15)), row=2, col=1)
            # 1.3 Power
            fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub.roi1_auc, colorscale="Viridis", zmin=min(df.roi1_auc),
                                     zmax=max(df.roi1_auc), colorbar=dict(len=0.3, y=0.1, thickness=15)), row=3, col=1)


            # 2. Column 2: Hi axis
            dfsub = df.loc[df["Hi"] == init_Hi].dropna()

            # 2.1 Freq
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.He, z=dfsub.roi1_Hz, zmin=min(df.roi1_Hz), zmax=max(df.roi1_Hz), showscale=False,
                                     colorbar=dict(len=0.3, y=0.9, thickness=15)), row=1, col=2)
            # 2.2 Firing Rate
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.He, z=dfsub.roi1_meanFR, colorscale="Cividis", zmin=min(df.roi1_meanFR), showscale=False,
                                     zmax=max(df.roi1_meanFR), colorbar=dict(len=0.3, y=0.1, thickness=15)), row=2, col=2)
            # 2.3 Power
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.He, z=dfsub.roi1_auc, colorscale="Viridis", zmin=min(df.roi1_auc), showscale=False,
                                     zmax=max(df.roi1_auc), colorbar=dict(len=0.3, y=0.5, thickness=15)), row=3, col=2)


            # 3. Column 3: He axis
            dfsub = df.loc[df["He"] == init_He].dropna()

            # 3.1 Freq
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.Hi, z=dfsub.roi1_Hz, zmin=min(df.roi1_Hz), zmax=max(df.roi1_Hz), showscale=False,
                                     colorbar=dict(len=0.3, y=0.9, thickness=15)), row=1, col=3)
            # 3.2 Firing Rate
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.Hi, z=dfsub.roi1_meanFR, colorscale="Cividis", zmin=min(df.roi1_meanFR), showscale=False,
                                     zmax=max(df.roi1_meanFR), colorbar=dict(len=0.3, y=0.1, thickness=15)), row=2, col=3)
            # 3.3 pow
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.Hi, z=dfsub.roi1_auc, colorscale="Viridis", zmin=min(df.roi1_auc), showscale=False,
                                     zmax=max(df.roi1_auc), colorbar=dict(len=0.3, y=0.5, thickness=15)), row=3, col=3)


            # 4. Plot scatters for trajectory
            sub_traj = minmaxt_inSets[minmaxt_inSets["p"]==0].dropna()
            # sub_traj = df[df["p"] == 0].dropna()
            hover = ["He%0.2f, Hi%0.2f, p%0.2f<br>tmin - %0.2f  |  tmax - %0.2f" %
                     (row.He, row.Hi, row.p, row.tmin, row.tmax) for i, row in sub_traj.iterrows()]
            fig.add_trace(go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=3, col=1)

            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover, hoverinfo="text", showlegend=False), row=2, col=2)
            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover, hoverinfo="text", showlegend=False), row=3, col=2)

            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=3)
            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=2, col=3)
            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=3, col=3)

            frames = []

            lastHi, lastHe = init_Hi, init_He
            for i, p in enumerate(sorted(set(df.p), reverse=True)):

                sub_p = df.loc[df["p"] == p].dropna()

                # Define H @p and subset
                sub_traj = minmaxt_inSets[minmaxt_inSets["p"] == p].dropna()

                lastHi = sub_traj.Hi.min() if len(sub_traj) > 0 else lastHi
                lastHe = sub_traj.He.min() if len(sub_traj) > 0 else lastHe

                sub_Hi = df.loc[df["Hi"] == lastHi].dropna()
                sub_He = df.loc[df["He"] == lastHe].dropna()


                hover = ["He%0.2f, Hi%0.2f, p%0.2f <br>tmin - %0.2f  |  tmax - %0.2f" %
                         (row.He, row.Hi, row.p, row.tmin, row.tmax) for i, row in sub_traj.iterrows()]

                frames.append(go.Frame(data=[

                    go.Heatmap(x=sub_p.He, y=sub_p.Hi, z=sub_p.roi1_Hz),
                    go.Heatmap(x=sub_p.He, y=sub_p.Hi, z=sub_p.roi1_meanFR),
                    go.Heatmap(x=sub_p.He, y=sub_p.Hi, z=sub_p.roi1_auc),

                    go.Heatmap(x=sub_Hi.p, y=sub_Hi.He, z=sub_Hi.roi1_Hz),
                    go.Heatmap(x=sub_Hi.p, y=sub_Hi.He, z=sub_Hi.roi1_meanFR),
                    go.Heatmap(x=sub_Hi.p, y=sub_Hi.He, z=sub_Hi.roi1_auc),

                    go.Heatmap(x=sub_He.p, y=sub_He.Hi, z=sub_He.roi1_Hz),
                    go.Heatmap(x=sub_He.p, y=sub_He.Hi, z=sub_He.roi1_meanFR),
                    go.Heatmap(x=sub_He.p, y=sub_He.Hi, z=sub_He.roi1_auc),

                    go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover),
                    go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover),
                    go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover),

                    go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover),
                    go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover),
                    go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover),

                    go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover),
                    go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover),
                    go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover)],

                traces=list(np.arange(18)), name=str(round(p, 4))))

            fig.update(frames=frames)

            He_axis = dict(title="He", range=[min(df.He), max(df.He)], autorange=False, showticklabels=True)
            Hi_axis = dict(title="Hi", range=[min(df.Hi), max(df.Hi)], autorange=False, showticklabels=True)
            p_axis = dict(title="p", range=[min(df.p), max(df.p)], autorange=False, showticklabels=True)

            # CONTROLS : Add sliders and buttons
            fig.update_layout(
                title="Param trajectory on 3D parameter space (%s) <br> Heatmaps from single-node sims <br> init. conditions [%s]" % (mode, param_info),
                template="plotly_white",
                xaxis1=He_axis, yaxis1=Hi_axis,
                xaxis2=p_axis, yaxis2=He_axis,
                xaxis3=p_axis, yaxis3=Hi_axis,

                xaxis4=He_axis, yaxis4=Hi_axis,
                xaxis5=p_axis, yaxis5=He_axis,
                xaxis6=p_axis, yaxis6=Hi_axis,

                xaxis7=He_axis, yaxis7=Hi_axis,
                xaxis8=p_axis, yaxis8=He_axis,
                xaxis9=p_axis, yaxis9=Hi_axis,

                updatemenus=[dict(type="buttons", showactive=True, y=1.30, x=1.05, xanchor="right",
                                  buttons=[
                                      dict(label="Play", method="animate",
                                           args=[None,
                                                 dict(frame=dict(duration=500, redraw=True, easing="cubic-in-out"),
                                                      transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                                      dict(label="Pause", method="animate",
                                           args=[[None],
                                                 dict(frame=dict(duration=0, redraw=False, easing="cubic-in-out"),
                                                      transition=dict(duration=0), mode="immediate")])])],
                sliders=[dict(
                    steps=[dict(args=[[f.name],
                                      dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                           transition=dict(duration=0))], label=f.name, method='animate', )
                           for f in frames],
                    x=0.97, xanchor="right", y=1.35, len=0.5,
                    currentvalue=dict(font=dict(size=15), prefix="p - ", visible=True, xanchor="left"),
                    tickcolor="white")],
                )

            pio.write_html(fig, file=folder + "/Trajectory_onAnimatedPSE3D_"+mode+".html", auto_open=auto_open, auto_play=False)

        elif mode == "freq":

            df = df.iloc[:, :-2].copy()
            # df["freq"].loc[df["freq"] == 0] = None

            fig = make_subplots(rows=1, cols=3,
                                specs=[[{}, {}, {}]], shared_xaxes=True,
                                row_titles=["Frequency (Hz)"])

            # 1. column 1: p axis
            dfsub = df.loc[df["p"] == init_p].dropna()
            # 1.1  Freq
            fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub.roi1_Hz, zmin=min(df.roi1_Hz), zmax=max(df.roi1_Hz),
                                     colorbar=dict(len=0.3, y=0.9, thickness=15)), row=1, col=1)

            # 2. Column 2: Hi axis
            dfsub = df.loc[df["Hi"] == init_Hi].dropna()
            # 2.1 Freq
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.He, z=dfsub.roi1_Hz, zmin=min(df.roi1_Hz), zmax=max(df.roi1_Hz),
                                     showscale=False,
                                     colorbar=dict(len=0.3, y=0.9, thickness=15)), row=1, col=2)
            # 3. Column 3: He axis
            dfsub = df.loc[df["He"] == init_He].dropna()
            # 3.1 Freq
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.Hi, z=dfsub.roi1_Hz, zmin=min(df.roi1_Hz), zmax=max(df.roi1_Hz),
                                     showscale=False,
                                     colorbar=dict(len=0.3, y=0.9, thickness=15)), row=1, col=3)

            # 4. Plot scatters for trajectory
            sub_traj = minmaxt_inSets[minmaxt_inSets["p"] == 0].dropna()
            # sub_traj = df[df["p"] == 0].dropna()
            hover = ["He%0.2f, Hi%0.2f, p%0.2f<br>tmin - %0.2f  |  tmax - %0.2f" %
                     (row.He, row.Hi, row.p, row.tmin, row.tmax) for i, row in sub_traj.iterrows()]

            fig.add_trace(go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False),
                          row=1, col=1)

            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover, hoverinfo="text", showlegend=False),
                          row=1, col=2)

            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False),
                          row=1, col=3)

            frames = []

            lastHi, lastHe = init_Hi, init_He
            for i, p in enumerate(sorted(set(df.p), reverse=True)):
                sub_p = df.loc[df["p"] == p].dropna()

                # Define H @p and subset
                sub_traj = minmaxt_inSets[minmaxt_inSets["p"] == p].dropna()

                lastHi = sub_traj.Hi.min() if len(sub_traj) > 0 else lastHi
                lastHe = sub_traj.He.min() if len(sub_traj) > 0 else lastHe

                sub_Hi = df.loc[df["Hi"] == lastHi].dropna()
                sub_He = df.loc[df["He"] == lastHe].dropna()

                hover = ["He%0.2f, Hi%0.2f, p%0.2f <br>tmin - %0.2f  |  tmax - %0.2f" %
                         (row.He, row.Hi, row.p, row.tmin, row.tmax) for i, row in sub_traj.iterrows()]

                frames.append(go.Frame(data=[

                    go.Heatmap(x=sub_p.He, y=sub_p.Hi, z=sub_p.roi1_Hz),

                    go.Heatmap(x=sub_Hi.p, y=sub_Hi.He, z=sub_Hi.roi1_Hz),

                    go.Heatmap(x=sub_He.p, y=sub_He.Hi, z=sub_He.roi1_Hz),

                    go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover),

                    go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover),

                    go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover)],

                    traces=list(np.arange(6)), name=str(round(p, 4))))

            fig.update(frames=frames)

            He_axis = dict(title="He", range=[min(df.He), max(df.He)], autorange=False, showticklabels=True)
            Hi_axis = dict(title="Hi", range=[min(df.Hi), max(df.Hi)], autorange=False, showticklabels=True)
            p_axis = dict(title="p", range=[min(df.p), max(df.p)], autorange=False, showticklabels=True)

            # CONTROLS : Add sliders and buttons
            fig.update_layout(
                title="Param trajectory on 3D parameter space (%s) <br> Heatmaps from single-node sims <br> init. conditions [%s]" % (
                mode, param_info),
                template="plotly_white",
                xaxis1=He_axis, yaxis1=Hi_axis,
                xaxis2=p_axis, yaxis2=He_axis,
                xaxis3=p_axis, yaxis3=Hi_axis,

                updatemenus=[dict(type="buttons", showactive=True, y=1.30, x=1.05, xanchor="right",
                                  buttons=[
                                      dict(label="Play", method="animate",
                                           args=[None,
                                                 dict(frame=dict(duration=500, redraw=True, easing="cubic-in-out"),
                                                      transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                                      dict(label="Pause", method="animate",
                                           args=[[None],
                                                 dict(frame=dict(duration=0, redraw=False, easing="cubic-in-out"),
                                                      transition=dict(duration=0), mode="immediate")])])],
                sliders=[dict(
                    steps=[dict(args=[[f.name],
                                      dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                           transition=dict(duration=0))], label=f.name, method='animate', )
                           for f in frames],
                    x=0.97, xanchor="right", y=1.35, len=0.5,
                    currentvalue=dict(font=dict(size=15), prefix="p - ", visible=True, xanchor="left"),
                    tickcolor="white")],
            )

            pio.write_html(fig, file=folder + "/Trajectory_onAnimatedPSE3D_" + mode + ".html", auto_open=auto_open,
                           auto_play=False)

        elif mode == "rate":

            df = df.iloc[:, :-2].copy()
            # df["freq"].loc[df["freq"] == 0] = None

            fig = make_subplots(rows=1, cols=3,
                                specs=[[{}, {}, {}]], shared_xaxes=True,
                                row_titles=["Frequency (Hz)", "meanFR (Hz)", "Power (dB)"])

            # 1. column 1: p axis
            dfsub = df.loc[df["p"] == init_p].dropna()
            # 1.2 Firing Rate
            fig.add_trace(go.Heatmap(x=dfsub.He, y=dfsub.Hi, z=dfsub.roi1_meanFR, colorscale="Cividis", zmin=min(df.roi1_meanFR),
                                     zmax=max(df.roi1_meanFR), colorbar=dict(len=0.3, y=0.5, thickness=15)), row=1, col=1)

            # 2. Column 2: Hi axis
            dfsub = df.loc[df["Hi"] == init_Hi].dropna()
            # 2.2 Firing Rate
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.He, z=dfsub.roi1_meanFR, colorscale="Cividis", zmin=min(df.roi1_meanFR), showscale=False,
                                     zmax=max(df.roi1_meanFR), colorbar=dict(len=0.3, y=0.1, thickness=15)), row=1, col=2)

            # 3. Column 3: He axis
            dfsub = df.loc[df["He"] == init_He].dropna()
            # 3.2 Firing Rate
            fig.add_trace(go.Heatmap(x=dfsub.p, y=dfsub.Hi, z=dfsub.roi1_meanFR, colorscale="Cividis", zmin=min(df.roi1_meanFR), showscale=False,
                                     zmax=max(df.roi1_meanFR), colorbar=dict(len=0.3, y=0.1, thickness=15)), row=1, col=3)



            # 4. Plot scatters for trajectory
            sub_traj = minmaxt_inSets[minmaxt_inSets["p"]==0].dropna()
            # sub_traj = df[df["p"] == 0].dropna()
            hover = ["He%0.2f, Hi%0.2f, p%0.2f<br>tmin - %0.2f  |  tmax - %0.2f" %
                     (row.He, row.Hi, row.p, row.tmin, row.tmax) for i, row in sub_traj.iterrows()]

            fig.add_trace(go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover, hoverinfo="text", showlegend=False), row=1, col=3)


            frames = []

            lastHi, lastHe = init_Hi, init_He
            for i, p in enumerate(sorted(set(df.p), reverse=True)):

                sub_p = df.loc[df["p"] == p].dropna()

                # Define H @p and subset
                sub_traj = minmaxt_inSets[minmaxt_inSets["p"] == p].dropna()

                lastHi = sub_traj.Hi.min() if len(sub_traj) > 0 else lastHi
                lastHe = sub_traj.He.min() if len(sub_traj) > 0 else lastHe

                sub_Hi = df.loc[df["Hi"] == lastHi].dropna()
                sub_He = df.loc[df["He"] == lastHe].dropna()


                hover = ["He%0.2f, Hi%0.2f, p%0.2f <br>tmin - %0.2f  |  tmax - %0.2f" %
                         (row.He, row.Hi, row.p, row.tmin, row.tmax) for i, row in sub_traj.iterrows()]

                frames.append(go.Frame(data=[

                    go.Heatmap(x=sub_p.He, y=sub_p.Hi, z=sub_p.roi1_meanFR),
                    go.Heatmap(x=sub_Hi.p, y=sub_Hi.He, z=sub_Hi.roi1_meanFR),
                    go.Heatmap(x=sub_He.p, y=sub_He.Hi, z=sub_He.roi1_meanFR),

                    go.Scatter(x=sub_traj.He, y=sub_traj.Hi, hovertext=hover),

                    go.Scatter(x=sub_traj.p, y=sub_traj.He, hovertext=hover),

                    go.Scatter(x=sub_traj.p, y=sub_traj.Hi, hovertext=hover)],

                traces=list(np.arange(6)), name=str(round(p, 4))))

            fig.update(frames=frames)

            He_axis = dict(title="He", range=[min(df.He), max(df.He)], autorange=False, showticklabels=True)
            Hi_axis = dict(title="Hi", range=[min(df.Hi), max(df.Hi)], autorange=False, showticklabels=True)
            p_axis = dict(title="p", range=[min(df.p), max(df.p)], autorange=False, showticklabels=True)

            # CONTROLS : Add sliders and buttons
            fig.update_layout(
                title="Param trajectory on 3D parameter space (%s) <br> Heatmaps from single-node sims <br> init. conditions [%s]" % (mode, param_info),
                template="plotly_white",
                xaxis1=He_axis, yaxis1=Hi_axis,
                xaxis2=p_axis, yaxis2=He_axis,
                xaxis3=p_axis, yaxis3=Hi_axis,

                updatemenus=[dict(type="buttons", showactive=True, y=1.30, x=1.05, xanchor="right",
                                  buttons=[
                                      dict(label="Play", method="animate",
                                           args=[None,
                                                 dict(frame=dict(duration=500, redraw=True, easing="cubic-in-out"),
                                                      transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                                      dict(label="Pause", method="animate",
                                           args=[[None],
                                                 dict(frame=dict(duration=0, redraw=False, easing="cubic-in-out"),
                                                      transition=dict(duration=0), mode="immediate")])])],
                sliders=[dict(
                    steps=[dict(args=[[f.name],
                                      dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                           transition=dict(duration=0))], label=f.name, method='animate', )
                           for f in frames],
                    x=0.97, xanchor="right", y=1.35, len=0.5,
                    currentvalue=dict(font=dict(size=15), prefix="p - ", visible=True, xanchor="left"),
                    tickcolor="white")],
                )

            pio.write_html(fig, file=folder + "/Trajectory_onAnimatedPSE3D_"+mode+".html", auto_open=auto_open, auto_play=False)




def braidPlot(out_circ, conn, mode="surface", rho_vals=None, title="new", folder="figures", auto_open=True):

    # # Regions in Braak stages for TAU from Putra (2021)
    # rI = ["ctx-rh-entorhinal", "ctx-lh-entorhinal"]
    # rII = ["Right-Hippocampus", "Left-Hippocampus"]
    # rIII = ["ctx-rh-parahippocampal", "ctx-lh-parahippocampal"]
    # rIV = ["ctx-rh-caudalanteriorcingulate", "ctx-rh-rostralanteriorcingulate",
    #        "ctx-lh-caudalanteriorcingulate", "ctx-lh-rostralanteriorcingulate"]
    # rV = ["ctx-rh-cuneus", 'ctx-rh-pericalcarine', 'ctx-rh-lateraloccipital', 'ctx-rh-lingual',
    #       "ctx-lh-cuneus", 'ctx-lh-pericalcarine', 'ctx-lh-lateraloccipital', 'ctx-lh-lingual']
    #
    # r_names = ["Entorhinal", "Hippocampus", "Parahippocampus", "Anterior Cingulate", "other Cortical"]


    # Regions in Braak stages for TAU from Therriault (2022)
    rI = ["ctx-rh-entorhinal", "ctx-lh-entorhinal"]

    rII = ["Right-Hippocampus", "Left-Hippocampus"]

    rIII = ["ctx-rh-parahippocampal", "ctx-lh-parahippocampal",
            'Right-Amygdala', 'Left-Amygdala',
            'ctx-rh-fusiform', 'ctx-lh-fusiform',
            'ctx-rh-lingual', 'ctx-lh-lingual']

    rIV = ['ctx-rh-insula', 'ctx-lh-insula',
           'ctx-rh-inferiortemporal', 'ctx-lh-inferiortemporal',
           'ctx-rh-posteriorcingulate', 'ctx-lh-posteriorcingulate',
           'ctx-rh-inferiorparietal', 'ctx-lh-inferiorparietal']

    rV = ['ctx-rh-medialorbitofrontal', 'ctx-lh-medialorbitofrontal',
          'ctx-rh-superiortemporal', 'ctx-lh-superiortemporal',
          'ctx-rh-cuneus', 'ctx-lh-cuneus',
          "ctx-rh-caudalanteriorcingulate", "ctx-lh-caudalanteriorcingulate",
          "ctx-rh-rostralanteriorcingulate", "ctx-lh-rostralanteriorcingulate",
          'ctx-rh-supramarginal', 'ctx-lh-supramarginal',
          'ctx-rh-lateraloccipital', 'ctx-lh-lateraloccipital',
          'ctx-rh-precuneus', 'ctx-lh-precuneus',
          'ctx-rh-superiorparietal', 'ctx-lh-superiorparietal',
          'ctx-rh-superiorfrontal', 'ctx-lh-superiorfrontal',
          'ctx-rh-rostralmiddlefrontal', 'ctx-lh-rostralmiddlefrontal']

    # When working with reduced versions of the DK,
    # you will need to remove Braak reference regions that are not in conn
    rI = [roi for roi in rI if roi in conn.region_labels]
    rII = [roi for roi in rII if roi in conn.region_labels]
    rIII = [roi for roi in rIII if roi in conn.region_labels]
    rIV = [roi for roi in rIV if roi in conn.region_labels]
    rV = [roi for roi in rV if roi in conn.region_labels]

    r_names = ["rI", "rII", "rIII", "rIV", "rV"]


    if mode == "data_intime":

        tau_dyn = np.asarray(out_circ[1])[:, 3, :]

        rxs = [[tau_dyn[:, list(conn.region_labels).index(roi)] for roi in rx] for rx in [rI, rII, rIII, rIV, rV]]
        rxs_avg = np.asarray([np.average(np.asarray(rx), axis=0) for rx in rxs])
        # For each percentage; tell me in what time it was rebased.

        rxs_perc = rxs_avg.transpose() / rxs_avg[:, -1] * 100
                                                        # (-)rxs_avg to sort in descending order
        # sortRx_TIME = [str(np.asarray(r_names)[np.argsort(-row)]) for row in rxs_perc]

        return rxs_perc.transpose()

    elif (mode == "diagram") or (mode == "data_inperc"):

        tau_dyn = np.asarray(out_circ[1])[:, 3, :]

        rxs = [[tau_dyn[:, list(conn.region_labels).index(roi)] for roi in rx] for rx in [rI, rII, rIII, rIV, rV]]
        rxs_avg = np.asarray([np.average(np.asarray(rx), axis=0) for rx in rxs])
        # For each percentage; tell me in what time it was rebased.

        rxs_perc = rxs_avg.transpose() / np.max(rxs_avg, axis=1) * 100

        percxRx_TIME = np.asarray([[np.min(np.argwhere(rx >= perc)) for rx in rxs_perc.T] for perc in range(1, 100)])
        # columns=["rI", "rII", "rIII", "rIV", "rV"])
        # Tell me the rx order based on perc_time
        percxRx_ORDER_braiddiag = np.asarray([np.argsort(np.argsort(row)) for row in percxRx_TIME])
        percxRx_ORDER_braiddiag = percxRx_ORDER_braiddiag.astype(float)

        # Take care with equalities in perc_time that are sorted randomly
        for i, row in enumerate(percxRx_TIME):
            if len(set(row)) < 5:
                for val in set(row):
                    if len(row[row == val]) > 1:
                        percxRx_ORDER_braiddiag[i, row == val] = \
                            np.tile(np.average(percxRx_ORDER_braiddiag[i, row == val]), (len(row[row == val])))

        # compact the diagram in a line with categorical patters
        patterns = [str(row) for row in percxRx_ORDER_braiddiag]

        if mode == "diagram":
            # Plotting
            cmap = px.colors.qualitative.Set2
            fig = make_subplots(rows=2, cols=1, subplot_titles=["Temporal dynamics: first time surpassing percentage",
                                                                "Braid diagram"])
            for i in range(percxRx_ORDER_braiddiag.shape[1]):
                fig.add_trace(go.Scatter(x=list(range(len(rxs_perc))), y=rxs_perc[:, i], name=r_names[i],
                                         legendgroup="r" + str(i + 1),
                                         showlegend=False, line=dict(color=cmap[i])), row=1, col=1)
                fig.add_trace(go.Scatter(x=list(range(1, 100)), y=percxRx_ORDER_braiddiag[:, i], name=r_names[i],
                                         legendgroup="r" + str(i + 1), showlegend=True, line=dict(color=cmap[i])),
                              row=2, col=1)

            fig.update_layout(xaxis1=dict(title="Timestep"), yaxis1=dict(title="% Concentration (M)"),
                              xaxis2=dict(title="Percentage (%)"), yaxis2=dict(title="ORDER<br>of % reaching"))
            pio.write_html(fig, file=folder + "/BraidDiagram_" + title + ".html", auto_open=auto_open)

        return patterns

    elif mode == "surface":

        rnd_id = np.random.randint(0, len(rho_vals))
        patterns = []
        # for each value of rho
        for i, rho in enumerate(rho_vals):

            # compute the diagram
            tau_dyn = out_circ[i, :, :]
            tau_dyn_perc = tau_dyn / tau_dyn[-1, :] * 100

            rxs = [[tau_dyn_perc[:, list(conn.region_labels).index(roi)] for roi in rx] for rx in
                   [rI, rII, rIII, rIV, rV]]
            rxs_avg = np.asarray([np.average(np.asarray(rx), axis=0) for rx in rxs])
            # For each percentage; tell me in what time it was rebased.
            percxRx_TIME = np.asarray([[np.min(np.argwhere(rx >= perc)) for rx in rxs_avg] for perc in range(1, 100)])
            # columns=["rI", "rII", "rIII", "rIV", "rV"])
            # Tell me the rx order based on perc_time
            percxRx_ORDER_braiddiag = np.asarray([np.argsort(np.argsort(row)) for row in percxRx_TIME])
            percxRx_ORDER_braiddiag = percxRx_ORDER_braiddiag.astype(float)

            if rnd_id == i:
                diag = [percxRx_TIME, percxRx_ORDER_braiddiag]

            # Take care with equalities in perc_time that are sorted randomly
            for i, row in enumerate(percxRx_TIME):
                if len(set(row)) < 5:
                    for val in set(row):
                        if len(row[row == val]) > 1:
                            percxRx_ORDER_braiddiag[i, row == val] = \
                                np.tile(np.average(percxRx_ORDER_braiddiag[i, row == val]), (len(row[row == val])))

            # compact the diagram in a line with categorical patters
            patterns.append([str(row) for row in percxRx_ORDER_braiddiag])

        rho_patts = sorted(list(set([p for patt in patterns for p in set(patt)])))

        patt_int = np.asarray(patterns)
        corresp = []
        for i, pattern in enumerate(rho_patts):
            patt_int[np.where(patt_int == rho_patts[i])] = i
            corresp.append([i, pattern])

        patt_int = patt_int.astype(int)

        # Plotting
        cmap = px.colors.qualitative.Set2
        fig = make_subplots(rows=2, cols=2, specs=[[{}, {"rowspan": 2}], [{}, {}]],
                            subplot_titles=["Temporal dynamics - RANDOM rho (" + str(rho_vals[rnd_id]) + ")",
                                            "Braid surface",
                                            "Braid diagram- RANDOM rho (" + str(rho_vals[rnd_id]) + ")"])

        fig.add_trace(go.Heatmap(x=list(range(1, 100)), y=rho_vals, z=patt_int, colorscale="Turbo"), row=1, col=2)

        ## Add a random diagram as example
        percxRx_TIME, percxRx_ORDER_braiddiag = diag
        for i in range(percxRx_ORDER_braiddiag.shape[1]):
            fig.add_trace(
                go.Scatter(x=percxRx_TIME[:, i], y=list(range(1, 100)), name=r_names[i], legendgroup="r" + str(i + 1),
                           showlegend=False, line=dict(color=cmap[i])), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(1, 100)), y=percxRx_ORDER_braiddiag[:, i], name=r_names[i],
                                     legendgroup="r" + str(i + 1), showlegend=True, line=dict(color=cmap[i])), row=2,
                          col=1)

        fig.update_layout(xaxis1=dict(title="Timestep"), yaxis1=dict(title="% Concentration (M)"),
                          xaxis2=dict(title="Percentage (%)"), yaxis2=dict(title="rho (diffusion factor)", type="log"),
                          xaxis3=dict(title="Percentage (%)"), yaxis3=dict(title="ORDER<br>of % reaching"),
                          legend=dict(orientation="h"))
        pio.write_html(fig, file=folder + "/BraidSurface_" + title + ".html", auto_open=auto_open)

        corresp = pd.DataFrame(corresp, columns=["id", "pattern"])

        return corresp


def correlations_v2(output, conn, scatter="REL_simple", band="3-alpha", title="new", folder="figures", auto_open=True):
    """
    Calculate and return both absolute and relative correlations
    for all the interesting variables (PET-tau, PET-ab, FC, FC-dmn, FC-theory, SC).
    PET correlations are calculated against toxic simulated burden.

    Then plot everything if asked for.

    :param output: full Model output
    :param scatter: None if scatter is not desired; [REL | ABS] for increase relative to previous stage (or timepoint)
    or increase relative initial value; [simple | multstages | color] for only one scatter, one scater
    per stage, one scatter with color per regions
    :param band: for empirical FC correlations
    :param title:
    :return:
    """

    # Here we wanna get a simplify array with (n_refcond, time) containing correlation values
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

    adni_refgroups = ["CN", "SMC", "EMCI", "LMCI", "AD"]
    c3n_refgroups = ["HC-fam", "FAM", "QSM", "MCI", "MCI-conv"]

    CORRs = []
    ## A. Compute Correlations
    ## 1. ABSOLUTE CORRELATIONS (for PET cumulative change; for FC & SC full matrices comparisons
    print("Working on ABSOLUTE correlations: PET", end=", ")
    # 1.1 PET
    #    ADNI PET DATA       ##########
    ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupREL_2CN.csv", index_col=0)

    # Check label order
    PETlabs = list(ADNI_AVG.columns[12:])
    PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]

    # loop over refgroups: ["CN", "SMC", "EMCI", "LMCI", "AD"]

    corr_groups = []
    df_corr = pd.DataFrame()
    for j, group in enumerate(adni_refgroups):

        transition = adni_refgroups[j] + "_rel2CN"

        AB_emp = np.squeeze(
            np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Group"] == group)].iloc[:, 12:]))
        AB_emp = AB_emp[PET_idx]

        TAU_emp = np.squeeze(
            np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Group"] == group)].iloc[:, 12:]))
        TAU_emp = TAU_emp[PET_idx]

        # Calculate the derivatives on the simulated data
        dABt = np.asarray(
            [np.asarray(output[1])[i, 1, :] - np.asarray(output[1])[0, 1, :] for i in range(len(output[1]) - 1)])
        dTAUt = np.asarray(
            [np.asarray(output[1])[i, 3, :] - np.asarray(output[1])[0, 3, :] for i in range(len(output[1]) - 1)])

        corr_group_t = []
        for i in range(len(dABt)):
            # Correlate increase in empirical with derivatives in simulated
            corr_group_t.append([np.corrcoef(AB_emp, dABt[i, :])[0, 1], np.corrcoef(TAU_emp, dTAUt[i, :])[0, 1]])

            if "ABS" in scatter:
                # Create dataframe to plot
                df_corr = df_corr.append(
                    pd.DataFrame(
                        [["rel2CN"] * len(dABt[i, :]), [output[0][i]] * len(dABt[i, :]), [transition] * len(dABt[i, :]),
                         ["ABt"] * len(dABt[i, :]), dABt[i, :], AB_emp, conn.region_labels]).transpose())

                df_corr = df_corr.append(
                    pd.DataFrame(
                        [["rel2CN"] * len(dTAUt[i, :]), [output[0][i]] * len(dTAUt[i, :]), [transition] * len(dTAUt[i, :]),
                         ["TAUt"] * len(dTAUt[i, :]), dTAUt[i, :], TAU_emp, conn.region_labels]).transpose())

        corr_groups.append(corr_group_t)
    CORRs.append(corr_groups)

    print("FC", end=", ")
    # 1.2 FC
    # Define regions implicated in Functional analysis: not considering subcortical ROIs
    #  Load FC labels, transform to SC format; check if match SC.
    FClabs = list(np.loadtxt(data_folder + "FCavg_matrices/" + c3n_refgroups[0] + "_roi_labels.txt", dtype=str))
    FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    FC_cortex_idx = [FClabs.index(roi) for roi in conn.region_labels[conn.cortical]]  # find indexes in FClabs that matches cortical_rois

    SClabs = list(conn.region_labels)
    SC_cortex_idx = [SClabs.index(roi) for roi in conn.region_labels[conn.cortical]]

    shortout = [np.array([out[2] for i, out in enumerate(output[2]) if len(out) > 1]),
                np.array([output[0][i] for i, out in enumerate(output[2]) if len(out) > 1])]

    corr_groups = []
    for group in c3n_refgroups:
        plv_emp = np.loadtxt(data_folder + "FCavg_matrices/" + group + "_" + band + "_plv_avg.txt", delimiter=',')[:,
                  FC_cortex_idx][
            FC_cortex_idx]

        t1 = np.zeros(shape=(2, len(plv_emp) ** 2 // 2 - len(plv_emp) // 2))
        t1[0, :] = plv_emp[np.triu_indices(len(plv_emp), 1)]

        corr_group_t = []
        for plv_sim in shortout[0]:
            t1[1, :] = plv_sim[np.triu_indices(len(plv_emp), 1)]
            corr_group_t.append(np.corrcoef(t1)[0, 1])

        corr_groups.append(corr_group_t)

    CORRs.append([corr_groups, shortout[1]])
    #
    # print("SC", end=".\n\n")
    # # 1.3 SC
    # corr_groups = []
    # for group in c3n_refgroups:
    #     conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + group + "_aparc_aseg-mni_09c.zip")
    #     conn.weights = conn.scaled_weights(mode="tract")
    #
    #
    #     t1 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
    #     t1[0, :] = conn.weights[np.triu_indices(len(conn.region_labels), 1)]
    #
    #     # correlate and save
    #     corr_group_t = []
    #     for i in range(len(output[2])):
    #         t1[1, :] = output[2][i][0][np.triu_indices(len(conn.region_labels), 1)]
    #         corr_group_t.append(np.corrcoef(t1)[0, 1])
    #
    #     corr_groups.append(corr_group_t)
    #
    # CORRs.append(corr_groups)
    #
    # ## 2. RELATIVE CORRELATIONS
    # print("Working on RELATIVE correlations: PET", end=", ")
    #
    # # 2.1 PET
    # #    ADNI PET DATA       ##########
    # ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupREL_2PrevStage.csv", index_col=0)
    #
    # # Check label order
    # conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/HC-fam_aparc_aseg-mni_09c.zip")
    # PETlabs = list(ADNI_AVG.columns[7:])
    # PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]
    #
    # corr_groups = []
    # for j, group in enumerate(adni_refgroups[:-1]):
    #
    #     transition = group + "-" + adni_refgroups[j + 1]
    #
    #     AB_emp = np.squeeze(
    #         np.asarray(ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV45") & (ADNI_AVG["Transition"] == transition)].iloc[:, 7:]))
    #     AB_emp = AB_emp[PET_idx]
    #
    #     TAU_emp = np.squeeze(np.asarray(
    #         ADNI_AVG.loc[(ADNI_AVG["PET"] == "AV1451") & (ADNI_AVG["Transition"] == transition)].iloc[:, 7:]))
    #     TAU_emp = TAU_emp[PET_idx]
    #
    #     # Calculate the derivatives on the simulated data
    #     dABt = np.asarray(
    #         [np.asarray(output[1])[i + 1, 1, :] - np.asarray(output[1])[i, 1, :] for i in range(len(output[1]) - 1)])
    #     dTAUt = np.asarray(
    #         [np.asarray(output[1])[i + 1, 3, :] - np.asarray(output[1])[i, 3, :] for i in range(len(output[1]) - 1)])
    #
    #     corr_group_t = []
    #     for i in range(len(dABt)):
    #         # Correlate increase in empirical with derivatives in simulated
    #         corr_group_t.append([np.corrcoef(AB_emp, dABt[i, :])[0, 1], np.corrcoef(TAU_emp, dTAUt[i, :])[0, 1]])
    #
    #         if "REL" in scatter:
    #             # Create dataframe to plot
    #             df_corr = df_corr.append(
    #                 pd.DataFrame(
    #                     [["rel2PS"] * len(dABt[i, :]), [output[0][i]] * len(dABt[i, :]), [transition] * len(dABt[i, :]),
    #                      ["ABt"] * len(dABt[i, :]), dABt[i, :], AB_emp, conn.region_labels]).transpose())
    #
    #             df_corr = df_corr.append(
    #                 pd.DataFrame(
    #                     [["rel2PS"] * len(dTAUt[i, :]), [output[0][i]] * len(dTAUt[i, :]), [transition] * len(dTAUt[i, :]),
    #                      ["TAUt"] * len(dTAUt[i, :]), dTAUt[i, :], TAU_emp, conn.region_labels]).transpose())
    #
    #     corr_groups.append(corr_group_t)
    #
    # CORRs.append(corr_groups)
    #
    #
    # print("FC", end=", ")
    # # 2.2 FC
    # #  Load FC labels, transform to SC format; check if match SC.
    # FClabs = list(np.loadtxt(data_folder + "FCavg_matrices/" + c3n_refgroups[0] + "_roi_labels.txt", dtype=str))
    # FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    # FC_cortex_idx = [FClabs.index(roi) for roi in cortical_rois]  # find indexes in FClabs that matches cortical_rois
    #
    # shortout_rel = [np.array([out[3] for i, out in enumerate(output[2]) if len(out) > 1]),
    #                 np.array([output[0][i] for i, out in enumerate(output[2]) if len(out) > 1])]
    #
    # corr_groups = []
    # for j, group in enumerate(c3n_refgroups[:-1]):
    #
    #     plv_emp0 = np.loadtxt(data_folder + "FCavg_matrices/" + group + "_" + band + "_plv_avg.txt", delimiter=',')[:,
    #                FC_cortex_idx][
    #         FC_cortex_idx]
    #
    #     plv_emp1 = \
    #     np.loadtxt(data_folder + "FCavg_matrices/" + c3n_refgroups[j + 1] + "_" + band + "_plv_avg.txt", delimiter=',')[
    #     :, FC_cortex_idx][
    #         FC_cortex_idx]
    #
    #     dplv_emp = plv_emp1 - plv_emp0
    #
    #     t1 = np.zeros(shape=(2, len(dplv_emp) ** 2 // 2 - len(dplv_emp) // 2))
    #     t1[0, :] = dplv_emp[np.triu_indices(len(dplv_emp), 1)]
    #
    #     # TODO compare with derivatives
    #     corr_group_t = []
    #     for i, plv_sim0 in enumerate(shortout_rel[0][:-1]):
    #         dplv_sim = shortout_rel[0][i + 1] - plv_sim0
    #
    #         t1[1, :] = dplv_sim[np.triu_indices(len(dplv_emp), 1)]
    #         corr_group_t.append(np.corrcoef(t1)[0, 1])
    #
    #     corr_groups.append(corr_group_t)
    #
    # CORRs.append([corr_groups, shortout[1]])
    #
    # # # Do it for DMN
    # # corr_groups = []
    # # for j, group in enumerate(c3n_refgroups[:-1]):
    # #
    # #     plv_emp0 = np.loadtxt(data_folder + "FCavg_matrices/" + group + "_" + band + "_plv_avg.txt", delimiter=',')[:,
    # #                FC_dmn_idx][
    # #         FC_dmn_idx]
    # #
    # #     plv_emp1 = \
    # #     np.loadtxt(data_folder + "FCavg_matrices/" + c3n_refgroups[j + 1] + "_" + band + "_plv_avg.txt", delimiter=',')[
    # #     :,
    # #     FC_dmn_idx][
    # #         FC_dmn_idx]
    # #
    # #     dplv_emp = plv_emp1 - plv_emp0
    # #
    # #     t1 = np.zeros(shape=(2, len(dplv_emp) ** 2 // 2 - len(dplv_emp) // 2))
    # #     t1[0, :] = dplv_emp[np.triu_indices(len(dplv_emp), 1)]
    # #
    # #     # TODO compare with derivatives
    # #     corr_group_t = []
    # #     for i, plv_sim0 in enumerate(shortout_rel[0][:-1]):
    # #         plv_sim0 = plv_sim0[:, FC_dmn_idx][FC_dmn_idx]
    # #         plv_sim1 = shortout_rel[0][i + 1][:, FC_dmn_idx][FC_dmn_idx]
    # #         dplv_sim = plv_sim1 - plv_sim0
    # #
    # #         t1[1, :] = dplv_sim[np.triu_indices(len(dplv_emp), 1)]
    # #         corr_group_t.append(np.corrcoef(t1)[0, 1])
    # #
    # #     corr_groups.append(corr_group_t)
    # #
    # # CORRs.append([corr_groups, shortout[1]])
    #
    #
    #
    # print("SC", end=".\n\n")
    # # 2.3 SC
    # corr_groups = []
    # for i, group in enumerate(c3n_refgroups[:-1]):
    #     conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + group + "_aparc_aseg-mni_09c.zip")
    #     weights0 = conn.scaled_weights(mode="tract")
    #
    #     conn = connectivity.Connectivity.from_file(
    #         data_folder + "SC_matrices/" + c3n_refgroups[i + 1] + "_aparc_aseg-mni_09c.zip")
    #     weights1 = conn.scaled_weights(mode="tract")
    #
    #     dweights_emp = weights1 - weights0
    #
    #     t1 = np.zeros(shape=(2, len(conn.region_labels) ** 2 // 2 - len(conn.region_labels) // 2))
    #     t1[0, :] = dweights_emp[np.triu_indices(len(conn.region_labels), 1)]
    #
    #     # correlate and save
    #     corr_group_t = []
    #     for i in range(len(output[2]))[:-1]:
    #         dweights_sim = output[2][i + 1][0] - output[2][i][0]
    #
    #         t1[1, :] = dweights_sim[np.triu_indices(len(conn.region_labels), 1)]
    #         corr_group_t.append(np.corrcoef(t1)[0, 1])
    #
    #     corr_groups.append(corr_group_t)
    #
    # CORRs.append(corr_groups)

    ## B. PLOTTING (if needed)
    cmap_adni = px.colors.sample_colorscale("Phase", np.arange(0, 1, 1/len(adni_refgroups)))
    cmap_c3n = px.colors.sample_colorscale("Phase", np.insert(np.arange(0, 1, 1/len(adni_refgroups)), 1, 0.1))
    op_p, op_s = 0.4, 0.9

    fig_corr = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        column_titles=["ADNI PET", "C3N FC<br>(alpha [8-12Hz])", "C3N SC"])


    # 3.1 Add Absolute lines
    for i, group in enumerate(adni_refgroups):
        fig_corr.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[0])[i, :, 0], mode="lines", name=group + " _AB",
                                 legendgroup="corradni", opacity=op_p,
                                 line=dict(color=cmap_adni[i], width=3), showlegend=True), row=1, col=1)

        fig_corr.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[0])[i, :, 1], mode="lines", name=group + " _TAU",
                                 legendgroup="corradni", opacity=op_s,
                                 line=dict(color=cmap_adni[i], width=2, dash="dash"),  # visible="legendonly",
                                 showlegend=True), row=1, col=1)

    for i, group in enumerate(c3n_refgroups):
        fig_corr.add_trace(go.Scatter(x=CORRs[1][1], y=np.array(CORRs[1][0][i]), mode="lines", name=group + " _FC",
                                 legendgroup="corrc3n", opacity=op_p,
                                 line=dict(color=cmap_c3n[i], width=3), showlegend=True), row=1, col=2)

        # fig_corr.add_trace(go.Scatter(x=CORRs[1][1], y=np.array(CORRs[2][0][i]), mode="lines", name=group + " _FCdmn",
        #                          legendgroup="corrc3n", opacity=op_s,
        #                          line=dict(color=cmap_c3n[i], width=2, dash="dash"), showlegend=True), row=1, col=2)

        # fig_corr.add_trace(
        #     go.Scatter(x=output[0], y=np.array(CORRs[3][i]), mode="lines", name=group + " _SC", legendgroup="corrc3n",
        #                line=dict(color=cmap_c3n[i], width=3), opacity=op_p, showlegend=False), row=1, col=3)


    # # 3.2 Add  Relative lines
    # for i, group in enumerate(adni_refgroups[:-1]):
    #     transition = group + "-" + adni_refgroups[i + 1]
    #     fig_corr.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[4])[i, :, 0], mode="lines", name=transition + " _AB",
    #                              legendgroup="corradnirel", opacity=op_p,
    #                              line=dict(color=cmap_adni[i+1], width=3), showlegend=True), row=2, col=1)
    #
    #     fig_corr.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[4])[i, :, 1], mode="lines", name=transition + " _TAU",
    #                              legendgroup="corradnirel", opacity=op_s,
    #                              line=dict(color=cmap_adni[i+1], width=2, dash="dash"),  # visible="legendonly",
    #                              showlegend=False), row=2, col=1)
    #
    # for i, group in enumerate(c3n_refgroups[:-1]):
    #     transition = group + "-" + c3n_refgroups[i + 1]
    #     fig_corr.add_trace(go.Scatter(x=CORRs[1][1], y=np.array(CORRs[5][0][i]), mode="lines", name=transition + " _FC",
    #                              legendgroup="corrc3nrel", opacity=op_p,
    #                              line=dict(color=cmap_c3n[i+1], width=3), showlegend=True), row=2, col=2)
    #
    #     fig_corr.add_trace(go.Scatter(x=CORRs[1][1], y=np.array(CORRs[6][0][i]), mode="lines", name=transition + " _FCdmn",
    #                              legendgroup="corrc3nrel", opacity=op_s,
    #                              line=dict(color=cmap_c3n[i+1], width=2, dash="dash"), showlegend=True), row=2, col=2)
    #
    #     fig_corr.add_trace(go.Scatter(x=output[0], y=np.array(CORRs[7][i]), mode="lines", name=transition + " _SC",
    #                              legendgroup="corrc3nrel", opacity=op_p,
    #                              line=dict(color=cmap_c3n[i+1], width=3), showlegend=False), row=2, col=3)

    fig_corr.update_layout(template="plotly_white", legend=dict(groupclick="toggleitem"),
                      yaxis1=dict(title="PET Cumulative change (from CN)<br>Pearson's r", range=[-1, 1]),
                      yaxis2=dict(title="FC matrices<br>Pearson's r", range=[-1, 1]),
                      yaxis3=dict(title="SC matrices<br>Pearson's r", range=[-1, 1]),
                      yaxis4=dict(title="PET Relative change (from prev. stage)<br>Pearson's r", range=[-1, 1]), xaxis5=dict(title="Time (years)"),
                      yaxis5=dict(title="FC Relative changes (from prev. stage)<br>Pearson's r", range=[-1, 1]), xaxis6=dict(title="Time (years)"),
                      yaxis6=dict(title="SC Relative changes (from prev. stage)<br>Pearson's r", range=[-1, 1]), xaxis7=dict(title="Time (years)"))

    pio.write_html(fig_corr, folder + "/CORRELATIONS_" + title + ".html", auto_open=auto_open, auto_play=False)


    # plot scatter
    if "ABS" in scatter:

        add_space = 0.005
        print("CORRELATIONS  _Plotting animation - wait patiently")
        df_corr.columns = ["mode", "time", "group", "pet", "sim", "emp", "roi"]

        df_sub = df_corr.loc[df_corr["mode"] == "rel2CN"]

        if "mult" in scatter:
            fig_sc = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", facet_col="group", animation_frame="time", hover_name="roi")

        elif "color" in scatter:
            df_sub = df_sub.loc[df_sub["group"] == "AD_rel2CN"]
            fig_sc = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time", hover_name="roi", color="roi")

        elif "simple" in scatter:
            df_sub = df_sub.loc[df_sub["group"] == "AD_rel2CN"]
            fig_sc = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time", hover_name="roi")

        fig_sc.update_layout(title=title + " | Increased Protein concentration (emp-sim) relative to: CN (emp), and t=0 (sim)", template="plotly_white",
                          yaxis1=dict(range=[min(df_sub["sim"].values) - add_space, max(df_sub["sim"].values) + add_space]))
        pio.write_html(fig_sc, folder + "/CorrScatter_ABS-" + scatter + "-" + title + ".html",
                       auto_open=auto_open, auto_play=False)

    if "REL" in scatter:

        add_space = 0.001
        print("CORRELATIONS  _Plotting animation - wait patiently")
        df_corr.columns = ["mode", "time", "transition", "pet", "sim", "emp", "roi"]

        df_sub = df_corr.loc[df_corr["mode"] == "rel2PS"]

        if "mult" in scatter:
            fig_sc = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", facet_col="transition", animation_frame="time",
                             hover_name="roi")

        elif "color" in scatter:
            df_sub = df_sub.loc[df_sub["transition"] == "LMCI-AD"]
            fig_sc = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time", hover_name="roi", color="roi", )

        elif "simple" in scatter:
            df_sub = df_sub.loc[df_sub["transition"] == "LMCI-AD"]
            fig_sc = px.scatter(df_sub, x="emp", y="sim", facet_row="pet", animation_frame="time", hover_name="roi")

        fig_sc.update_layout(title=title + " | Increased Protein concentration relative to: previous AD stage (emp), and t(-1) (sim)", template="plotly_white", yaxis1=dict(
            range=[min(df_sub["sim"].values) - add_space, max(df_sub["sim"].values) + add_space]))
        pio.write_html(fig_sc, folder + "/CorrScatter_REL-" + scatter + "-" + title + ".html",
                       auto_open=auto_open, auto_play=False)

    return CORRs


def animate_propagation_v4(output, corrs, refgroups, reftype, conn, timeref=True, title="", folder="figures", auto_open=True):

    adni_refgroups = ["CN", "SMC", "EMCI", "LMCI", "AD"]
    c3n_refgroups = ["HC-fam", "FAM", "QSM", "MCI", "MCI-conv"]

    # Create text labels per ROI
    hovertext3d = [["<b>" + roi + "</b><br>"
                    + str(round(output[1][ii][0, i], 5)) + "(M) a-beta<br>"
                    + str(round(output[1][ii][1, i], 5)) + "(M) a-beta toxic <br>"
                    + str(round(output[1][ii][2, i], 5)) + "(M) pTau <br>"
                    + str(round(output[1][ii][3, i], 5)) + "(M) pTau toxic <br>"
                    for i, roi in enumerate(conn.region_labels)] for ii, t in enumerate(output[0])]

    sz_ab, sz_t = 25, 10  # Different sizes for AB and pT nodes

    if any(len(out) > 1 for out in output[2]):

        shortout = [np.array([np.average(out[1]) for i, out in enumerate(output[2]) if len(out) > 1]),
                    np.array([np.average(out[2]) for i, out in enumerate(output[2]) if len(out) > 1]),
                    np.array([output[0][i] for i, out in enumerate(output[2]) if len(out) > 1])]

        ## ADD INITIAL TRACE for 3dBrain - t0
        fig = make_subplots(rows=2, cols=3,
                            specs=[[{"rowspan": 2, "type": "surface"}, {}, {}], [{}, {}, {"secondary_y": True}]],
                            column_widths=[0.5, 0.25, 0.25], shared_xaxes=True, horizontal_spacing=0.075,
                            subplot_titles=(
                            ['<b>Protein accumulation dynamics</b>', '', '', '', 'Correlations (emp-sim)', '']))

        # Add trace for AB + ABt
        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="AB", showlegend=True,
                                   legendgroup="AB",
                                   marker=dict(size=(np.abs(output[1][0][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                               cmax=0.5, cmin=-0.25,
                                               color=np.abs(output[1][0][1, :]) / np.abs(output[1][0][0, :]),
                                               opacity=0.5,
                                               line=dict(color="grey", width=1), colorscale="YlOrBr")), row=1, col=1)

        # Add trace for TAU + TAUt
        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="TAU", showlegend=True,
                                   legendgroup="TAU",
                                   marker=dict(size=(np.abs(output[1][0][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                               cmax=0.5, cmin=-0.25,
                                               color=np.abs(output[1][0][3, :]) / np.abs(output[1][0][2, :]), opacity=1,
                                               line=dict(color="grey", width=1), colorscale="BuPu", symbol="diamond")),
                      row=1, col=1)

        ## ADD INITIAL TRACE for lines
        sim_pet_avg = np.average(np.asarray(output[1]), axis=2)

        if timeref:
            # Add dynamic reference - t0
            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[-0.15, 1.15], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=1, col=2)

            min_r, max_r = np.min(corrs) - 0.15, np.max(corrs) + 0.15
            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[min_r, max_r], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=2, col=2)

            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[0, 25], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=1, col=3)

            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[0, max(shortout[1])], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=2, col=3)

        # Add static lines - PET proteins concentrations
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 0], mode="lines", name="AB", legendgroup="AB",
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7,
                                 showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 1], mode="lines", name="AB toxic", legendgroup="AB",
                                 line=dict(color=px.colors.sequential.YlOrBr[5], width=3), opacity=0.7,
                                 showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 2], mode="lines", name="TAU", legendgroup="TAU",
                                 line=dict(color=px.colors.sequential.BuPu[3], width=3), opacity=0.7, showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 3], mode="lines", name="TAU toxic", legendgroup="TAU",
                                 line=dict(color=px.colors.sequential.BuPu[5], width=3), opacity=0.7, showlegend=True),
                      row=1, col=2)

        # Add static lines - data correlations
        cmap_adni = px.colors.sample_colorscale("Phase", np.arange(0, 1, 1 / len(adni_refgroups)))
        cmap_c3n = px.colors.sample_colorscale("Phase", np.insert(np.arange(0, 1, 1 / len(adni_refgroups)), 1, 0.1))
        op_p, op_s = 0.4, 0.9
        for ii, group in enumerate(refgroups):
            if "PET" in reftype:
                c = ii + 1 if "rel" in reftype else ii
                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 0], mode="lines", name=group + " - AB",
                                         legendgroup="corrAB", opacity=op_p,
                                         line=dict(color=cmap_adni[c], width=3), showlegend=True), row=2, col=2)

                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 1], mode="lines", name=group + " - rTAU",
                                         legendgroup="corrTAU", opacity=op_s,
                                         line=dict(color=cmap_adni[c], width=2, dash="dash"),  # visible="legendonly",
                                         showlegend=True), row=2, col=2)

            else:
                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :], mode="lines", name=group + " - r" + reftype,
                                         legendgroup="corr", opacity=op_s,
                                         line=dict(color=cmap_c3n[c], width=3), showlegend=True), row=2, col=2)

        # Add static lines - parameter values
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 6], mode="lines", name="He", legendgroup="params",
                                 line=dict(width=3), opacity=0.7, showlegend=True),
                      row=1, col=3)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 7], mode="lines", name="Hi", legendgroup="params",
                                 line=dict(width=3), opacity=0.7, showlegend=True),
                      row=1, col=3)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 8], mode="lines", name="tau_e", legendgroup="params",
                                 line=dict(width=3), opacity=0.7, showlegend=True),
                      row=1, col=3)

        # Add static lines - average spectral properties
        fig.add_trace(go.Scatter(x=shortout[2], y=shortout[1], mode="lines", name="Power", legendgroup="spectra",
                                 line=dict(width=4, color="lawngreen"), opacity=0.7, showlegend=True),
                      row=2, col=3)

        fig.add_trace(go.Scatter(x=shortout[2], y=shortout[0], mode="lines", name="Frequency", legendgroup="spectra",
                                 line=dict(width=2, color="mediumvioletred"), opacity=0.7, showlegend=True),
                      row=2, col=3, secondary_y=True)

        ## ADD FRAMES - t[1:end]
        if timeref:

            fig.update(frames=[go.Frame(data=[
                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=(np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                         color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=(np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                         color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),

                go.Scatter(x=[output[0][i], output[0][i]]),
                go.Scatter(x=[output[0][i], output[0][i]]),
                go.Scatter(x=[output[0][i], output[0][i]]),
                go.Scatter(x=[output[0][i], output[0][i]])
            ],
                traces=[0, 1, 2, 3, 4, 5], name=str(i)) for i, t in enumerate(output[0])])
        else:
            fig.update(frames=[go.Frame(data=[
                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :]) * sz_ab,
                                         color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :]) * sz_t,
                                         color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),
            ],
                traces=[0, 1], name=str(i)) for i, t in enumerate(output[0])])

        # CONTROLS : Add sliders and buttons
        fig.update_layout(
            template="plotly_white", legend=dict(x=1, y=0.55, tracegroupgap=10, groupclick="toggleitem"),
            scene=dict(xaxis=dict(title="Sagital axis<br>(L-R)"),
                       yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                       zaxis=dict(title="Horizontal axis<br>(inf-sup)")),
            xaxis1=dict(title="Time (Years)"), xaxis2=dict(title="Time (Years)"),
            xaxis4=dict(title="Time (Years)"), xaxis5=dict(title="Time (Years)"),
            yaxis1=dict(title="Concentration (M)"), yaxis2=dict(title="param value"),
            yaxis4=dict(title="Pearson's r"), yaxis5=dict(title="Power (dB)"),
            yaxis6=dict(title="Frequency (Hz)", range=[0, 14]),

            sliders=[dict(
                steps=[
                    dict(method='animate', args=[[str(i)], dict(mode="immediate", frame=dict(duration=250, redraw=True,
                                                                                             easing="cubic-in-out"),
                                                                transition=dict(duration=0))], label=str(t)) for i, t
                    in enumerate(output[0])],
                transition=dict(duration=0), x=0.15, xanchor="left", y=1.4,
                currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
                len=0.8, tickcolor="white")],

            updatemenus=[dict(type="buttons", showactive=False, y=1.35, x=0, xanchor="left",
                              buttons=[
                                  dict(label="Play", method="animate",
                                       args=[None,
                                             dict(frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                                  transition=dict(duration=0),
                                                  fromcurrent=True, mode='immediate')]),
                                  dict(label="Pause", method="animate",
                                       args=[[None],
                                             dict(frame=dict(duration=250, redraw=False, easing="cubic-in-out"),
                                                  transition=dict(duration=0),
                                                  mode="immediate")])])])

        pio.write_html(fig, file=folder + "/ProteinPropagation_&corr" + reftype + "_" + title + ".html", auto_open=auto_open,
                       auto_play=False)

    else:

        ## ADD INITIAL TRACE for 3dBrain - t0
        fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2, "type": "surface"}, {}], [{}, {}]],
                            column_widths=[0.6, 0.4], shared_xaxes=True,
                            subplot_titles=(
                                ['<b>Protein accumulation dynamics</b> ' + title + " - ref: " + reftype, '', '',
                                 'Correlations (emp-sim)', ]))

        # Add trace for AB + ABt
        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="AB", showlegend=True,
                                   legendgroup="AB",
                                   marker=dict(size=(np.abs(output[1][0][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                               cmax=0.5, cmin=-0.25,
                                               color=np.abs(output[1][0][1, :]) / np.abs(output[1][0][0, :]),
                                               opacity=0.5,
                                               line=dict(color="grey", width=1), colorscale="YlOrBr")), row=1, col=1)

        # Add trace for TAU + TAUt
        fig.add_trace(go.Scatter3d(x=conn.centres[:, 0], y=conn.centres[:, 1], z=conn.centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="TAU", showlegend=True,
                                   legendgroup="TAU",
                                   marker=dict(size=(np.abs(output[1][0][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                               cmax=0.5, cmin=-0.25,
                                               color=np.abs(output[1][0][3, :]) / np.abs(output[1][0][2, :]), opacity=1,
                                               line=dict(color="grey", width=1), colorscale="BuPu", symbol="diamond")),
                      row=1, col=1)

        ## ADD INITIAL TRACE for lines
        sim_pet_avg = np.average(np.asarray(output[1]), axis=2)

        if timeref:
            # Add dynamic reference - t0
            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[-0.15, 1.15], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=1, col=2)

            min_r, max_r = np.min(corrs) - 0.15, np.max(corrs) + 0.15
            fig.add_trace(
                go.Scatter(x=[output[0][0], output[0][0]], y=[min_r, max_r], mode="lines", legendgroup="timeref",
                           line=dict(color="black", width=1), showlegend=False), row=2, col=2)

        # Add static lines - PET proteins concentrations
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 0], mode="lines", name="AB", legendgroup="AB",
                                 line=dict(color=px.colors.sequential.YlOrBr[3], width=3), opacity=0.7,
                                 showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 1], mode="lines", name="AB toxic", legendgroup="AB",
                                 line=dict(color=px.colors.sequential.YlOrBr[5], width=3), opacity=0.7,
                                 showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 2], mode="lines", name="TAU", legendgroup="TAU",
                                 line=dict(color=px.colors.sequential.BuPu[3], width=3), opacity=0.7, showlegend=True),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=output[0], y=sim_pet_avg[:, 3], mode="lines", name="TAU toxic", legendgroup="TAU",
                                 line=dict(color=px.colors.sequential.BuPu[5], width=3), opacity=0.7, showlegend=True),
                      row=1, col=2)

        # Add static lines - data correlations
        cmap_p = px.colors.qualitative.Pastel2
        cmap_s = px.colors.qualitative.Set2
        for ii, group in enumerate(refgroups):
            if "PET" in reftype:
                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 0], mode="lines", name=group + " - AB",
                                         legendgroup="corrAB",
                                         line=dict(color=cmap_p[ii], width=3), showlegend=True), row=2, col=2)

                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :, 1], mode="lines", name=group + " - rTAU",
                                         legendgroup="corrTAU",
                                         line=dict(color=cmap_s[ii], width=2, dash="dash"),  # visible="legendonly",
                                         showlegend=True), row=2, col=2)

            else:
                fig.add_trace(go.Scatter(x=output[0], y=corrs[ii, :], mode="lines", name=group + " - r" + reftype,
                                         legendgroup="corr",
                                         line=dict(color=cmap_p[ii], width=3), showlegend=True), row=2, col=2)

        # fig.show("browser")

        ## ADD FRAMES - t[1:end]
        if timeref:

            fig.update(frames=[go.Frame(data=[
                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=(np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :])) * sz_ab,
                                         color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=(np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :])) * sz_t,
                                         color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),

                go.Scatter(x=[output[0][i], output[0][i]]),
                go.Scatter(x=[output[0][i], output[0][i]])
            ],
                traces=[0, 1, 2, 3], name=str(i)) for i, t in enumerate(output[0])])
        else:
            fig.update(frames=[go.Frame(data=[
                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=np.abs(output[1][i][0, :]) + np.abs(output[1][0][1, :]) * sz_ab,
                                         color=np.abs(output[1][i][1, :]) / np.abs(output[1][0][0, :]))),

                go.Scatter3d(hovertext=hovertext3d[i],
                             marker=dict(size=np.abs(output[1][i][2, :]) + np.abs(output[1][0][3, :]) * sz_t,
                                         color=np.abs(output[1][i][3, :]) / np.abs(output[1][0][2, :]))),
            ],
                traces=[0, 1], name=str(i)) for i, t in enumerate(output[0])])

        # CONTROLS : Add sliders and buttons
        fig.update_layout(
            template="plotly_white", legend=dict(x=1.05, y=0.55, tracegroupgap=40, groupclick="toggleitem"),
            scene=dict(xaxis=dict(title="Sagital axis<br>(L-R)"),
                       yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                       zaxis=dict(title="Horizontal axis<br>(inf-sup)")),
            xaxis1=dict(title="Time (Years)"), xaxis3=dict(title="Time (Years)"),
            yaxis1=dict(title="Concentration (M)"), yaxis3=dict(title="Pearson's corr"),
            sliders=[dict(
                steps=[
                    dict(method='animate', args=[[str(i)], dict(mode="immediate", frame=dict(duration=100, redraw=True,
                                                                                             easing="cubic-in-out"),
                                                                transition=dict(duration=300))], label=str(t)) for i, t
                    in enumerate(output[0])],
                transition=dict(duration=100), x=0.15, xanchor="left", y=1.4,
                currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
                len=0.8, tickcolor="white")],
            updatemenus=[dict(type="buttons", showactive=False, y=1.35, x=0, xanchor="left",
                              buttons=[
                                  dict(label="Play", method="animate",
                                       args=[None,
                                             dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                  transition=dict(duration=300),
                                                  fromcurrent=True, mode='immediate')]),
                                  dict(label="Pause", method="animate",
                                       args=[[None],
                                             dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                  transition=dict(duration=300),
                                                  mode="immediate")])])])

        pio.write_html(fig, file="figures/ProtProp_&corr" + reftype + "_" + title + ".html", auto_open=auto_open)


def g_explore(output, g_sel, param="g", mode="html", folder="figures", auto_open=True):
    n_g = len(g_sel)
    col_titles = [""] + [param + "==" + str(g) for g in g_sel]
    specs = [[{} for g in range(n_g + 1)]] * 3
    id_emp = (n_g + 1) * 2
    sp_titles = ["Empirical" if i == id_emp else "" for i in range((n_g + 1) * 3)]
    fig = make_subplots(rows=3, cols=n_g + 1, specs=specs, row_titles=["signals", "FFT", "FC"],
                        column_titles=col_titles, shared_yaxes=True, subplot_titles=sp_titles)

    for i, g in enumerate(g_sel):

        sl = True if i < 1 else False

        # Unpack output
        _, signals, timepoints, plv, plv_emp, r_plv, regionLabels, simLength, transient = output[i]

        freqs = np.arange(len(timepoints) / 2)
        freqs = freqs / (simLength - transient / 1000)

        cmap = px.colors.qualitative.Plotly
        for ii, signal in enumerate(signals):
            # Timeseries
            fig.add_trace(go.Scatter(x=timepoints[:5000] / 1000, y=signal[:5000], name=regionLabels[ii],
                                     legendgroup=regionLabels[ii],
                                     showlegend=sl, marker_color=cmap[ii % len(cmap)]), row=1, col=i + 2)
            # Spectra
            freqRange = [2, 40]
            fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
            fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
            fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
            fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                     marker_color=cmap[ii % len(cmap)], name=regionLabels[ii],
                                     legendgroup=regionLabels[ii], showlegend=False), row=2, col=i + 2)

        # Functional Connectivity
        fig.add_trace(go.Heatmap(z=plv, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4),
                                 colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=i + 2)

    # empirical FC matrices
    fig.add_trace(go.Heatmap(z=plv_emp, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4), legendgroup="",
                             colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=1)

    w_ = 800 if n_g < 3 else 1000
    fig.update_layout(legend=dict(yanchor="top", y=1.05, tracegroupgap=1),
                      template="plotly_white", height=900, width=w_)

    # Update layout
    for col in range(n_g + 1):  # +1 empirical column
        # first row
        idx = col + 1  # +1 to avoid 0 indexing in python
        if idx > 1:
            fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Time (s)"}
            if idx == 2:
                fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Voltage (mV)"}

        # second row
        idx = 1 * (n_g + 1) + (col + 1)  # +1 to avoid 0 indexing in python
        if idx > 1 + n_g:
            fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Frequency (Hz)"}
            if idx == 3 + n_g:
                fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Power (dB)"}

        # third row
        # idx = 2 * n_g+1 + (col+1)  # +1 to avoid 0 indexing in python
        # fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
        # fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}

    if mode == "html":
        pio.write_html(fig, file=folder + "/g_explore.html", auto_open=auto_open)
    elif mode == "png":
        pio.write_image(fig, file=folder + "/g_explore" + str(time.time()) + ".png", engine="kaleido")
    elif mode == "svg":
        pio.write_image(fig, file=folder + "/g_explore.svg", engine="kaleido")

    elif mode == "inline":
        plotly.offline.iplot(fig)


def animateFC(data, conn, mode="3Dcortex", threshold=0.05, title="new", folder="figures", surrogates=None, auto_open=True):

    # Define regions implicated in Functional analysis: not considering subcortical ROIs
    cortical_rois = ['ctx-lh-bankssts', 'ctx-rh-bankssts', 'ctx-lh-caudalanteriorcingulate',
                     'ctx-rh-caudalanteriorcingulate',
                     'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-rh-cuneus',
                     'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
                     'ctx-lh-fusiform', 'ctx-rh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
                     'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal', 'ctx-lh-insula', 'ctx-rh-insula',
                     'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                     'ctx-rh-lateraloccipital',
                     'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-rh-lingual',
                     'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-middletemporal',
                     'ctx-rh-middletemporal',
                     'ctx-lh-paracentral', 'ctx-rh-paracentral', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
                     'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-rh-parsorbitalis',
                     'ctx-lh-parstriangularis', 'ctx-rh-parstriangularis', 'ctx-lh-pericalcarine',
                     'ctx-rh-pericalcarine',
                     'ctx-lh-postcentral', 'ctx-rh-postcentral', 'ctx-lh-posteriorcingulate',
                     'ctx-rh-posteriorcingulate',
                     'ctx-lh-precentral', 'ctx-rh-precentral', 'ctx-lh-precuneus', 'ctx-rh-precuneus',
                     'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
                     'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
                     'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal', 'ctx-lh-superiorparietal',
                     'ctx-rh-superiorparietal',
                     'ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'ctx-lh-supramarginal',
                     'ctx-rh-supramarginal',
                     'ctx-lh-temporalpole', 'ctx-rh-temporalpole', 'ctx-lh-transversetemporal',
                     'ctx-rh-transversetemporal']
    dmn_rois = [  # ROIs not in Gianlucas set, from cingulum bundle description
        'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
        'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal',
        'ctx-lh-insula', 'ctx-rh-insula',
        'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',  # A6. in Gianlucas DMN
        'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',  # A6. in Gianlucas DMN
        'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate',
        'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',  # A7. in Gianlucas DMN
        'ctx-lh-middletemporal', 'ctx-rh-middletemporal',  # A5. in Gianlucas DMN
        'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal',  # A4. in Gianlucas DMN
        'ctx-lh-superiorparietal', 'ctx-rh-superiorparietal',
        'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',  # A3. in Gianlucas DMN
        'ctx-lh-precuneus', 'ctx-rh-precuneus'  # A1. in Gianlucas DMN
    ]

    # Load SC labels.
    SClabs = list(conn.region_labels)
    if "cortex" in mode:
        SC_idx = [SClabs.index(roi) for roi in cortical_rois]
    elif "dmn" in mode:
        SC_idx = [SClabs.index(roi) for roi in dmn_rois]

    # 3d mode
    if "3D" in mode:

        regionLabels = conn.region_labels[SC_idx]
        weights = conn.weights[:, SC_idx][SC_idx]
        centres = conn.centres[SC_idx, :]

        # fig = make_subplots(rows=2, cols=1, specs=[[{"type": "surface"}],[{}]])
        fig = go.Figure()

        # Edges trace
        ## Filter edges to show: remove low connected nodes via thresholding
        edges_ids = list(combinations([i for i, roi in enumerate(regionLabels)], r=2))
        edges_ids = [(i, j) for i, j in edges_ids if weights[i, j] > threshold]

        ## Define [start, end, None] per coordinate and connection
        edges_x = [elem for sublist in [[centres[i, 0]] + [centres[j, 0]] + [None] for i, j in edges_ids] for elem in sublist]
        edges_y = [elem for sublist in [[centres[i, 1]] + [centres[j, 1]] + [None] for i, j in edges_ids] for elem in sublist]
        edges_z = [elem for sublist in [[centres[i, 2]] + [centres[j, 2]] + [None] for i, j in edges_ids] for elem in sublist]

        ## Define color per connection based on FC changes
        increaseFC = [data[1][i][3] - data[1][0][3] for i, t in enumerate(data[0])]

        if surrogates is not None:
            surrogates = np.array(surrogates)
            increaseFC_norm = [[(increaseFC[ii][i, j] + 1)/2
                                if pg.ttest(x=surrogates[:, i, j], y=increaseFC[ii][i, j])["p-val"].values < 0.05/len(edges_ids)
                                else (0.9 + 1)/2 for i, j in edges_ids] for ii, t in enumerate(data[0])]
        else:
            increaseFC_norm = [[(increaseFC[ii][i, j] + 1)/2 for i, j in edges_ids] for ii, t in enumerate(data[0])]

        edges_color = [[px.colors.sample_colorscale("Jet", e)[0] for e in inc for i in range(3)] for inc in increaseFC_norm]

        fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines", hoverinfo="skip",
                                   line=dict(color=edges_color[0], width=3), opacity=0.6, name="Edges"))

        # Nodes trace
        ## Define size per degree
        degree, size_range = np.sum(weights, axis=1), [8, 25]
        size = ((degree - np.min(degree)) * (size_range[1]-size_range[0]) / (np.max(degree) - np.min(degree))) + size_range[0]

        ## Define color per power
        increasePOW = [data[1][i][2] - data[1][0][2] for i, t in enumerate(data[0])]
        increasePOW_norm = [[((increasePOW[ii][i] - np.min(increasePOW))/(np.max(increasePOW) - np.min(increasePOW)))
                             for i, roi in enumerate(regionLabels)] for ii, t in enumerate(data[0])]
        nodes_color = [[px.colors.sample_colorscale("Jet", e)[0] for e in inc] for inc in increasePOW_norm]

        # Create text labels per ROI
        hovertext3d = [["<b>" + roi + "</b>"
                        "<br>Power (dB) " + str(round(data[1][ii][2][i], 5)) +
                        "<br>Frequency (Hz) " + str(round(data[1][ii][1][i], 5)) +
                        "<br><br>Power increase (dB) " + str(round(data[1][ii][2][i] - data[1][0][2][i], 5))
                        for i, roi in enumerate(regionLabels)] for ii, t in enumerate(data[0])]

        fig.add_trace(go.Scatter3d(x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], hoverinfo="text",
                                   hovertext=hovertext3d[0], mode="markers", name="Nodes",
                                   marker=dict(size=size, color=nodes_color[0], opacity=1, line=dict(color="gray", width=2))))

        fig = addpial(fig, mode="surface", opacity=0.1)

        # Update frames
        fig.update(frames=[go.Frame(data=[
            go.Scatter3d(line=dict(color=edges_color[i])),
            go.Scatter3d(hovertext=hovertext3d[i], marker=dict(color=nodes_color[i]))],
            traces=[0, 1], name=str(t)) for i, t in enumerate(data[0])])

        # CONTROLS : Add sliders and buttons
        fig.update_layout(
            template="plotly_white", legend=dict(x=0.8, y=0.5),
            scene=dict(xaxis=dict(title="Sagital axis<br>(R-L)"),
                       yaxis=dict(title="Coronal axis<br>(pos-ant)"),
                       zaxis=dict(title="Horizontal axis<br>(sup-inf)")),
            sliders=[dict(
                steps=[
                    dict(method='animate', args=[[str(t)], dict(mode="immediate", frame=dict(duration=250, redraw=True,
                                                                                             easing="cubic-in-out"),
                                                                transition=dict(duration=0))], label=str(t)) for i, t
                    in enumerate(data[0])],
                transition=dict(duration=0), x=0.15, xanchor="left", y=1.1,
                currentvalue=dict(font=dict(size=15), prefix="Time (years) - ", visible=True, xanchor="right"),
                len=0.8, tickcolor="white")],
            updatemenus=[dict(type="buttons", showactive=False, y=1.05, x=0, xanchor="left",
                              buttons=[
                                  dict(label="Play", method="animate",
                                       args=[None,
                                             dict(frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                                  transition=dict(duration=0),
                                                  fromcurrent=True, mode='immediate')]),
                                  dict(label="Pause", method="animate",
                                       args=[[None],
                                             dict(frame=dict(duration=250, redraw=False, easing="cubic-in-out"),
                                                  transition=dict(duration=0),
                                                  mode="immediate")])])])

        pio.write_html(fig, file=folder + "/IncreaseFC_3Dbrain_" + title + ".html", auto_open=auto_open, auto_play=False)


def surrogatesFC(n, subj, conn, model, g, s, simLength, params=None, params_vCC=None):

    if params:

        He, Hi, taue, taui = params

        title_surrogates = "surrogates_v3-n%i_subj%s_g%is%0.1f_He%0.2fHi%0.2ftaue%0.2ftaui%0.2f_t%i.pkl"\
                           % (n, subj, g, s, He, Hi, taue, taui, simLength)

        if os.path.isfile("surrogates_fc/" + title_surrogates):
            with open("surrogates_fc/" + title_surrogates, "rb") as f:
                surrogates = pickle.load(f)

        else:
            surrogates = []
            for i in range(n):
                print("WE ARE ON SURROGATE: %i" % i, end="\r")
                _, _, _, plv, _, _, _, _, _, _ = \
                    simulate_v3(subj, conn.weights, model, g, s, p_th=0.1085, sigma=0, sv=[He, Hi, taue, taui], t=simLength)
                surrogates.append(plv)

            with open("surrogates_fc/" + title_surrogates, 'wb') as f:
                pickle.dump(surrogates, f)

    elif params_vCC:

        He, Cee, Cie = params_vCC

        title_surrogates = "surrogates_v3-n%i_subj%s_g%is%0.1f_He%0.2Cee%0.2fCie%0.2f_t%i.pkl" % (
        n, subj, g, s, He, Cee, Cie, simLength)

        if os.path.isfile("surrogates_fc/" + title_surrogates):
            with open("surrogates_fc/" + title_surrogates, "rb") as f:
                surrogates = pickle.load(f)

        else:
            surrogates = []
            for i in range(n):
                print("WE ARE ON SURROGATE: %i" % i, end="\r")
                _, _, _, plv, _, _, _, _, _, _ = \
                    simulate_v2(subj, conn.weights, model, g, s, p_th=0.1085, sigma=0, sv=[Cee, Cie], t=simLength)
                surrogates.append(plv)

            with open("surrogates_fc/" + title_surrogates, 'wb') as f:
                pickle.dump(surrogates, f)

    return surrogates




