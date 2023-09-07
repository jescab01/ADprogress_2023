
import time
import numpy as np
import pandas as pd
import os
import pickle

from tvb.simulator.lab import connectivity
from ADpg_functions import circApproach, CircularADpgModel_vCC, paramtraj_in3D, correlations_v2, braidPlot


## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys

    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.mixes import timeseries_spectra

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    data_folder = wd + "ADprogress_data/"

#     1.  PREPARE EMPIRICAL DATA      #########################
#   STRUCTURAL CONNECTIVITY   #########
#  Define structure through which the proteins will spread;
#  Not necessarily the same than the one used to simulate activity.
subj, g, s, sigma = "HC-fam", 25, 20, 0.022
conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices_old21.03.22/" + subj + "_aparc_aseg-mni_09c.zip")
conn.weights = conn.scaled_weights(mode="tract")

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
conn.cortical = np.array([True if roi in cortical_rois else False for roi in conn.region_labels])

## TO USE JUST CB
useCB = True
cb_rois = [  # ROIs not in Gianlucas set, from cingulum bundle description
    'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal',
    'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal',
    'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate',
    'ctx-lh-frontalpole', 'ctx-rh-frontalpole',

    'ctx-lh-rostralmiddlefrontal', 'ctx-rh-rostralmiddlefrontal',
    'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal',
    'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal',

    'ctx-lh-insula', 'ctx-rh-insula',
    'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate',
    'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate',
    'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate',
    'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',

    'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal',

    'ctx-lh-superiorparietal', 'ctx-rh-superiorparietal',
    'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',

    'ctx-lh-precuneus', 'ctx-rh-precuneus',
    'Left-Hippocampus', 'Right-Hippocampus',
    'Left-Thalamus', 'Right-Thalamus',
    'Left-Amygdala', 'Right-Amygdala',
    'ctx-lh-entorhinal', 'ctx-rh-entorhinal'  # seed for TAUt
]
if useCB:
    # load SC labels.
    SClabs = list(conn.region_labels)
    SC_dmn_idx = [SClabs.index(roi) for roi in cb_rois]

    # #  Load FC labels, transform to SC format; check if match SC.
    # FClabs = list(np.loadtxt(data_folder + "FCavg_matrices/" + subj + "_roi_labels.txt", dtype=str))
    # FClabs = ["ctx-lh-" + lab[:-2] if lab[-1] == "L" else "ctx-rh-" + lab[:-2] for lab in FClabs]
    # FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois_dk]  # find indexes in FClabs that matches cortical_rois

    conn.weights = conn.weights[:, SC_dmn_idx][SC_dmn_idx]
    conn.tract_lengths = conn.tract_lengths[:, SC_dmn_idx][SC_dmn_idx]
    conn.region_labels = conn.region_labels[SC_dmn_idx]
    conn.cortical = conn.cortical[SC_dmn_idx]
    conn.centres = conn.centres[SC_dmn_idx]

#    ADNI PET DATA       ##########
ADNI_AVG = pd.read_csv(data_folder + "ADNI/.PET_AVx_GroupAVERAGED.csv", index_col=0)

# Check label order
PETlabs = list(ADNI_AVG.columns[12:])
PET_idx = [PETlabs.index(roi.lower()) for roi in list(conn.region_labels)]



#     2.  DEFINE INITIAL CONDITIONS       ######################################

# TODO - check structural matrix damage; tackle rebound (limiting TAUhp?);
#  tackle stability of POWdam and thus AB42;

"""
                A)   Alex Init  
Circular model  -  Alexandersen (2022) initial conditions

Seeding areas rise much faster than the rest of the nodes.
"""

AB_initMap, TAU_initMap = [[1 for roi in conn.region_labels]] * 2

##  REGIONAL SEEDs for toxic proteins
# seeding of AB from Palmqvist 2017; seeding of tau from
AB_seeds = ["ctx-lh-precuneus", "ctx-lh-isthmuscingulate", "ctx-lh-insula", "ctx-lh-medialorbitofrontal",
            "ctx-lh-lateralorbitofrontal", "ctx-lh-posteriorcingulate",
            "ctx-rh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-insula", "ctx-rh-medialorbitofrontal",
            "ctx-rh-lateralorbitofrontal", "ctx-rh-posteriorcingulate"]

TAU_seeds = ["ctx-lh-entorhinal", "ctx-rh-entorhinal"]

ABt_initMap = [0.05 / len(AB_seeds) if roi in AB_seeds else 0 for roi in conn.region_labels]
TAUt_initMap = [0.005 / len(TAU_seeds) if roi in TAU_seeds else 0 for roi in conn.region_labels]

AB_initdam, TAU_initdam = [[0 for roi in conn.region_labels]] * 2
HA_initdam = [0 for roi in conn.region_labels]


#    3. PARAMETERS   and   SIMULATE      ########################################
title, tic = "vCC_cModel_AlexInit_cb" + str(useCB), time.time()
prop_simL, prop_dt = 40, 0.25
bnm_simL, transient, bnm_dt = 10, 2, 1  # Units (seconds, seconds, years)

circmodel = CircularADpgModel_vCC(
    conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam, TAU_initdam, HA_initdam,
    TAU_dam2SC=5e-2, HA_damrate=5, maxTAU2SCdam=0.3,  # maxHAdam=1.25,  # origins @ ¿?, 0.01, 1.5
    init_He=3.25, init_Cee=108, init_Cie=33.75,  # origins 3.25, 108, 33.75 || Initial values for NMM variable parameters
    rho=50, toxicSynergy=0.4,  # origins 5, 2 || rho as a diffusion constant
    prodAB=3, clearAB=3, transAB2t=3, clearABt=2.4,
    prodTAU=3, clearTAU=3, transTAU2t=3, clearTAUt=2.55,
    cABexc=0.8, cABinh=0.4, cTAUexc=1.8, cTAUinh=1.8)

# 3.2 Define parameter ranges of change
rHe = [0.35, 0.35]
circmodel.init_He["range"] = [circmodel.init_He["value"][0] - rHe[0], circmodel.init_He["value"][0] + rHe[1]]

rCee = [80, 40]  # origins (54, 160) :: (-54+x, x+54)
circmodel.init_Cee["range"] = [circmodel.init_Cee["value"][0] - rCee[0], circmodel.init_Cee["value"][0] + rCee[1]]

rCie = [20.5, 10]  # origins (15, 50) :: (-16.75+x, x+16.25)
circmodel.init_Cie["range"] = [circmodel.init_Cie["value"][0] - rCie[0], circmodel.init_Cie["value"][0] + rCie[1]]


# 3.3 Run
out_circ = circmodel.run(time=prop_simL, dt=prop_dt, sim=[subj, g, s, sigma, bnm_simL, transient], sim_dt=bnm_dt)
                                            # sim=[False] (xor) [subj, model, g, s, time(s), transient(s)](simParams)


##    4.   BUILD SPEC FOLDER and SAVE PARAMETERS    ############
spec_fold = "results/" + title + "_bnm-dt" + str(bnm_dt) + \
            "sL" + str(bnm_simL) + "_" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
os.mkdir(spec_fold)

with open(spec_fold + "/.PARAMETERS.txt", "w") as f:
    f.write('Title - %s; simtime: %0.2f(min)\nsubj = %s; useDMN = %s; g = %0.2f; s = %0.2f; sigma = %0.2f\n'
            'prop_simL = %0.2f(y); prop_dt = %0.2f(y)\nbnm_simL = %0.2f(s); transient = %0.2f(s); bnm_dt = %0.2f(y)\n\n'
            % (title, (time.time()-tic)/60, subj, str(useCB), g, s, sigma, prop_simL, prop_dt, bnm_simL, transient, bnm_dt))
    # add title subj etc
    for key, val in vars(circmodel).items():
        f.write('%s:%s\n' % (key, val))

##   4b.  if needed SAVE DATA    #############
save = True
if save:
    with open(spec_fold + "/.DATA.pkl", "wb") as file:
        pickle.dump([out_circ, conn], file)
        file.close()


##    5.  PLOTTING MACHINERY        #########################################
# 5.1 Default new plot
circApproach(out_circ, conn, title, folder=spec_fold)
braidPlot(out_circ, conn, mode="diagram")


