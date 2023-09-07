
import os
import time
import numpy as np
from mpi4py import MPI
import pandas as pd
import datetime

from tvb.simulator.lab import connectivity


def adpgCirc_parallel(params_):

    result = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"

        import sys
        sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
        from ADprogress.ADpg_functions import CircularADpgModel_vH, CircularADpgModel_vCC, braidPlot

        # from toolbox.fft import multitapper
        # from toolbox.signals import epochingTool
        # from toolbox.fc import PLV
        # from toolbox.dynamics import dynamic_fc, kuramoto_order

    ## Folder structure - CLUSTER CesViMa
    elif "t192" in os.getcwd():
        wd = "/home/t192/t192950/mpi/"
        data_folder = wd + "ADprogress_data/"

        from ADpg_functions import CircularADpgModel_vH, CircularADpgModel_vCC, braidPlot
        # import sys
        # sys.path.append(wd)
        # from toolbox.fft import multitapper
        # from toolbox.signals import epochingTool
        # from toolbox.fc import PLV
        # from toolbox.dynamics import dynamic_fc, kuramoto_order

    ## Folder structure - CLUSTER BRIGIT
    else:
        wd = "/mnt/lustre/home/jescab01/"
        data_folder = wd + "ADprogress_data/"

        from ADpg_functions import CircularADpgModel_vH, CircularADpgModel_vCC, braidPlot
        # import sys
        # sys.path.append(wd)
        # from toolbox.fft import multitapper
        # from toolbox.signals import epochingTool
        # from toolbox.fc import PLV
        # from toolbox.dynamics import dynamic_fc, kuramoto_order

    for ii, set in enumerate(params_):

        print("Rank %i out of %i  ::  %i/%i " % (rank, size, ii + 1, len(params_)))

        print(set)
        # maxHe, mCie, mCee, maxTAU2SC, rho, HAdamrate
        g, s, maxHe, mCie, mCee, maxTAU2SC, rho, HAdamrate, cABinh, cTAUinh = set

        #     1.  PREPARE EMPIRICAL DATA      #########################
        #   STRUCTURAL CONNECTIVITY   #########
        #  Define structure through which the proteins will spread;
        #  Not necessarily the same than the one used to simulate activity.
        subj, sigma = "HC-fam", 0.022  # Noise following David and Friston 2003.
        conn = connectivity.Connectivity.from_file(data_folder + "SC_matrices/" + subj + "_aparc_aseg-mni_09c.zip")
        conn.weights = conn.scaled_weights(mode="tract")

        cortical_rois = ['ctx-lh-bankssts', 'ctx-rh-bankssts', 'ctx-lh-caudalanteriorcingulate',
                         'ctx-rh-caudalanteriorcingulate',
                         'ctx-lh-caudalmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-rh-cuneus',
                         'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-frontalpole', 'ctx-rh-frontalpole',
                         'ctx-lh-fusiform', 'ctx-rh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal',
                         'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal', 'ctx-lh-insula', 'ctx-rh-insula',
                         'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                         'ctx-rh-lateraloccipital',
                         'ctx-lh-lateralorbitofrontal', 'ctx-rh-lateralorbitofrontal', 'ctx-lh-lingual',
                         'ctx-rh-lingual',
                         'ctx-lh-medialorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-lh-middletemporal',
                         'ctx-rh-middletemporal',
                         'ctx-lh-paracentral', 'ctx-rh-paracentral', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal',
                         'ctx-lh-parsopercularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsorbitalis',
                         'ctx-rh-parsorbitalis',
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

        ## TO USE JUST DMN
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
        """
                        A)   Alex Init  
        Circular model  -  Alexandersen (2022) initial conditions

        Seeding areas rise much faster than the rest of the nodes.
        """

        AB_initMap, TAU_initMap = [[1 for roi in conn.region_labels]] * 2

        ##  REGIONAL SEEDs for toxic proteins
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
        title, tic = "vH_cModel_AlexInit_dmn" + str(useCB), time.time()
        prop_simL, prop_dt = 40, 0.25
        bnm_simL, transient, bnm_dt = 10, 2, 1  # Units (seconds, seconds, years)

        circmodel = CircularADpgModel_vCC(
            conn, AB_initMap, TAU_initMap, ABt_initMap, TAUt_initMap, AB_initdam, TAU_initdam, HA_initdam,
            TAU_dam2SC=5e-2, HA_damrate=HAdamrate, maxTAU2SCdam=maxTAU2SC,  # maxHAdam=maxHAdam,  # origins @ ¿?, 0.01, 1.5
            init_He=3.25, init_Cee=108, init_Cie=33.75, # init_He=3.25, init_Hi=22,  # origins 3.25, 22 || Initial values for NMM variable parameters
            rho=rho, toxicSynergy=0.4,  # origins 5, 2 || rho as a diffusion constant
            prodAB=3, clearAB=3, transAB2t=3, clearABt=2.4,
            prodTAU=3, clearTAU=3, transTAU2t=3, clearTAUt=2.55,
            cABinh=cABinh, cTAUinh=cTAUinh)

        # 3.2 Define parameter ranges of change
        # rHe = [0.4, 0.4]
        # circmodel.init_He["range"] = [circmodel.init_He["value"][0] - rHe[0], circmodel.init_He["value"][0] + rHe[1]]
        #
        # rHi = [minHi, 2]
        # circmodel.init_Hi["range"] = [circmodel.init_Hi["value"][0] - rHi[0], circmodel.init_Hi["value"][0] + rHi[1]]
        rHe = [0.35, maxHe]
        circmodel.init_He["range"] = [circmodel.init_He["value"][0] - rHe[0], circmodel.init_He["value"][0] + rHe[1]]

        rCee = [mCee, 40]  # origins (54, 160) :: (-54+x, x+54)
        circmodel.init_Cee["range"] = [circmodel.init_Cee["value"][0] - rCee[0],
                                       circmodel.init_Cee["value"][0] + rCee[1]]

        rCie = [mCie, 10]  # origins (15, 50) :: (-16.75+x, x+16.25)
        circmodel.init_Cie["range"] = [circmodel.init_Cie["value"][0] - rCie[0],
                                       circmodel.init_Cie["value"][0] + rCie[1]]

        # 3.3 Run
        out_circ = circmodel.run(time=prop_simL, dt=prop_dt, sim=[subj, g, s, sigma, bnm_simL, transient], sim_dt=bnm_dt)
        # sim=[False] (xor) [subj, model, g, s, time(s), transient(s)](simParams)

        ## v1. SAVE ALL
        # result.append(out_circ + [set])

        ## v2. PREPROCESS to SAVE just needed
        # 1. Static lines for posterior, anterior and rate
        posterior_rois = ['ctx-lh-cuneus', 'ctx-lh-inferiorparietal', 'ctx-lh-isthmuscingulate',
                          'ctx-lh-lateraloccipital',
                          'ctx-lh-lingual', 'ctx-lh-pericalcarine', 'ctx-lh-postcentral', 'ctx-lh-posteriorcingulate',
                          'ctx-lh-precuneus', 'ctx-lh-superiorparietal', 'ctx-lh-supramarginal',

                          'ctx-rh-cuneus', 'ctx-rh-inferiorparietal', 'ctx-rh-isthmuscingulate',
                          'ctx-rh-lateraloccipital',
                          'ctx-rh-lingual', 'ctx-rh-pericalcarine', 'ctx-rh-postcentral', 'ctx-rh-posteriorcingulate',
                          'ctx-rh-precuneus', 'ctx-rh-superiorparietal', 'ctx-rh-supramarginal']
        anterior_rois = [
            'ctx-lh-caudalanteriorcingulate', 'ctx-lh-caudalmiddlefrontal',
            'ctx-lh-lateralorbitofrontal',
            'ctx-lh-medialorbitofrontal',
            'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis',
            'ctx-lh-parstriangularis', 'ctx-lh-precentral',
            'ctx-lh-rostralanteriorcingulate', 'ctx-lh-rostralmiddlefrontal',
            'ctx-lh-superiorfrontal', 'ctx-lh-frontalpole',

            'ctx-rh-caudalanteriorcingulate', 'ctx-rh-caudalmiddlefrontal',
            'ctx-rh-lateralorbitofrontal',
            'ctx-rh-medialorbitofrontal',
            'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis',
            'ctx-rh-parstriangularis', 'ctx-rh-precentral',
            'ctx-rh-rostralanteriorcingulate', 'ctx-rh-rostralmiddlefrontal',
            'ctx-rh-superiorfrontal', 'ctx-rh-frontalpole']

        posterior_ids = np.array([i for i, roi in enumerate(conn.region_labels) if roi in posterior_rois])
        anterior_ids = [i for i, roi in enumerate(conn.region_labels) if roi in anterior_rois]

        braidrx_time = braidPlot(out_circ, conn, mode="data_intime")

        timepoints, peak_freqs, pow_alpha, post_avgrate, ant_avgrate, posterior_fc, anterior_fc = [], [], [], [], [], [], []
        braidI, braidII, braidIII, braidIV, braidV = [], [], [], [], []

        for i, simpack in enumerate(out_circ[2]):
            if len(simpack) > 1:
                # 0. timepoints
                timepoints.append(out_circ[0][i])

                # 1. Frequency peaks
                peak_freqs.append(np.average(simpack[1][0]))

                # Calculate rel_power per bands
                ffts, freqs = simpack[1][3:]

                # AVG and normalize
                avgfft = np.average(ffts, axis=0)

                normavg_fft = (avgfft - min(avgfft)) / (max(avgfft) - min(avgfft))

                # 2. Rel power
                # pow_beta = sum(normavg_fft[(12 < freqs)]) / sum(normavg_fft)
                pow_alpha.append(sum(normavg_fft[(8 < freqs) & (freqs < 12)]) / sum(normavg_fft))
                # pow_theta = sum(normavg_fft[(4 < freqs) & (freqs < 8)]) / sum(normavg_fft)
                # pow_delta = sum(normavg_fft[(2 < freqs) & (freqs < 4)]) / sum(normavg_fft)

                # 3. AVG FR
                post_avgrate.append(np.average(out_circ[2][i][3][posterior_ids]))
                ant_avgrate.append(np.average(out_circ[2][i][3][anterior_ids]))

                # 4. AVG FC
                posterior_fc.append(np.average(out_circ[2][i][2][:, posterior_ids][posterior_ids]))
                anterior_fc.append(np.average(out_circ[2][i][2][:, anterior_ids][anterior_ids]))

                # 5. Braid data
                braidI.append(braidrx_time[0, i])
                braidII.append(braidrx_time[1, i])
                braidIII.append(braidrx_time[2, i])
                braidIV.append(braidrx_time[3, i])
                braidV.append(braidrx_time[4, i])

        # result.append(np.array([[minHi]*len(peak_freqs), [maxTAU2SC]*len(peak_freqs),
        #                         timepoints, peak_freqs, pow_alpha,
        #                         post_avgrate, ant_avgrate, posterior_fc, anterior_fc]).transpose())

        result.append(np.array([[g]*len(peak_freqs), [s]*len(peak_freqs),

                                [maxHe]*len(peak_freqs), [mCie]*len(peak_freqs),
                                [mCee]*len(peak_freqs), [maxTAU2SC]*len(peak_freqs), [rho]*len(peak_freqs),
                                [HAdamrate]*len(peak_freqs), [cABinh]*len(peak_freqs), [cTAUinh]*len(peak_freqs),
                                timepoints, peak_freqs, pow_alpha,
                                post_avgrate, ant_avgrate, posterior_fc, anterior_fc,
                                braidI, braidII, braidIII, braidIV, braidV]).transpose())

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    result = np.vstack(result)

    return result
