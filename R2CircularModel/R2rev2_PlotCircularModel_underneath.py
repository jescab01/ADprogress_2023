
"""
    This script plots the results of the Circular model
"""

import pandas as pd
import pingouin as pg

from itertools import combinations
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

from tvb.simulator.lab import *

import pickle
import numpy as np

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.littlebrains import addpial


######        1. Load data: out_circ and conn from saved simulation      ######
simname = "vCC_cModel_AlexInit_cbTrue_bnm-dt1sL10_m11d07y2023-t21h.32m.04s"
main_folder = "E:\LCCN_Local\PycharmProjects\ADprogress\\\PAPER\R2CircularModel\\"


with open(main_folder + simname + "\\.DATA.pkl", "rb") as file:
    out_circ, conn = pickle.load(file)
    file.close()


#####      2. Process the data to extract the variables of interest      #######
# 2a. Extract values from proteinopathy dynamics
time = out_circ[0]
data = np.array(out_circ[1])
_, ABt, _, TAUt, ABdam, TAUdam, He, Cep, Cip, HAdam = np.array(out_circ[1]).transpose((1,2,0))
weights = np.array([np.average(t_list[0], axis=1) for t_list in out_circ[2]]).transpose()


# 2b. Extract FC and RATE values from the simulations
time_short, data_fc, data_fr = [], [], []
data_freqs, data_pow, data_relpow, abs_ffts, norm_ffts = [], [], [], [], []
for i, simpack in enumerate(out_circ[2]):
    if len(simpack) > 1:
        time_short.append(out_circ[0][i])
        data_fr.append(out_circ[2][i][3])  # data_fr=[time x rois]
        data_fc.append(np.average(out_circ[2][i][2], axis=1))

        data_freqs.append(simpack[1][0])
        # Calculate rel_power per bands
        ffts, freqs = simpack[1][3:]
        data_pow.append(np.sum(ffts, axis=1))

        # AVG and normalize
        norm_ffts = [(fft - min(fft)) / (max(fft) - min(fft)) for fft in ffts]
        pow_beta = [sum(norm_fft[(12 < freqs)]) / sum(norm_fft) for norm_fft in norm_ffts]
        pow_alpha = [sum(norm_fft[(8 < freqs) & (freqs < 12)]) / sum(norm_fft) for norm_fft in norm_ffts]
        pow_theta = [sum(norm_fft[(4 < freqs) & (freqs < 8)]) / sum(norm_fft) for norm_fft in norm_ffts]
        pow_delta = [sum(norm_fft[(2 < freqs) & (freqs < 4)]) / sum(norm_fft) for norm_fft in norm_ffts]

        data_relpow.append([pow_beta, pow_alpha, pow_theta, pow_delta])

data_fr, data_fc, data_pow, data_freqs = (np.array(data_fr).transpose(), np.array(data_fc).transpose(),
                                          np.array(data_pow).transpose(), np.array(data_freqs).transpose())


## Group rois into lobes: frontal, parietal, temporal, occipital; and left and right
lobes = True
if lobes:
    groups = [
        ("Left Frontal",
         ['ctx-lh-medialorbitofrontal','ctx-lh-lateralorbitofrontal', 'ctx-lh-frontalpole',
          'ctx-lh-rostralmiddlefrontal','ctx-lh-caudalmiddlefrontal','ctx-lh-superiorfrontal',]),
        ("Left Parietal",
         ['ctx-lh-superiorparietal','ctx-lh-inferiorparietal','ctx-lh-precuneus',]),
        ("Left Cingulate",
         ['ctx-lh-isthmuscingulate','ctx-lh-caudalanteriorcingulate','ctx-lh-rostralanteriorcingulate','ctx-lh-posteriorcingulate',]),
        ("Left Temporal",
         ['ctx-lh-parahippocampal','ctx-lh-inferiortemporal','Left-Hippocampus', 'ctx-lh-entorhinal',]),

        ("Right Frontal",
         [ 'ctx-rh-medialorbitofrontal',  'ctx-rh-lateralorbitofrontal', 'ctx-rh-frontalpole',
         'ctx-rh-rostralmiddlefrontal','ctx-rh-caudalmiddlefrontal', 'ctx-rh-superiorfrontal',]),
        ("Right Parietal",
         ['ctx-rh-superiorparietal','ctx-rh-inferiorparietal','ctx-rh-precuneus',]),
        ("Right Cingulate",
         [  'ctx-rh-isthmuscingulate', 'ctx-rh-caudalanteriorcingulate','ctx-rh-rostralanteriorcingulate','ctx-rh-posteriorcingulate', ]),
        ("Right Temporal",
         ['ctx-rh-parahippocampal','ctx-rh-inferiortemporal','Right-Hippocampus','ctx-rh-entorhinal'])]

    ABt_avg, TAUt_avg, ABdam_avg, TAUdam_avg, He_avg, Cep_avg, Cip_avg, HAdam_avg, wij_avg = [], [], [], [], [], [], [], [], []
    data_fc_avg, data_fr_avg, data_freqs_avg, data_pow_avg = [], [], [], []

    for group, rois in groups:
        print(group)
        # sacar los indices de esas regiones en el conn
        ids = [list(conn.region_labels).index(roi) for roi in rois]

        # convertir cada medida usada en una mas simple
        ABt_avg.append(np.average(ABt[ids, :], axis=0))

        TAUt_avg.append(np.average(TAUt[ids, :], axis=0))

        ABdam_avg.append(np.average(ABdam[ids, :], axis=0))
        TAUdam_avg.append(np.average(TAUdam[ids, :], axis=0))
        He_avg.append(np.average(He[ids, :], axis=0))
        Cip_avg.append(np.average(Cip[ids, :], axis=0))
        Cep_avg.append(np.average(Cep[ids, :], axis=0))
        HAdam_avg.append(np.average(HAdam[ids, :], axis=0))
        wij_avg.append(np.average(weights[ids, :], axis=0))

        data_fc_avg.append(np.average(data_fc[ids, :], axis=0))
        data_fr_avg.append(np.average(data_fr[ids, :], axis=0))
        data_freqs_avg.append(np.average(data_freqs[ids, :], axis=0))
        data_pow_avg.append(np.average(data_pow[ids, :], axis=0))

    ROIs = [group for group, _ in groups]

else:
    ABt_avg, TAUt_avg, ABdam_avg, TAUdam_avg, He_avg, Cep_avg, Cip_avg, HAdam_avg, wij_avg = (
        ABt, TAUt, ABdam, TAUdam, He, Cep, Cip, HAdam, weights)
    data_fc_avg, data_fr_avg, data_freqs_avg, data_pow_avg = data_fc, data_fr, data_freqs, data_pow

    ROIs = conn.region_labels  # for individual traces

### PLOT
cmap = px.colors.qualitative.Pastel2[:4] + px.colors.qualitative.Set2[:4]
fig = make_subplots(rows=5, cols=4, horizontal_spacing=0.1)

for i, roi in enumerate(ROIs):

    # ADD ABt/TAUt traces
    fig.add_trace(go.Scatter(x=time, y=ABt_avg[i], name=roi, legendgroup=roi, line=dict(color=cmap[i])), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=TAUt_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=3, col=1)

    # ADD qABt/qTAUt/qHA traces
    fig.add_trace(go.Scatter(x=time, y=ABdam_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=2, col=2)
    fig.add_trace(go.Scatter(x=time, y=TAUdam_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=3, col=2)
    fig.add_trace(go.Scatter(x=time, y=HAdam_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=4, col=1)

    # ADD NMM params traces
    fig.add_trace(go.Scatter(x=time, y=He_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=1, col=3)
    fig.add_trace(go.Scatter(x=time, y=Cip_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=2, col=3)
    fig.add_trace(go.Scatter(x=time, y=Cep_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=3, col=3)
    fig.add_trace(go.Scatter(x=time, y=wij_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=4, col=3)

    # ADD functional measures
    fig.add_trace(go.Scatter(x=time_short, y=data_fr_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=5, col=1)
    fig.add_trace(go.Scatter(x=time_short, y=data_freqs_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=5, col=2)
    fig.add_trace(go.Scatter(x=time_short, y=data_pow_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=5, col=3)
    fig.add_trace(go.Scatter(x=time_short, y=data_fc_avg[i], name=roi, legendgroup=roi, showlegend=False, line=dict(color=cmap[i])), row=5, col=4)


fig.update_layout(template="plotly_white", legend=dict(x=0, y=1.2, xanchor="left", orientation="h"), height=750, width=1200,

                  xaxis17=dict(title="Time (years)"), xaxis18=dict(title="Time (years)"),
                  xaxis19=dict(title="Time (years)"), xaxis20=dict(title="Time (years)"),

                  yaxis3=dict(title="H<sub>e"),
                  yaxis5=dict(title=r"$\tilde{A\beta}$"), yaxis6=dict(title=r"$q^(\tilde{A\beta})$"), yaxis7=dict(title="C<sub>ip"),
                  yaxis9=dict(title=r"$\tilde{T}$"), yaxis10=dict(title=r"$q^(\tilde{T})$"), yaxis11=dict(title="C<sub>ep"),
                  yaxis13=dict(title=r"$q^(ha)$"), yaxis15=dict(title="w<sub>ij"),
                  yaxis17=dict(title="Firing rate (kHz)"), yaxis18=dict(title="Frequency (Hz)"),
                  yaxis19=dict(title="Power (db)"), yaxis20=dict(title="PLV"))

pio.write_html(fig, "E:\LCCN_Local\PycharmProjects\ADprogress\PAPER\Rreview\\figures\R2rev2_CircularModel_underneath.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, "E:\LCCN_Local\PycharmProjects\ADprogress\PAPER\Rreview\\figures\R2rev2_CircularModel_underneath.svg")











