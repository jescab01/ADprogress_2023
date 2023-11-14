
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
simname = "vCC_cModel_AlexInit_cbTrue_bnm-dt1sL10_m06d12y2023-t18h.17m.28s"
main_folder = "E:\LCCN_Local\PycharmProjects\ADprogress\\\PAPER\R2CircularModel\\"

with open(main_folder + simname + "\\.DATA.pkl", "rb") as file:
    out_circ, conn = pickle.load(file)
    file.close()


#####      2. Process the data to extract the variables of interest      #######
# 2a. Extract values from proteinopathy dynamics
timepoints = out_circ[0]
_, ABt, _, TAUt, ABdam, TAUdam, He, Cee, Cie, HAdam = np.average(out_circ[1], axis=2).transpose()
avg_weights = np.array([np.average(t_list[0]) for t_list in out_circ[2]])

# 2b. Extract FC and RATE values from the simulations
# Separate the results into anterior and posterior
posterior_rois = ['ctx-lh-cuneus', 'ctx-lh-inferiorparietal', 'ctx-lh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                  'ctx-lh-lingual', 'ctx-lh-pericalcarine', 'ctx-lh-postcentral', 'ctx-lh-posteriorcingulate',
                  'ctx-lh-precuneus', 'ctx-lh-superiorparietal', 'ctx-lh-supramarginal',

                  'ctx-rh-cuneus', 'ctx-rh-inferiorparietal', 'ctx-rh-isthmuscingulate', 'ctx-rh-lateraloccipital',
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

timepoints_short, avg_fc, avg_rate, posterior_fc, anterior_fc, ant_avgrate, post_avgrate = [], [], [], [], [], [], []
peak_freqs, band_powers, abs_ffts, norm_ffts = [], [], [], []
for i, simpack in enumerate(out_circ[2]):
    if len(simpack) > 1:
        timepoints_short.append(out_circ[0][i])
        avg_rate.append(np.average(out_circ[2][i][3]))
        post_avgrate.append(np.average(out_circ[2][i][3][posterior_ids]))
        ant_avgrate.append(np.average(out_circ[2][i][3][anterior_ids]))
        avg_fc.append(np.average(out_circ[2][i][2]))
        posterior_fc.append(np.average(out_circ[2][i][2][:, posterior_ids][posterior_ids]))
        anterior_fc.append(np.average(out_circ[2][i][2][:, anterior_ids][anterior_ids]))

        peak_freqs.append(np.average(simpack[1][0]))
        # Calculate rel_power per bands
        ffts, freqs = simpack[1][3:]
        # AVG and normalize
        avgfft = np.average(ffts, axis=0)
        abs_ffts.append(avgfft)
        # normavg_fft = avg_fft / sum(avg_fft)
        normavg_fft = (avgfft - min(avgfft)) / (max(avgfft) - min(avgfft))
        pow_beta = sum(normavg_fft[(12 < freqs)]) / sum(normavg_fft)
        pow_alpha = sum(normavg_fft[(8 < freqs) & (freqs < 12)]) / sum(normavg_fft)
        pow_theta = sum(normavg_fft[(4 < freqs) & (freqs < 8)]) / sum(normavg_fft)
        pow_delta = sum(normavg_fft[(2 < freqs) & (freqs < 4)]) / sum(normavg_fft)

        band_powers.append([pow_beta, pow_alpha, pow_theta, pow_delta])
        norm_ffts.append(normavg_fft)

band_powers = np.asarray(band_powers)


## PLOTTING - General
line_width, opacity = 2.5, 0.7
cmap = px.colors.qualitative.Plotly
c_ab, c_tau, c_he, c_cie, c_cee, c_sc, c_fr = cmap[2], cmap[0], "goldenrod", "cornflowerblue", "indianred", "dimgray", "darkkhaki"



#####       4. R2b - Plot the outputs found        #######

## Part 1: Line plots
fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.1)

# Frequency
fig.add_trace(go.Scatter(x=timepoints_short, y=peak_freqs, name="Frequency peak", legendgroup="netsim",
                         line=dict(width=2, color="black"), opacity=0.7), row=1, col=1)

# Relative power
fig.add_trace(go.Scatter(x=timepoints_short, y=band_powers[:, 2], name="Theta power", legendgroup="pow",
                         line=dict(width=2, color="palevioletred"), opacity=0.8), row=1, col=2)
fig.add_trace(go.Scatter(x=timepoints_short, y=band_powers[:, 1], name="Alpha power", legendgroup="pow",
                         line=dict(width=3, color="sandybrown"), opacity=0.8), row=1, col=2)
fig.add_trace(go.Scatter(x=timepoints_short, y=band_powers[:, 0], name="Beta power", legendgroup="pow",
                         line=dict(width=1, color="darkkhaki"), opacity=0.8), row=1, col=2)

# Firing rate
fig.add_trace(go.Scatter(x=timepoints_short, y=post_avgrate, name="Firing rate (posterior)", legendgroup="rate",
                         line=dict(width=2, color="firebrick"), opacity=0.8),
              row=1, col=3)
fig.add_trace(go.Scatter(x=timepoints_short, y=ant_avgrate, name="Firing rate (anterior)", legendgroup="rate",
                         line=dict(width=2, color="cornflowerblue"), opacity=0.8),
              row=1, col=3)
fig.add_trace(go.Scatter(x=timepoints_short, y=posterior_fc, name="PLV (posterior)", legendgroup="fc",
                         line=dict(width=3, color="firebrick"), opacity=0.7), row=1, col=4)
fig.add_trace(go.Scatter(x=timepoints_short, y=anterior_fc, name="PLV (anterior)", legendgroup="fc",
                         line=dict(width=2, color="cornflowerblue"), opacity=0.7), row=1, col=4)

fig.update_layout(template="plotly_white", height=275, width=1000, legend=dict(orientation="h", y=1.85, x=0.15),
                  xaxis1=dict(title="Time (years)"), xaxis2=dict(title="Time (years)"), xaxis3=dict(title="Time (years)"), xaxis4=dict(title="Time (years)"),
                  yaxis1=dict(title="Frequency (Hz)", range=[4, 12], title_standoff=0), yaxis2=dict(title="Relative power", title_standoff=0),
                  yaxis3=dict(title="Firing rate (kHz)", title_standoff=0), yaxis4=dict(title="PLV", title_standoff=0),)

pio.write_html(fig, main_folder + simname + "\\R2b_Part1_CircularModel_output.html", auto_open=True)
pio.write_image(fig, main_folder + simname + "\\R2b_Part1_CircularModel_output.svg")



## PART 2: timepoints
def FCplusrate_3dviz(fig, out_circ, conn, timepoint, pos, surrogates, threshold=0.05):

    # 1. Static lines for posterior, anterior and rate
    posterior_rois = ['ctx-lh-cuneus', 'ctx-lh-inferiorparietal', 'ctx-lh-isthmuscingulate', 'ctx-lh-lateraloccipital',
                      'ctx-lh-lingual', 'ctx-lh-pericalcarine','ctx-lh-postcentral', 'ctx-lh-posteriorcingulate',
                      'ctx-lh-precuneus', 'ctx-lh-superiorparietal', 'ctx-lh-supramarginal',

                      'ctx-rh-cuneus',  'ctx-rh-inferiorparietal', 'ctx-rh-isthmuscingulate', 'ctx-rh-lateraloccipital',
                      'ctx-rh-lingual', 'ctx-rh-pericalcarine', 'ctx-rh-postcentral', 'ctx-rh-posteriorcingulate',
                      'ctx-rh-precuneus', 'ctx-rh-superiorparietal', 'ctx-rh-supramarginal']

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
                               line=dict(color=edges_color[timepoint], width=3), opacity=0.6, name="Edges"), row=pos[0], col=pos[1])

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
                               marker=dict(size=size, color=nodes_color[timepoint], opacity=1,
                                           line=dict(color="gray", width=2))), row=pos[0], col=pos[1])

    fig = addpial(fig, mode="surface", opacity=0.1, row=pos[0], col=pos[1])

    return fig, edges_color, nodes_color, hovertext3d

zoom = 0.2
camera = dict(eye=dict(x=1.25 * (1-zoom), y=1.25 * (1-zoom), z=1.25 * (1-zoom)))

tpoints = [5, 15, 20, 25, 30]
sp_titles = ["t == " + str(t) for t in tpoints] + [""]*5

fig = make_subplots(rows=2, cols=5, row_heights=[0.65, 0.35], subplot_titles=sp_titles, vertical_spacing=0.03, horizontal_spacing=0.03, shared_yaxes=True,
                    specs=[[{"type":"surface"}, {"type":"surface"}, {"type":"surface"}, {"type":"surface"}, {"type":"surface"}], [{},{},{},{},{}]])

for j, t in enumerate(tpoints):

    ## Plot Spectra
    fig.add_trace(go.Scatter(x=freqs, y=norm_ffts[t], line=dict(color="dimgray", width=2)), row=2, col=1+j)

    # Plot little brains
    fig, _, _, _ = FCplusrate_3dviz(fig, out_circ, conn, t, (1, j+1), surrogates=None, threshold=0.05)

fig.update_layout(template="plotly_white", height=400, width=1000, showlegend=False,

                  xaxis1=dict(title="Frequency (Hz)"),  xaxis2=dict(title="Frequency (Hz)"),  xaxis3=dict(title="Frequency (Hz)"),  xaxis4=dict(title="Frequency (Hz)"),  xaxis5=dict(title="Frequency (Hz)"),
                  yaxis1=dict(title="Norm. power"),

                  scene1=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=camera),
                  scene2=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=camera),
                  scene3=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=camera),
                  scene4=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=camera),
                  scene5=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=camera),)

pio.write_html(fig, main_folder + simname + "\\R2b_Part2_CircularModel_output.html", auto_open=True)
pio.write_image(fig, main_folder + simname + "\\R2b_Part2_CircularModel_output.svg")




