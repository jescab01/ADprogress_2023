
"""

Adding a couple of experiments and visualizations to respond to reviews from eNeuro.

Experiments have to do with Theta band and heterogeneity. Taking the output from simulations to
calculate PLV in other bands (delta, theta, alpha and broadband).

TODO - 2. evaluate the evolution of FC in other bands: how is the rise-decay dynamic?

"""

import os
import pickle
import itertools
import scipy.signal
from mne import filter
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


if "LCCN_Local" in os.getcwd():
    data_folder = "E:\\LCCN_Local\PycharmProjects\ADprogress_data\\"
    import sys

    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import FFTpeaks, multitapper, PSD
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV



##### 0. LOAD simulated out_circ data

sim_fold = "E:\LCCN_Local\PycharmProjects\ADprogress\PAPER\Rreview\\"
sim_tag = "vCC_cModel_AlexInit_cbTrue_bnm-dt1sL10_m11d07y2023-t21h.32m.04s"

with open(sim_fold + sim_tag + "/.DATA.pkl", "rb") as file:
    [out_circ, conn] = pickle.load(file)
    file.close()

shortout = [np.array([out_circ[0][i] for i, out in enumerate(out_circ[2]) if len(out) > 1]),
            np.array([out[4] for i, out in enumerate(out_circ[2]) if len(out) > 1])]  # [40years, 40rois, 8000secs]


#### 2. FC in THETA/DELTA bands

# bands = [["3-alpha"], [(8, 12)]]
# bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]
bands = [["Delta", "Theta", "Alpha"], [(2, 4), (4, 8), (8, 12)]]

# This process last long time - 5 min :: be patient.
plv_per_bands = []
for t, pspPyr in enumerate(shortout[1]):
    print("Working on t==" + str(t))

    plv_temp = []
    for b in range(len(bands[0])):

        (lowcut, highcut) = bands[1][b]

        samplingFreq = 1000  # Hz

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

        # CONNECTIVITY MEASURES
        ## PLV and plot
        plv_sim = PLV(efPhase, verbose=False)
        plv_temp.append(plv_sim)

    plv_per_bands.append(plv_temp)  # [40yrs, 4bands, 40rois x 40rois]

plv_per_bands_avg = np.array([[np.average(band[np.triu_indices(len(band), 1)]) for band in t] for t in plv_per_bands])



fig = go.Figure()

for i, band in enumerate(bands[0]):

    fig.add_trace(go.Scatter(x=shortout[0], y=plv_per_bands_avg[:, i], name=band, legendgroup=band))

fig.update_layout(template="plotly_white", height=400, width=600,
                  xaxis=dict(title="Time (years)"), yaxis=dict(title="Averaged PLV"))

pio.write_html(fig, file=sim_fold + sim_tag + "/Rrev2_PLV_per_band.html", auto_open=True)
pio.write_image(fig, file=sim_fold + sim_tag + "/Rrev2_PLV_per_band.svg")
pio.write_image(fig, file=sim_fold + sim_tag + "/Rrev2_PLV_per_band.png")



