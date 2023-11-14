
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


samplingFreq = 1000  # Hz

##### 0. LOAD simulated out_circ data

sim_fold = "E:\LCCN_Local\PycharmProjects\ADprogress\PAPER\Rreview\\"
sim_tag = "vCC_cModel_AlexInit_cbTrue_bnm-dt1sL10_m11d07y2023-t21h.32m.04s"

with open(sim_fold + sim_tag + "/.DATA.pkl", "rb") as file:
    [out_circ, conn] = pickle.load(file)
    file.close()

shortout = [np.array([out_circ[0][i] for i, out in enumerate(out_circ[2]) if len(out) > 1]),
            [out_circ[1][i] for i, out in enumerate(out_circ[2]) if len(out) > 1],
            [out for i, out in enumerate(out_circ[2]) if len(out) > 1]]


#### 1. REGIONAL SPECTRA :: Is alpha actually slowing?
yrs = shortout[0]

# Extracting fft data from out circ
# freqs = shortout[2][0][1][4]  # [outs][t==0][spectral_info][curve_freqs]
# spectra = [out[1][3] for out in shortout[2]]  # spectra:[40t, 40rois, 231freqs]

# Extracting signals to calculate spectra
signals = [out[4] for out in shortout[2]]

# Calculate spectra with multitapper to reduce noise
spectra = []
for signals_t in signals:
    freqs, spec = multitapper(signals_t, samplingFreq, conn.region_labels, epoch_length=4, ntapper=4, smoothing=0.5)
    spectra.append(spec)


# Extract static plots to add supplementary information
timepoints = 5  # Define the number of traces to plot
skip_yrs = len(yrs)//timepoints

roi_names = [roi[4:] if "ctx" in roi else roi for roi in conn.region_labels]


##  1A. NORMALIZED SPECTRA
cmap = px.colors.sequential.Viridis
cmap = cmap[::len(cmap)//timepoints]

spectra_norm = [[spec[r]/np.max(spec[r]) for r, roi in enumerate(conn.region_labels)] for t, spec in enumerate(spectra)]
plotout = [spectra_norm[i] for i, t in enumerate(yrs) if t in yrs[::skip_yrs]]

fig = make_subplots(rows=10, cols=4, shared_yaxes=True, shared_xaxes=True, subplot_titles=roi_names,
                    x_title="Frequency (Hz)", y_title="Normalized power")

for i, t in enumerate(yrs[::skip_yrs]):
    for r, roi in enumerate(conn.region_labels):
        col = 1 + r%4
        row = 1 + r//4
        sl = True if r == 1 else False
        fig.add_trace(go.Scatter(x=freqs, y=plotout[i][r], opacity=0.8, marker=dict(color=cmap[i]), line=dict(width=(1+i)*0.5),
                                 name="year==" + str(t), legendgroup="year==" + str(t), showlegend=sl),
                      row=row, col=col)
fig.update_layout(title="Normalized spectra", template="plotly_white", height=1500, width=1100)
pio.write_html(fig, file=sim_fold + sim_tag + "/Rrev1b_SpectraNorm_lh_rois.html", auto_open=True, auto_play=False)
pio.write_image(fig, file=sim_fold + sim_tag + "/Rrev1b_SpectraNorm_lh_rois.svg")
pio.write_image(fig, file=sim_fold + sim_tag + "/Rrev1b_SpectraNorm_lh_rois.png")




##  1B. NORMALIZED SPECTRA (animated)
cmap = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
cmap_lr = list(itertools.chain(*zip(cmap, cmap)))

fig = make_subplots(rows=1, cols=2, column_titles=["Left hemisphere", "Right hemisphere"],
                    y_title="Normalized power", shared_yaxes=True)

# Add Spectra for the first timestep
for i, roi in enumerate(conn.region_labels):
    col = 1 if ("ctx-lh" in roi) or ("Left" in roi) else 2
    name = roi[7:] if "ctx" in roi else roi[5:] if "Left" in roi else roi[6:]
    sl = True if col == 1 else False
    fig.add_trace(go.Scatter(x=freqs, y=spectra[0][i], marker=dict(color=cmap_lr[i]),
                             name=name, legendgroup=name, showlegend=sl), row=1, col=col)
fig.update(frames=[go.Frame(data=[go.Scatter(y=spectra_norm[i][ii]) for ii, roi in enumerate(conn.region_labels)],
                            traces=list(range(len(conn.region_labels))), name=str(t)) for i, t in enumerate(yrs)])

# CONTROLS : Add sliders and buttons
fig.update_layout(title="Normalized spectra",
    template="plotly_white", xaxis1=dict(title="Frequency (Hz)"), xaxis2=dict(title="Frequency (Hz)"),
    sliders=[dict(
        steps=[
            dict(method='animate', args=[[str(t)], dict(mode="immediate", frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                                        transition=dict(duration=0))], label=str(t)) for i, t in enumerate(yrs)],
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

pio.write_html(fig, file=sim_fold + sim_tag + "/Rrev1b_SpectraNorm_lh_rois_anim.html", auto_open=True, auto_play=False)




## 1C. ABSOLUTE SPECTRA
cmap = px.colors.sequential.Viridis
cmap = cmap[::len(cmap)//timepoints]

plotout = [spectra[i] for i, t in enumerate(yrs) if t in yrs[::skip_yrs]]

fig = make_subplots(rows=10, cols=4, shared_yaxes=True, shared_xaxes=True, subplot_titles=roi_names,
                    x_title="Frequency (Hz)", y_title="Power")

for i, t in enumerate(yrs[::skip_yrs]):
    for r, roi in enumerate(conn.region_labels):
        col = 1 + r % 4
        row = 1 + r // 4
        sl = True if r == 1 else False
        fig.add_trace(go.Scatter(x=freqs, y=plotout[i][r], opacity=0.8, marker=dict(color=cmap[i]), line=dict(width=(i+1)*0.5),
                                 name="year==" + str(t), legendgroup="year==" + str(t), showlegend=sl),
                      row=row, col=col)
fig.update_layout(title="Absolute spectra",template="plotly_white", height=1500, width=1100)
pio.write_html(fig, file=sim_fold + sim_tag + "/Rrev1b_SpectraAbs_lh_rois.html", auto_open=True, auto_play=False)
pio.write_image(fig, file=sim_fold + sim_tag + "/Rrev1b_SpectraAbs_lh_rois.svg")
pio.write_image(fig, file=sim_fold + sim_tag + "/Rrev1b_SpectraAbs_lh_rois.png")




## 1D. ANIMATION on time -
cmap = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
cmap_lr = list(itertools.chain(*zip(cmap, cmap)))

fig = make_subplots(rows=1, cols=2, column_titles=["Left hemisphere", "Right hemisphere"],
                    y_title="Normalized power", shared_yaxes=True)
# Add Spectra for the first timestep
for i, roi in enumerate(conn.region_labels):
    col = 1 if ("ctx-lh" in roi) or ("Left" in roi) else 2
    name = roi[7:] if "ctx" in roi else roi[5:] if "Left" in roi else roi[6:]
    sl = True if col == 1 else False
    fig.add_trace(go.Scatter(x=freqs, y=spectra[0][i], marker=dict(color=cmap_lr[i]),
                             name=name, legendgroup=name, showlegend=sl), row=1, col=col)
fig.update(frames=[go.Frame(data=[go.Scatter(y=spectra[i][ii]) for ii, roi in enumerate(conn.region_labels)],
                            traces=list(range(len(conn.region_labels))), name=str(t)) for i, t in enumerate(yrs)])
# CONTROLS : Add sliders and buttons
fig.update_layout(title="Absolute spectra",
    template="plotly_white", xaxis1=dict(title="Frequency (Hz)"), xaxis2=dict(title="Frequency (Hz)"),
    sliders=[dict(
        steps=[
            dict(method='animate', args=[[str(t)], dict(mode="immediate", frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                                        transition=dict(duration=0))], label=str(t)) for i, t in enumerate(yrs)],
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

pio.write_html(fig, file=sim_fold + sim_tag + "/Rrev1b_SpectraAbs_lh_rois_anim.html", auto_open=True, auto_play=False)















