
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from collections import Counter

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
simulations_tag = "PSEmpi_ADpgCirc_vCC-m06d12y2023-t16h.29m.08s"  # Tag cluster job
main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\PAPER\\R3ParamSpaces\\'

results = pd.read_csv(main_folder + simulations_tag + "/results.csv")

title = "indepParams_vCC"

# Compute avgrate and avgfc from pos/ant
results["avgrate"] = (results.avgrate_pos + results.avgrate_ant) / 2
results["avgfc"] = (results.avgfc_pos + results.avgfc_ant) / 2



# Columns for measures with colorscales
measures = ['fpeak', 'relpow_alpha', 'avgrate', 'avgfc']
tbar = ["Hz", "dB", "kHz", "PLV"]
cmaps = ["Plasma", "Viridis", "Cividis",  "Turbo"]

# Rows for explored parameters
params = ["maxHe", "minCie", "minCee", "maxTAU2SC"]
init_paramvalues = [3.25, 33.75, 108, "scaledWeights"]

results["maxHe_p"] = results.maxHe.values / init_paramvalues[0] * 100
results["minCie_p"] = - results.minCie.values / init_paramvalues[1] * 100
results["minCee_p"] = - results.minCee.values / init_paramvalues[2] * 100
results["maxTAU2SC_p"] = results.maxTAU2SC.values * 100

params = ["maxHe_p", "minCie_p", "minCee_p", "maxTAU2SC"]


# Identify fixed parameters and save in readme.txt
fixed_params = [Counter(results[param].values).most_common(1)[0][0] for param in params]
with open(main_folder + simulations_tag + "\\fixed_params.txt", "w") as f:
    f.write("Init parameters:  He" + str(init_paramvalues[0]) + "; Cie" + str(init_paramvalues[1]) + "; minCee" + str(init_paramvalues[2]) + "; maxTAU2SC" + str(init_paramvalues[3]))
    f.write("Fixed change limits:  maxHe" + str(fixed_params[0]) + "; minCie" + str(fixed_params[1]) + "; minCee" + str(fixed_params[2]) + "; maxTAU2SC" + str(fixed_params[3]))
    f.close()

x_bar = [-0.045+(j*1.045/len(measures)+1/len(measures)) for j, measure in enumerate(measures)]
x_bar = [0.195, 0.466, 0.735, 1.005]
fig = make_subplots(rows=len(params), cols=len(measures), x_title="Time (years)",
                    shared_xaxes=True, horizontal_spacing=0.075, shared_yaxes=True,
                    column_titles=["Frequency peak", "Relative alpha power", "Firing rate", "Functional connectivity"])

for i, param in enumerate(params):

    subset = results.loc[(results["cABinh"]!=0) & (results["cTAUinh"]!=0)].groupby([param, "time"]).mean().reset_index()
    # subset = results.loc[results["minHi"] == minHi]
    sl = True if i == 0 else False
    for j, measure in enumerate(measures):
        fig.add_trace(
            go.Heatmap(z=subset[measure].values, x=subset.time, y=subset[param].values,
                       zmin=results[measure].min(), zmax=results[measure].max(), showscale=sl, colorscale=cmaps[j],
                       colorbar=dict(title=tbar[j], thickness=8, x=x_bar[j])),
            row=i+1, col=j+1)

        # Add line for no change
        fig.add_hline(y=fixed_params[i], line=dict(width=6, color="red"), opacity=0.3, row=i+1, col=j+1)


tickvals = [[0, 20, 40, 60], [0, -20, -40, -60, -80], [0, -20, -40, -60, -80]]
ticktext = [[str(val) + "%  " + str(round(init_paramvalues[i] + init_paramvalues[i] * val/100, 2)) for val in set_tickvals] for i, set_tickvals in enumerate(tickvals)]

fig.update_layout(yaxis1=dict(title="H<sub>e</sub><br>upper limit", title_standoff=0, tickvals=tickvals[0], ticktext=ticktext[0]),
                  yaxis5=dict(title="C<sub>ip</sub><br>lower limit", title_standoff=0, tickvals=tickvals[1], ticktext=ticktext[1]),
                  yaxis9=dict(title="C<sub>ep</sub><br>lower limit", title_standoff=0, tickvals=tickvals[2], ticktext=ticktext[2]),
                  yaxis13=dict(title="SC<br>damage limit"), height=750, width=1200)

pio.write_html(fig, file=main_folder + simulations_tag + "/Fig4_PSE_ADpgCirc" + title + ".html", auto_open=True)
pio.write_image(fig, file=main_folder + simulations_tag + "/Fig4_PSE_ADpgCirc" + title + ".svg")



## AUX plots for inhibition disentangling
fig = make_subplots(rows=2, cols=len(measures), x_title="Time (years)",
                    shared_xaxes=True, horizontal_spacing=0.075, shared_yaxes=True,
                    column_titles=["Frequency peak", "Relative alpha power", "Firing rate", "Functional connectivity"])

param = "minCie_p"
for i, (a, b) in enumerate([ (0.4, 0), (0, 1.8)]):
    subset = results.loc[(results["cABinh"]==a) & (results["cTAUinh"]==b)].groupby([param, "time"]).mean().reset_index()
    # subset = results.loc[results["minHi"] == minHi]
    sl = True if i == 0 else False
    for j, measure in enumerate(measures):
        fig.add_trace(
            go.Heatmap(z=subset[measure].values, x=subset.time, y=subset[param].values,
                       zmin=results[measure].min(), zmax=results[measure].max(), showscale=sl, colorscale=cmaps[j],
                       colorbar=dict(title=tbar[j], thickness=8, x=x_bar[j])),
            row=i+1, col=j+1)

        # Add line for no change
        fig.add_hline(y=fixed_params[1], line=dict(width=6, color="red"), opacity=0.3, row=i+1, col=j+1)

fig.update_layout(yaxis1=dict(title="C<sub>ip</sub><br>lower limit<br>(cAB<sub>inh</sub>=0.4, cTAU<sub>inh</sub>=0)", title_standoff=0, tickvals=tickvals[1], ticktext=ticktext[1]),
                  yaxis5=dict(title="C<sub>ip</sub><br>lower limit<br>(cAB<sub>inh</sub>=0, cTAU<sub>inh</sub>=1.8)", title_standoff=0, tickvals=tickvals[1], ticktext=ticktext[1]),
                  height=500, width=1200)

pio.write_html(fig, file=main_folder + simulations_tag + "/Fig5_PSE_ADpgCirc" + title + "_aux.html", auto_open=True)
pio.write_image(fig, file=main_folder + simulations_tag + "/Fig5_PSE_ADpgCirc" + title + "_aux.svg")


## Supplementary figure supporting Working point selection
fig = make_subplots(rows=2, cols=len(measures), x_title="Time (years)", row_heights=[0.65, 0.35],
                    shared_xaxes=True, horizontal_spacing=0.075, shared_yaxes=True,
                    column_titles=["Frequency peak", "Relative alpha power", "Firing rate", "Functional connectivity"])

params = ["g", "s"]
fixed_params=[25, 20]
for i, param in enumerate(params):
    subset = results.groupby([param, "time"]).mean().reset_index()

    sl = True if i == 0 else False
    for j, measure in enumerate(measures):
        fig.add_trace(
            go.Heatmap(z=subset[measure].values, x=subset.time, y=subset[param].values,
                       zmin=results[measure].min(), zmax=results[measure].max(), showscale=sl, colorscale=cmaps[j],
                       colorbar=dict(title=tbar[j], thickness=8, x=x_bar[j])),
            row=i+1, col=j+1)

        # Add line for no change
        fig.add_hline(y=fixed_params[i], line=dict(width=6, color="red"), opacity=0.3, row=i+1, col=j+1)

fig.update_layout(yaxis1=dict(title="Coupling factor (g)", title_standoff=0, ),#tickvals=tickvals[1], ticktext=ticktext[1]),
                  yaxis5=dict(title="Conduction speed (s)", title_standoff=0, ),#tickvals=tickvals[1], ticktext=ticktext[1]),
                  height=500, width=1200)

pio.write_html(fig, file=main_folder + simulations_tag + "/Fig4SM_PSE_ADpgCirc" + title + "_WP.html", auto_open=True)
pio.write_image(fig, file=main_folder + simulations_tag + "/Fig4SM_PSE_ADpgCirc" + title + "_WP.svg")



# ## AUX plot: spatial dissociation with tau -- NOT USED @ 12/06/2023
# fig = make_subplots(rows=2, cols=2, shared_yaxes=True, x_title="Time (years)",
#                     column_titles=["Cip lower limit = -60%<br>(cABinh=0.4, cTAUinh=0)", "Cip lower limit = -60%<br>(cABinh=0, cTAUinh=1.8)" ])
# c1, c2, c3, c4, w1, w2 = px.colors.qualitative.Set1[1], px.colors.qualitative.Set1[0], \
#                  px.colors.qualitative.Pastel1[1], px.colors.qualitative.Pastel1[0], 2, 3
#
# param = "minCie_p"
# for i, (a, b) in enumerate([(0.4, 0), (0, 1.8)]):
#     subset = results.loc[(results["minCie_p"]==fixed_params[1]) & (results["cABinh"]==a) & (results["cTAUinh"]==b)].groupby([param, "time"]).mean().reset_index()
#
#     # add Firing rate curves
#     fig.add_trace(go.Scatter(x=subset.time, y=subset.avgrate_ant, name="Anterior", legendgroup="Anterior",
#                              showlegend=i==0, line=dict(color=c1, width=w1)), row=1, col=1+i)
#     fig.add_trace(go.Scatter(x=subset.time, y=subset.avgrate_pos, name="Posterior", legendgroup="Posterior",
#                              showlegend=i==0, line=dict(color=c2, width=w1)), row=1, col=1+i)
#     # add FC curves
#     fig.add_trace(go.Scatter(x=subset.time, y=subset.avgfc_ant, name="Anterior", legendgroup="Anterior",
#                              showlegend=False, line=dict(color=c3, width=w2)), row=2, col=1+i)
#     fig.add_trace(go.Scatter(x=subset.time, y=subset.avgfc_pos, name="Posterior", legendgroup="Posterior",
#                              showlegend=False, line=dict(color=c4, width=w2)), row=2, col=1+i)
#
# fig.update_layout(template="plotly_white", width=700, height=400,
#                   yaxis1=dict(title="Firing rate (Hz)"), yaxis3=dict(title="PLV"))
#
# pio.write_html(fig, file=main_folder + simulations_tag + "/FigX_PSE_ADpgCirc" + title + "_aux-AntPosCurves.html", auto_open=True)
# pio.write_image(fig, file=main_folder + simulations_tag + "/FigX_PSE_ADpgCirc" + title + "_aux-AntPosCurves.svg")


