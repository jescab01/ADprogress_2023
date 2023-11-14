
import numpy as np
import pandas as pd
import pickle

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
pse_tag = "PSEmpi_FreqCharts4.0_wNoise-m03d21y2023-t23h.04m.25s"  # Tag cluster job

pse_folder = 'E:\LCCN_Local\PycharmProjects\ADprogress\PAPER\R1FrequencyCharts\\'
df = pd.read_csv(pse_folder + pse_tag + "/results.csv")

# Average out repetitions
df_avg = df.groupby(['mode', "p", 'He', 'Hi', 'taue', 'taui', 'Cee', 'Cie', 'exp']).mean().reset_index()

cmax_freq, cmin_freq = max(df_avg["roi1_Hz"].values), min(df_avg["roi1_Hz"].values)
cmax_pow, cmin_pow = max(df_avg["roi1_auc"].values), min(df_avg["roi1_auc"].values)
cmax_fr, cmin_fr = max(df_avg["roi1_meanFR"].values), min(df_avg["roi1_meanFR"].values)





### PLOTTING
cmap = [px.colors.qualitative.Plotly[3]] + [px.colors.qualitative.Set2[5]] + [px.colors.qualitative.Plotly[2]]
fig = make_subplots(rows=3, cols=4, shared_yaxes=True, shared_xaxes=True)

###### First col: He
sub_df = df_avg.loc[(df_avg["exp"] == "exp1") & (df_avg["Cie"] == 33.5)]

fig.add_trace(go.Scatter(y=sub_df.roi1_Hz, x=sub_df.He, mode="markers+lines",
                         marker=dict(color=sub_df.roi1_Hz, cmax=cmax_freq, cmin=cmin_freq, colorscale="Plasma"),
                         line=dict(color="black", width=1),
                         name="Frequency peak", legendgroup="freq", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(y=sub_df.roi1_meanFR, x=sub_df.He, mode="markers+lines",
                         marker=dict(color=sub_df.roi1_meanFR, cmax=cmax_fr, cmin=cmin_fr, colorscale="Cividis"),
                         line=dict(color="black", width=1),
                         name="Firing rate", legendgroup="fr", showlegend=False),  row=2, col=1)
fig.add_trace(go.Scatter(y=sub_df.roi1_auc, x=sub_df.He, mode="markers+lines",
                         marker=dict(color=sub_df.roi1_auc, cmax=cmax_pow, cmin=cmin_pow, colorscale="Viridis"),
                         line=dict(color="black", width=1),
                         name="Power", legendgroup="pow", showlegend=False),  row=3, col=1)

# Add reference for default value
for i in range(3):
    fig.add_vline(x=3.25, line=dict(dash="dot", width=1, color="lightgray"), row=i+1, col=1)


###### Second col: Cip
sub_df = df_avg.loc[(df_avg["exp"] == "exp1") & (df_avg["He"] == 3.25)]

fig.add_trace(go.Scatter(y=sub_df.roi1_Hz, x=sub_df.Cie,mode="markers+lines",
                         marker=dict(color=sub_df.roi1_Hz, cmax=cmax_freq, cmin=cmin_freq, colorscale="Plasma"),
                         line=dict(color="black", width=1),
                         name="Frequency peak", legendgroup="freq", showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(y=sub_df.roi1_meanFR, x=sub_df.Cie,  mode="markers+lines",
                         marker=dict(color=sub_df.roi1_meanFR, cmax=cmax_fr, cmin=cmin_fr, colorscale="Cividis"),
                         line=dict(color="black", width=1),
                         name="Firing rate", legendgroup="fr", showlegend=False),  row=2, col=2)
fig.add_trace(go.Scatter(y=sub_df.roi1_auc, x=sub_df.Cie, mode="markers+lines",
                         marker=dict(color=sub_df.roi1_auc, cmax=cmax_pow, cmin=cmin_pow, colorscale="Viridis"),
                         line=dict(color="black", width=1),
                         name="Power", legendgroup="pow", showlegend=False), row=3, col=2)
# Add reference for default value
for i in range(3):
    fig.add_vline(x=33.75, line=dict(dash="dot", width=1, color="lightgray"), row=i+1, col=2)



###### Third col: Cep
sub_df = df_avg.loc[(df_avg["exp"] == "exp4") & (df_avg["Cie"]>33) & (df_avg["Cie"]<34)]

fig.add_trace(go.Scatter(y=sub_df.roi1_Hz, x=sub_df.Cee, mode="markers+lines",
                         marker=dict(color=sub_df.roi1_Hz, cmax=cmax_freq, cmin=cmin_freq, colorscale="Plasma"),
                         line=dict(color="black", width=1),
                         name="Frequency peak", legendgroup="freq", showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(y=sub_df.roi1_meanFR, x=sub_df.Cee,  mode="markers+lines",
                         marker=dict(color=sub_df.roi1_meanFR, cmax=cmax_fr, cmin=cmin_fr, colorscale="Cividis"),
                         line=dict(color="black", width=1),
                         name="Firing rate", legendgroup="fr", showlegend=False),  row=2, col=3)
fig.add_trace(go.Scatter(y=sub_df.roi1_auc, x=sub_df.Cee, mode="markers+lines",
                         marker=dict(color=sub_df.roi1_auc, cmax=cmax_pow, cmin=cmin_pow, colorscale="Viridis"),
                         line=dict(color="black", width=1),
                         name="Power", legendgroup="pow", showlegend=False), row=3, col=3)
# Add reference for default value
for i in range(3):
    fig.add_vline(x=108, line=dict(dash="dot", width=1, color="lightgray"), row=i+1, col=3)


###### Fourth col: p
sub_df = df_avg.loc[(df_avg["exp"] == "exp2") & (df_avg["Cee"]>108) & (df_avg["Cee"]<110)]
fig.add_trace(go.Scatter(y=sub_df.roi1_Hz, x=sub_df.p, mode="markers+lines",
                         marker=dict(color=sub_df.roi1_Hz, cmax=cmax_freq, cmin=cmin_freq, colorscale="Plasma"),
                         line=dict(color="black", width=1),
                         name="Frequency peak", legendgroup="freq", showlegend=False), row=1, col=4)
fig.add_trace(go.Scatter(y=sub_df.roi1_meanFR, x=sub_df.p,  mode="markers+lines",
                         marker=dict(color=sub_df.roi1_meanFR, cmax=cmax_fr, cmin=cmin_fr, colorscale="Cividis"),
                         line=dict(color="black", width=1),
                         name="Firing rate", legendgroup="fr", showlegend=False),  row=2, col=4)
fig.add_trace(go.Scatter(y=sub_df.roi1_auc, x=sub_df.p, mode="markers+lines",
                         marker=dict(color=sub_df.roi1_auc, cmax=cmax_pow, cmin=cmin_pow, colorscale="Viridis"),
                         line=dict(color="black", width=1),
                         name="Power", legendgroup="pow", showlegend=False), row=3, col=4)
# Add reference for default value
for i in range(3):
    fig.add_vline(x=0.22, line=dict(dash="dot", width=1, color="lightgray"), row=i+1, col=4)


fig.update_layout(template="plotly_white", height=700, width=1100,
                  yaxis1=dict(title="Frequency peak (Hz)"), xaxis1=dict(showticklabels=True),
                  yaxis2=dict(showticklabels=True), xaxis2=dict(showticklabels=True),
                  yaxis3=dict(showticklabels=True), xaxis3=dict(showticklabels=True),
                  yaxis4=dict(showticklabels=True), xaxis4=dict(showticklabels=True),
                  yaxis5=dict(title="Firing rate (kHz)"), xaxis5=dict(showticklabels=True),
                  yaxis6=dict(showticklabels=True), xaxis6=dict(showticklabels=True),
                  yaxis7=dict(showticklabels=True), xaxis7=dict(showticklabels=True),
                  yaxis8=dict(showticklabels=True), xaxis8=dict(showticklabels=True),
                  yaxis9=dict(title="Power (dB)"), xaxis9=dict(title="H<sub>e"),
                  yaxis10=dict(showticklabels=True), xaxis10=dict(title="C<sub>ip"),
                  yaxis11=dict(showticklabels=True), xaxis11=dict(title="C<sub>ep"),
                  yaxis12=dict(showticklabels=True), xaxis12=dict(title="p"))

pio.write_html(fig, file="E:\LCCN_Local\PycharmProjects\ADprogress\PAPER\Rreview\\figures\R2rev1.1_Heatmapsin2D.html", auto_open=True)
pio.write_image(fig, file="E:\LCCN_Local\PycharmProjects\ADprogress\PAPER\Rreview\\figures\R2rev1.1_Heatmapsin2D.svg")


