
import numpy as np
import pandas as pd

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


# Define PSE folder
simulations_tag = "PSEmpi_FreqCharts4.0_wNoise-m03d21y2023-t23h.04m.25s"  # Tag cluster job

main_folder = 'E:\LCCN_Local\PycharmProjects\ADprogress\PAPER\R1FrequencyCharts\\'
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")

# df.He = [float(row["He"]) if "classical" in row["mode"] else float(row["He"][1:-1]) for i, row in df.iterrows()]
# df.Hi = [float(row["Hi"]) if "classical" in row["mode"] else float(row["Hi"][1:-1]) for i, row in df.iterrows()]
# df.taue = [float(row["taue"]) if "classical" in row["mode"] else float(row["taue"][1:-1]) for i, row in df.iterrows()]
# df.taui = [float(row["taui"]) if "classical" in row["mode"] else float(row["taui"][1:-1]) for i, row in df.iterrows()]




# Average out repetitions
df_avg = df.loc[df["roi1_auc"] > 1e-8].groupby(['mode', "p", 'He', 'Hi', 'taue', 'taui', 'Cee', 'Cie', 'exp']).mean().reset_index()

# Define max and min values per measure
cmax_freq, cmin_freq = max(df_avg["roi1_Hz"].values), min(df_avg["roi1_Hz"].values)
cmax_pow, cmin_pow = max(df_avg["roi1_auc"].values), min(df_avg["roi1_auc"].values)
cmax_fr, cmin_fr = max(df_avg["roi1_meanFR"].values), min(df_avg["roi1_meanFR"].values)
cmax_rel, cmin_rel = 0.75, 0

rows = 5
fig = make_subplots(rows=rows, cols=4, vertical_spacing=0.075, horizontal_spacing=0.075, #column_titles=["Exp. 1", "Exp. 2", "Exp. 3", "Exp. 4"],
                    row_titles=["Frequency Peak", "Firing Rate", "Absolute power", "Rel. Alpha power", "Rel. Theta power"])

barlen, tk = 0.16, 8
# First col: Cie - He charts
sub_df = df_avg.loc[(df_avg["exp"] == "exp1")]
fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.He, y=sub_df.Cie,
                         colorbar=dict(thickness=tk, title="Hz", len=barlen, y=0.93),
                         zmax=cmax_freq, zmin=cmin_freq), row=1, col=1)

fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.He, y=sub_df.Cie,
                         colorbar=dict(thickness=tk, title="kHz", len=barlen, y=0.715),
                         zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=2, col=1)

fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.He, y=sub_df.Cie,
                         colorbar=dict(thickness=tk, title="dB", len=barlen, y=0.5),
                         zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=3, col=1)

fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.He, y=sub_df.Cie,
                         colorbar=dict(thickness=tk, title="dB", len=barlen, y=0.285),
                        zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=4, col=1)

fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.He, y=sub_df.Cie,
                         colorbar=dict(thickness=tk, title="dB", len=barlen, y=0.07),
                         zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=5, col=1)

# fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.He, y=sub_df.Cie,
#                          colorbar=dict(thickness=tk, title="dB", len=barlen, y=0.07),
#                          zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=5, col=1)




for i in range(rows):
    # sl = True if i == 0 else False
    fig.add_trace(go.Scatter(x=[3.25], y=[33.75], mode="markers",
                             marker=dict(symbol="circle-open", color="red"),
                             legendgroup="dot", name="initRef", showlegend=False), row=i+1, col=1)

# Second col: plot Cie - Cee
sub_df = df_avg.loc[(df_avg["exp"] == "exp4")]
fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.Cee, y=sub_df.Cie, showscale=False,
                         zmax=cmax_freq, zmin=cmin_freq), row=1, col=2)

fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.Cee, y=sub_df.Cie, showscale=False,
                         zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=2, col=2)

fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.Cee, y=sub_df.Cie, showscale=False,
                         zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=3, col=2)

fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.Cee, y=sub_df.Cie, showscale=False,
                         zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=4, col=2)

fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.Cee, y=sub_df.Cie, showscale=False,
                         zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=5, col=2)

# fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.Cee, y=sub_df.Cie, showscale=False,
#                          zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=5, col=4)

for i in range(rows):
    sl = True if i == 0 else False
    fig.add_trace(go.Scatter(x=[108], y=[33.75], mode="markers",
                             marker=dict(symbol="circle-open", color="red"),
                             legendgroup="dot", name="initRef", showlegend=False), row=i+1, col=2)

# Third col: Plot Cee - p (input)
sub_df = df_avg.loc[(df_avg["exp"] == "exp2")]
fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.Cee, y=sub_df.p, showscale=False,
                         zmax=cmax_freq, zmin=cmin_freq), row=1, col=3)

fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.Cee, y=sub_df.p, showscale=False,
                         zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=2, col=3)

fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.Cee, y=sub_df.p, showscale=False,
                         zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=3, col=3)

fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.Cee, y=sub_df.p, showscale=False,
                        zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=4, col=3)

fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.Cee, y=sub_df.p, showscale=False,
                         zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=5, col=3)

# fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.p, y=sub_df.Cee, showscale=False,
#                          zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=5, col=2)



for i in range(rows):
    sl = True if i == 0 else False
    fig.add_trace(go.Scatter(x=[108], y=[0.22], mode="markers",
                             marker=dict(symbol="circle-open", color="red"),
                             legendgroup="dot", name="initRef", showlegend=False), row=i+1, col=3)

# Fourth col: Plot Cie - p (input)
sub_df = df_avg.loc[(df_avg["exp"] == "exp3")]
fig.add_trace(go.Heatmap(z=sub_df.roi1_Hz, x=sub_df.p, y=sub_df.Cie, showscale=False,
                         zmax=cmax_freq, zmin=cmin_freq), row=1, col=4)

fig.add_trace(go.Heatmap(z=sub_df.roi1_meanFR, x=sub_df.p, y=sub_df.Cie, showscale=False,
                         zmax=cmax_fr, zmin=cmin_fr, colorscale="Cividis"), row=2, col=4)

fig.add_trace(go.Heatmap(z=sub_df.roi1_auc, x=sub_df.p, y=sub_df.Cie, showscale=False,
                         zmax=cmax_pow, zmin=cmin_pow, colorscale="Viridis"), row=3, col=4)

fig.add_trace(go.Heatmap(z=sub_df.roi1_aucAlpha, x=sub_df.p, y=sub_df.Cie, showscale=False,
                        zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=4, col=4)

fig.add_trace(go.Heatmap(z=sub_df.roi1_aucTheta, x=sub_df.p, y=sub_df.Cie, showscale=False,
                         zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=5, col=4)

# fig.add_trace(go.Heatmap(z=sub_df.roi1_aucBeta, x=sub_df.p, y=sub_df.Cie, showscale=False,
#                          zmax=cmax_rel, zmin=cmin_rel, colorscale="Viridis"), row=5, col=3)


for i in range(rows):
    # sl = True if i == 0 else False
    fig.add_trace(go.Scatter(x=[0.22], y=[33.75], mode="markers",
                             marker=dict(symbol="circle-open", color="red"),
                             legendgroup="dot", name="initRef", showlegend=False), row=i+1, col=4)



standoff_y, standoff_x = 0, 10
fig.update_layout(template="plotly_white",
                  xaxis1=dict(title="H<sub>e</sub>", title_standoff=standoff_x), yaxis1=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis2=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis2=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis3=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis3=dict(title="p", title_standoff=standoff_y),
                  xaxis4=dict(title="p", title_standoff=standoff_x), yaxis4=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),

                  xaxis5=dict(title="H<sub>e</sub>", title_standoff=standoff_x), yaxis5=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis6=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis6=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis7=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis7=dict(title="p", title_standoff=standoff_y),
                  xaxis8=dict(title="p", title_standoff=standoff_x), yaxis8=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),

                  xaxis9=dict(title="H<sub>e</sub>", title_standoff=standoff_x), yaxis9=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis10=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis10=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis11=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis11=dict(title="p", title_standoff=standoff_y),
                  xaxis12=dict(title="p", title_standoff=standoff_x), yaxis12=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),

                  xaxis13=dict(title="H<sub>e</sub>", title_standoff=standoff_x), yaxis13=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis14=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis14=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis15=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis15=dict(title="p", title_standoff=standoff_y),
                  xaxis16=dict(title="p", title_standoff=standoff_x), yaxis16=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),

                  xaxis17=dict(title="H<sub>e</sub>", title_standoff=standoff_x), yaxis17=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis18=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis18=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                  xaxis19=dict(title="C<sub>ep</sub>", title_standoff=standoff_x), yaxis19=dict(title="p", title_standoff=standoff_y),
                  xaxis20=dict(title="p", title_standoff=standoff_x), yaxis20=dict(title="C<sub>ip</sub>", title_standoff=standoff_y),
                   width=800, height=900)

pio.write_html(fig, file=main_folder + simulations_tag + "/R1_FreqCharts.html", auto_open=True)
pio.write_image(fig, file=main_folder + simulations_tag + "/R1_FreqCharts.svg")




