
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from collections import Counter

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

# Define PSE folder
simulations_tag = "PSEmpi_BraakStages-m11d13y2023-t15h.52m.30s"  # Tag cluster job
main_folder = 'E:\\LCCN_Local\PycharmProjects\ADprogress\PAPER\Rreview\\'

results = pd.read_csv(main_folder + simulations_tag + "/results.csv")


# Prepare the dataset: compute orders per timestep
results["braak"] = [str(list(row_data.sort_values(ascending=False).index))
                    for i, row_data in results.iloc[:, 9:14].iterrows()]

# Determine the main body of propagation for dominance consideration (all Rxs above 5% and bellow 95% of concentration).
results["mainprop"] = [0.5 if (row_data.mean() < 0.5) else 0.9 if (row_data.mean() < 0.9) else 0
                       for i, row_data in results.iloc[:, 9:14].iterrows()]



## PART B: Posterior-Anterior for AB-measures
modes = ['fixed', "ab_ant", "ab_pos", 'ab_rand', 'tau_ant', "tau_pos", "tau_rand"]

ant_post = pd.DataFrame(
    [[mode, rep] + list(results.loc[(results["mode"] == mode) & (results["r"] == rep),
                                    ["avgrate_pos", "avgrate_ant", "avgfc_pos", "avgfc_ant"]].reset_index().idxmax().values)
               for mode in modes for rep in set(results.r)],
    columns=["mode", "rep", "t_max", "t_maxrate_pos", "t_maxrate_ant", "t_maxfc_pos", "t_maxfc_ant"])



## B1. Functional connectivity violin plot
fig = go.Figure()

pointpos_ant, pointpos_post, size = [-0.9, -0.5, -0.95, -0.65], [1.1, 0.65, 0.9, 0.9], 4

for i, mode in enumerate(modes):

    sl = True if i == 0 else False

    subset = ant_post.loc[ant_post["mode"] == mode]

    fig.add_trace(go.Violin(x=subset["mode"].values, y=subset.t_maxfc_ant.values,
                            legendgroup="Anterior", scalegroup="Anterior", name="Anterior",
                            side="negative",  line_color="cornflowerblue",
                            marker=dict(size=size, opacity=0.7), showlegend=sl))

    fig.add_trace(go.Violin(x=subset["mode"].values, y=subset.t_maxfc_pos.values,
                            legendgroup="Posterior", scalegroup="Posterior", name="Posterior",
                            side="positive",  line_color="firebrick",
                            marker=dict(size=size, opacity=0.7), showlegend=sl))


# update characteristics shared by all traces
fig.update_traces(meanline_visible=True,
                  scalemode='count') #scale violin plot area with total count

fig.update_layout(template="plotly_white", height=500, width=600,
                  yaxis=dict(title="Functional Connectivity<br>Time to peak (years)", range=[5,30]),
                  xaxis1=dict(title=None, tickvals=modes, tickangle=25,
                              ticktext=["Fixed", "A\uA7B5 anterior", "A\uA7B5 posterior", "A\uA7B5 random",
                                        "Tau anterior", "Tau posterior", "Tau random"]), violingroupgap=0.1)


pio.write_html(fig, main_folder + simulations_tag + "\\R2rev7a_antpost-fc.html", auto_open=True)
pio.write_image(fig, main_folder + simulations_tag + "\\R2rev7a_antpost-fc.svg")



## B2. Firing rate box plot
fig = go.Figure()

subset = ant_post.loc[ant_post["mode"].isin(modes)]

fig.add_trace(go.Box(x=subset["mode"].values, y=subset.t_maxrate_ant.values, legendgroup="Anterior", name="Anterior",
                        marker=dict(size=size, opacity=0.7, color="cornflowerblue"), showlegend=True))

fig.add_trace(go.Box(x=subset["mode"].values, y=subset.t_maxrate_pos.values, legendgroup="Posterior", name="Posterior",
                         marker=dict(size=size, opacity=0.7, color="firebrick"), showlegend=True))


fig.update_layout(template="plotly_white", height=500, width=600, legend=dict(orientation="h", x=0.4, y=1.1),
                  yaxis=dict(title="Firing rate<br>Time to peak (years)", range=[5,30]), boxgroupgap=0.2, boxgap=0.75,
                  xaxis=dict(title=None, tickvals=modes, tickangle=25,
                              ticktext=["Fixed", "A\uA7B5 anterior", "A\uA7B5 posterior", "A\uA7B5 random",
                                        "Tau anterior", "Tau posterior", "Tau random"]), boxmode="group")


pio.write_html(fig, main_folder + simulations_tag + "\\R2rev7b_antpost-rate.html", auto_open=True)
pio.write_image(fig, main_folder + simulations_tag + "\\R2rev7b_antpost-rate.svg")




## B2b. Plot examples of each seeding strategy
fig = make_subplots(rows=7, cols=2,
                    horizontal_spacing=0.15, row_titles=["Fixed", "A\uA7B5 anterior", "A\uA7B5 posterior",
                                                         "A\uA7B5 random", "Tau anterior", "Tau posterior", "Tau random"])

for i, mode in enumerate(modes):
    sl = True if i==0 else False

    sim = results.loc[(results["mode"] == mode) & (results["r"] == 0)]

    fig.add_trace(go.Scatter(x=sim.time, y=sim.avgrate_pos, name="Posterior", legendgroup="pos",
                             line=dict(width=2, color="firebrick"), opacity=0.8, showlegend=sl), row=1+i, col=1)
    fig.add_trace(go.Scatter(x=sim.time, y=sim.avgrate_ant, name="Anterior", legendgroup="ant",
                             line=dict(width=2, color="cornflowerblue"), opacity=0.8, showlegend=sl), row=1+i, col=1)
    fig.add_trace(go.Scatter(x=sim.time, y=sim.avgfc_pos, legendgroup="pos",
                             line=dict(width=2, color="firebrick"), opacity=0.7, showlegend=False), row=1+i, col=2)
    fig.add_trace(go.Scatter(x=sim.time, y=sim.avgfc_ant, legendgroup="ant",
                             line=dict(width=2, color="cornflowerblue"), opacity=0.7, showlegend=False), row=1+i, col=2)

fig.update_layout(template="plotly_white", height=1000, width=600,
                  legend=dict(orientation="h", xanchor="center", x=0.5, y=1.1),
                  xaxis13=dict(title="Time (years)"), xaxis14=dict(title="Time (years)"),
                  yaxis1=dict(range=[0.002,0.005]), yaxis2=dict(range=[0.2,0.9]),
                  yaxis3=dict(range=[0.002,0.005]), yaxis4=dict(range=[0.2,0.9]),
                  yaxis5=dict(range=[0.002,0.005]), yaxis6=dict(range=[0.2,0.9]),
                  yaxis7=dict(title="Firing rate (kHz)", range=[0.002, 0.005]),
                  yaxis8=dict(title="PLV", range=[0.2,0.9]),
                  yaxis9=dict(range=[0.002,0.005]), yaxis10=dict(range=[0.2,0.9]),
                  yaxis11=dict(range=[0.002,0.005]), yaxis12=dict(range=[0.2,0.9]),
                  yaxis13=dict(range=[0.002,0.005]), yaxis14=dict(range=[0.2,0.9]),)

pio.write_html(fig, main_folder + simulations_tag + "\\R2rev7c_rateFc_time.html", auto_open=True)
pio.write_image(fig, main_folder + simulations_tag + "\\R2rev7c_rateFc_time.svg")

