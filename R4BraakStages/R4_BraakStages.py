
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
simulations_tag = "PSEmpi_BraakStages-m06d14y2023-t14h.19m.00s"  # Tag cluster job
main_folder = 'E:\\LCCN_Local\PycharmProjects\\ADprogress\PAPER\\R4BraakStages\\'

results = pd.read_csv(main_folder + simulations_tag + "/results.csv")


# Prepare the dataset: compute orders per timestep
results["braak"] = [str(list(row_data.sort_values(ascending=False).index))
                    for i, row_data in results.iloc[:, 9:14].iterrows()]

# Determine the main body of propagation for dominance consideration (all Rxs above 5% and bellow 95% of concentration).
results["mainprop"] = [0.5 if (row_data.mean() < 0.5) else 0.9 if (row_data.mean() < 0.9) else 0
                       for i, row_data in results.iloc[:, 9:14].iterrows()]



##  Approach 2: Stacked bars with dominance (% of simulations) per mode and sequence
braak_dom_persim = pd.DataFrame(
    [(mode, rep, braak, count, count/len(results.loc[(results["mode"] == mode) & (results["r"] == rep)]))
               for mode in set(results["mode"]) for rep in set(results.r)
               for braak, count in Counter(results.loc[(results["mode"] == mode) & (results["r"] == rep), "braak"]).most_common(1)],
    columns=["mode", "rep", "braak", "count", "prop"])

braak_dom = pd.DataFrame([(mode, braak, len(braak_dom_persim.loc[(braak_dom_persim["braak"] == braak) & (braak_dom_persim["mode"] == mode)]))
         for i, braak in enumerate(sorted(set(braak_dom_persim.braak))) for mode in ["fixed", "ab_rand", "tau_rand", "rand"]], columns=["mode", "braak", "count"])


fig = px.bar(braak_dom, x="mode", y="count", color="braak",
             color_discrete_sequence=px.colors.qualitative.Set2 + px.colors.qualitative.Pastel_r[1:] + px.colors.qualitative.Pastel1 )
fig.update_layout(template="plotly_white", width=500, height=400,
                  legend=dict(title="Braak sequence"), yaxis=dict(title="Dominance (%)"),
                  xaxis=dict(title=None, tickvals=["fixed", "ab_rand", "tau_rand", "rand"], ticktext=["Fixed", "A\uA7B5 random", "tau random", "Random"]))

pio.write_html(fig, main_folder + simulations_tag + "\\Fig8a_globDominance_stacked.html", auto_open=True)
pio.write_image(fig, main_folder + simulations_tag + "\\Fig8a_globDominance_stacked.svg")




## Plot examples of the most common sequences v1: Concentration and Braid plots
cmap1 = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel_r[1:] + px.colors.qualitative.Pastel1

sequences = [("fixed", "['rI', 'rII', 'rIII', 'rIV', 'rV']", cmap1[0]),
             ("ab_rand", "['rI', 'rIII', 'rII', 'rIV', 'rV']", cmap1[3]),
             ("tau_rand", "['rV', 'rIV', 'rII', 'rIII', 'rI']", cmap1[17]),
             ("rand", "['rV', 'rIV', 'rIII', 'rI', 'rII']", cmap1[18])]

cmap, w, w2, extra = px.colors.sequential.gray_r[::-2], 2, 1, 0

fig = make_subplots(rows=len(sequences), cols=1, y_title="TAUt concentration (M)", x_title="Time (years)")
for i, (mode, braak, color) in enumerate(sequences):

    # Extract from results a specific simulation as example
    max_count = braak_dom_persim.loc[(braak_dom_persim["mode"] == mode) & (braak_dom_persim["braak"] == braak), "count"].max()
    sel_rep = braak_dom_persim.loc[(braak_dom_persim["mode"] == mode) & (braak_dom_persim["braak"] == braak) & (braak_dom_persim["count"] == max_count), "rep"].values[-1]

    sim = results.loc[(results["mode"] == mode) & (results["r"] == sel_rep)]

    # Add traces
    fig.add_trace(go.Scatter(x=sim.time, y=sim.rI, name="rI", legendgroup="rI", showlegend=i==0, line=dict(color=cmap[0], width=w+0*extra)), row=1+i, col=1)
    fig.add_trace(go.Scatter(x=sim.time, y=sim.rII, name="rII", legendgroup="rII", showlegend=i==0, line=dict(color=cmap[1], width=w+1*extra)), row=1+i, col=1)
    fig.add_trace(go.Scatter(x=sim.time, y=sim.rIII, name="rIII", legendgroup="rIII", showlegend=i==0, line=dict(color=cmap[2], width=w+2*extra)), row=1+i, col=1)
    fig.add_trace(go.Scatter(x=sim.time, y=sim.rIV, name="rIV", legendgroup="rIV", showlegend=i==0, line=dict(color=cmap[3], width=w+3*extra)), row=1+i, col=1)
    fig.add_trace(go.Scatter(x=sim.time, y=sim.rV, name="rV", legendgroup="rV", showlegend=i==0, line=dict(color=cmap[4], width=w+4*extra)), row=1+i, col=1)

    # Add braak sequence as title per plot in color
    fig.add_annotation(text=braak, font=dict(color="black", size=10), bgcolor=color, x=7, y=1, showarrow=False, row=1+i, col=1)

fig.update_traces(opacity=0.7)
fig.update_layout(template="plotly_white", legend=dict(orientation="h", xanchor="center", x=0.5, y=1.1), height=600, width=500)

pio.write_html(fig, main_folder + simulations_tag + "\\Fig8b_globDominance_time.html", auto_open=True)
pio.write_image(fig, main_folder + simulations_tag + "\\Fig8b_globDominance_time.svg")






## PART 2: Posterior-Anterior for AB-measures
ant_post = pd.DataFrame(
    [[mode, rep] + list(results.loc[(results["mode"] == mode) & (results["r"] == rep),
                                    ["avgrate_pos", "avgrate_ant", "avgfc_pos", "avgfc_ant"]].reset_index().idxmax().values)
               for mode in ['fixed', "ab_ant", "ab_pos", 'ab_rand'] for rep in set(results.r)],
    columns=["mode", "rep", "t_max", "t_maxrate_pos", "t_maxrate_ant", "t_maxfc_pos", "t_maxfc_ant"])


# 1. Calculating differences for violin plots
# ant_post["t_diff_rate"] = [row["t_maxrate_ant"]-row["t_maxrate_pos"] for i, row in ant_post.iterrows()]
# ant_post["t_diff_fc"] = [row["t_maxfc_ant"]-row["t_maxfc_pos"] for i, row in ant_post.iterrows()]
# fig = px.violin(ant_post, x="mode", y="t_diff_fc", points="all")
# fig.show("browser")


## 2a Functional connectivity violin plot
fig = go.Figure()
pointpos_ant, pointpos_post, size = [-0.9, -0.5, -0.95, -0.65], [1.1, 0.65, 0.9, 0.9], 4
for i, mode in enumerate(['fixed', "ab_ant", "ab_pos", 'ab_rand']):
    sl = True if i == 0 else False
    subset = ant_post.loc[ant_post["mode"] == mode]
    fig.add_trace(go.Violin(x=subset["mode"].values, y=subset.t_maxfc_ant.values, legendgroup="Anterior", scalegroup="Anterior", name="Anterior",
                            side="negative", pointpos=pointpos_ant[i], line_color="cornflowerblue", marker=dict(size=size, opacity=0.7), showlegend=sl))
    fig.add_trace(go.Violin(x=subset["mode"].values, y=subset.t_maxfc_pos.values, legendgroup="Posterior", scalegroup="Posterior", name="Posterior",
                            side="positive", pointpos=pointpos_post[i], line_color="firebrick", marker=dict(size=size, opacity=0.7), showlegend=sl))
# update characteristics shared by all traces
fig.update_traces(meanline_visible=True,
                  points="all", # show all points
                  jitter=0.05,  # add some jitter on points for better visibility
                  scalemode='count') #scale violin plot area with total count

fig.update_layout(template="plotly_white", height=500, width=700,
                  yaxis=dict(title="Functional Connectivity<br>Time to peak (years)"),
                  xaxis1=dict(title=None, tickvals=['fixed', "ab_ant", "ab_pos", 'ab_rand'], tickangle=0,
                              ticktext=["Fixed", "A\uA7B5 anterior", "A\uA7B5 posterior", "A\uA7B5 random"]),
    violingap=0, violingroupgap=0, violinmode='overlay')

pio.write_html(fig, main_folder + simulations_tag + "\\Fig9a_antpost-fc_time.html", auto_open=True)
pio.write_image(fig, main_folder + simulations_tag + "\\Fig9a_antpost-fc_time.svg")



## 2b Firing rate box plot
fig = go.Figure()

subset = ant_post.loc[ant_post["mode"].isin(['fixed', "ab_ant", "ab_pos", 'ab_rand'])]

fig.add_trace(go.Box(x=subset["mode"].values, y=subset.t_maxrate_ant.values, legendgroup="Anterior", name="Anterior",
                        marker=dict(size=size, opacity=0.7, color="cornflowerblue"), showlegend=True))

fig.add_trace(go.Box(x=subset["mode"].values, y=subset.t_maxrate_pos.values, legendgroup="Posterior", name="Posterior",
                         marker=dict(size=size, opacity=0.7, color="firebrick"), showlegend=True))


fig.update_layout(template="plotly_white", height=450, width=450, legend=dict(orientation="h", x=0.4, y=1.1),
                  yaxis=dict(title="Firing rate<br>Time to peak (years)"),
                  xaxis=dict(title=None, tickvals=['fixed', "ab_ant", "ab_pos", 'ab_rand'], tickangle=25,
                              ticktext=["Fixed", "A\uA7B5 anterior", "A\uA7B5 posterior", "A\uA7B5 random"]), boxmode="group")

pio.write_html(fig, main_folder + simulations_tag + "\\Fig9b_antpost-rate_time.html", auto_open=True)
pio.write_image(fig, main_folder + simulations_tag + "\\Fig9b_antpost-rate_time.svg")




## Plot examples of each seeding strategy
fig = make_subplots(rows=4, cols=2,
                    horizontal_spacing=0.15, row_titles=["Fixed", "A\uA7B5 anterior", "A\uA7B5 posterior", "A\uA7B5 random"])

for i, mode in enumerate(['fixed', "ab_ant", "ab_pos", 'ab_rand']):
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

fig.update_layout(template="plotly_white", legend=dict(orientation="h", xanchor="center", x=0.5, y=1.1), height=700, width=600,
                  xaxis7=dict(title="Time (years)"), xaxis8=dict(title="Time (years)"),
                  yaxis5=dict(title="Firing rate (kHz)"), yaxis6=dict(title="PLV"))

pio.write_html(fig, main_folder + simulations_tag + "\\Fig9c_rateFc_time.html", auto_open=True)
pio.write_image(fig, main_folder + simulations_tag + "\\Fig9c_rateFc_time.svg")



## STATS
import pingouin as pg

ttest = list()
for mode in ["fixed", "ab_ant", "ab_pos", 'ab_rand']:
    subset = ant_post.loc[ant_post["mode"] == mode]
    temp = pg.ttest(x=subset.t_maxrate_ant, y=subset.t_maxrate_pos, paired=True)
    ttest.append([mode, "rate"] + list(temp.values[0]))

    temp = pg.ttest(x=subset.t_maxfc_ant, y=subset.t_maxfc_pos, paired=True)
    ttest.append([mode, "fc"] + list(temp.values[0]))

ttest = pd.DataFrame(ttest, columns=["mode", "measure"] + list(temp.columns.values))
ttest=ttest[1:]

# ## 1b Firing rate violin plot
# fig = go.Figure()
# pointpos_ant, pointpos_post, size = [-0.9, -0.5, -0.95, -0.65], [1.1, 0.65, 0.9, 0.9], 4
# for i, mode in enumerate(['fixed', "ab_ant", "ab_pos", 'ab_rand']):
#
#     sl = True if i == 0 else False
#
#     subset = ant_post.loc[ant_post["mode"] == mode]
#
#     fig.add_trace(go.Violin(x=subset["mode"].values, y=subset.t_maxrate_ant.values, legendgroup="Anterior", scalegroup="Anterior", name="Anterior",
#                             side="negative", pointpos=pointpos_ant[i], line_color="cornflowerblue", marker=dict(size=size, opacity=0.7), showlegend=sl))
#
#     fig.add_trace(go.Violin(x=subset["mode"].values, y=subset.t_maxrate_pos.values, legendgroup="Posterior", scalegroup="Posterior", name="Posterior",
#                             side="positive", pointpos=pointpos_post[i], line_color="indianred", marker=dict(size=size, opacity=0.7), showlegend=sl))
#
# # update characteristics shared by all traces
# fig.update_traces(meanline_visible=True,
#                   points="all", # show all points
#                   jitter=0.05,  # add some jitter on points for better visibility
#                   scalemode='count') #scale violin plot area with total count
#
# fig.update_layout(template="plotly_white", height=500, width=700,
#                   yaxis=dict(title="Firing rate<br>Time to peak (years)"),
#                   xaxis1=dict(title=None, tickvals=['fixed', "ab_ant", "ab_pos", 'ab_rand'], tickangle=0,
#                               ticktext=["Fixed", "A\uA7B5 anterior", "A\uA7B5 posterior", "A\uA7B5 random"]),
#     violingap=0, violingroupgap=0, violinmode='overlay')
#
# fig.show("browser")












# ## Plot examples of the most common sequences v2: Concentration and Braid plots
# cmap = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
#
# sequences = [("fixed", "['rI', 'rII', 'rIII', 'rIV', 'rV']", cmap[0]),
#              ("ab_rand", "['rI', 'rIII', 'rII', 'rIV', 'rV']", cmap[1]),
#              ("tau_rand", "['rIV', 'rV', 'rII', 'rI', 'rIII']", cmap[5]),
#              ("rand", "['rV', 'rIV', 'rII', 'rIII', 'rI']", cmap[10])]
#
# cmap, w, w2 = px.colors.sequential.gray_r[::-2], 1.5, 1
#
# def braiddata(time_data):
#
#     rxs_avg = time_data / time_data[-1, :] * 100
#
#     # For each percentage; tell me in what time it was rebased.
#     percxRx_TIME = np.asarray([[np.min(np.argwhere(rx >= perc)) for rx in np.transpose(rxs_avg)] for perc in range(1, 100)])
#     # columns=["rI", "rII", "rIII", "rIV", "rV"])
#     # Tell me the rx order based on perc_time
#     percxRx_ORDER_braiddiag = np.asarray([np.argsort(np.argsort(row)) for row in percxRx_TIME])
#     percxRx_ORDER_braiddiag = percxRx_ORDER_braiddiag.astype(float)
#
#     # Take care with equalities in perc_time that are sorted randomly
#     for i, row in enumerate(percxRx_TIME):
#         if len(set(row)) < 5:
#             for val in set(row):
#                 if len(row[row == val]) > 1:
#                     percxRx_ORDER_braiddiag[i, row == val] = \
#                         np.tile(np.average(percxRx_ORDER_braiddiag[i, row == val]), (len(row[row == val])))
#
#     return percxRx_ORDER_braiddiag
#
# fig = make_subplots(rows=len(sequences), cols=2, y_title="TAUt concentration (M)", x_title="Time (years)",
#                     column_widths=[0.7, 0.3])
# for i, (mode, braak, color) in enumerate(sequences):
#
#     # Extract from results a specific simulation as example
#     max_count = braak_dom_persim.loc[(braak_dom_persim["mode"] == mode) & (braak_dom_persim["braak"] == braak), "count"].max()
#     sel_rep = braak_dom_persim.loc[(braak_dom_persim["mode"] == mode) & (braak_dom_persim["braak"] == braak) & (braak_dom_persim["count"] == max_count), "rep"].values[-1]
#
#     sim = results.loc[(results["mode"] == mode) & (results["r"] == sel_rep)]
#
#     # Add traces
#     fig.add_trace(go.Scatter(x=sim.time, y=sim.rI, name="rI", legendgroup="rI", showlegend=i==0, line=dict(color=cmap[0], width=w)), row=1+i, col=1)
#     fig.add_trace(go.Scatter(x=sim.time, y=sim.rII, name="rII", legendgroup="rII", showlegend=i==0, line=dict(color=cmap[1], width=w)), row=1+i, col=1)
#     fig.add_trace(go.Scatter(x=sim.time, y=sim.rIII, name="rIII", legendgroup="rIII", showlegend=i==0, line=dict(color=cmap[2], width=w)), row=1+i, col=1)
#     fig.add_trace(go.Scatter(x=sim.time, y=sim.rIV, name="rIV", legendgroup="rIV", showlegend=i==0, line=dict(color=cmap[3], width=w)), row=1+i, col=1)
#     fig.add_trace(go.Scatter(x=sim.time, y=sim.rV, name="rV", legendgroup="rV", showlegend=i==0, line=dict(color=cmap[4], width=w)), row=1+i, col=1)
#
#     ## Create Braid Plots
#     percxRx_ORDER_braiddiag = braiddata(np.array(sim.iloc[:, 9:14]))
#     fig.add_trace(go.Scatter(x=percxRx_ORDER_braiddiag[:, 0], y=list(range(1, 100)), name="rI", legendgroup="rI",
#                              showlegend=False, line=dict(color=cmap[0], width=w2)), row=1+i, col=2)
#     fig.add_trace(go.Scatter(x=percxRx_ORDER_braiddiag[:, 1], y=list(range(1, 100)), name="rII", legendgroup="rII",
#                              showlegend=False, line=dict(color=cmap[1], width=w2)), row=1+i, col=2)
#     fig.add_trace(go.Scatter(x=percxRx_ORDER_braiddiag[:, 2], y=list(range(1, 100)), name="rIII", legendgroup="rIII",
#                              showlegend=False, line=dict(color=cmap[2], width=w2)), row=1+i, col=2)
#     fig.add_trace(go.Scatter(x=percxRx_ORDER_braiddiag[:, 3], y=list(range(1, 100)), name="rIV", legendgroup="rIV",
#                              showlegend=False, line=dict(color=cmap[3], width=w2)), row=1+i, col=2)
#     fig.add_trace(go.Scatter(x=percxRx_ORDER_braiddiag[:, 4], y=list(range(1, 100)), name="rV", legendgroup="rV",
#                              showlegend=False, line=dict(color=cmap[4], width=w2)), row=1+i, col=2)
#
# fig.update_layout(template="plotly_white", legend=dict(orientation="h", xanchor="center", x=0.5, y=1.1), height=600, width=800)
# fig.show("browser")



# ## Approach 1: Violin plots w/ time presence (%) of each sequence per simulation
#
# # Create a new df with braak orders and probabilities per simulation
# # loop over simulations, extract all possible orders, and calculate their percentages, then add to the list.
# braak_props = [(mode, rep, braak, count, count/len(results.loc[(results["mode"] == mode) & (results["r"] == rep) & (results["mainprop"] > 0)]))
#                for mode in set(results["mode"]) for rep in set(results.r)
#                for braak, count in Counter(results.loc[(results["mode"] == mode) & (results["r"] == rep) & (results["mainprop"] > 0), "braak"]).items()]
#
# df_braak = pd.DataFrame(braak_props, columns=["mode", "rep", "braak", "count", "prop"])
# df_braak["color"] = np.nan
#
# cmap = px.colors.qualitative.Plotly
# for i, braak in enumerate(sorted(set(df_braak.braak))):
#     df_braak.loc[df_braak["braak"] == braak, "color"] = cmap[i%len(cmap)]
#
# # Violin plot with combinations per type
# fig = make_subplots(rows=1, cols=4, shared_yaxes=True, column_widths=[0.15, 0.15, 0.35, 0.35],
#                     y_title="Sequence domination probability")
# cmap = px.colors.qualitative.Plotly
#
# braaks = []
# for j, mode in enumerate(["fixed", "ab_rand", "tau_rand", "rand"]):
#
#     sub1 = df_braak[(df_braak["mode"] == mode)]
#
#     for i, braak in enumerate(sorted(set(sub1.braak))):
#
#         sub2 = sub1[(sub1["braak"] == braak)]
#
#         if (sub2.prop.mean() > 0.05):
#             leg = True if braak not in braaks else False
#
#             fig.add_trace(go.Violin(x=sub2.braak, y=sub2.prop, name=braak, legendgroup=braak, scalegroup=mode, showlegend=leg, line_color=sub2["color"].values[0]), row=1, col=1+j)
#
#             braaks.append(braak)
#
# fig.update_traces(meanline_visible=True, width=.6, points=False)
# fig.update_layout(template="plotly_white", violinmode="group", legend=dict(orientation="h"),
#                   title="Braak sequences dominating tau propagation",
#                   xaxis1=dict(title="Fixed", showticklabels=False))
# fig.show("browser")




# 1. Trying stacked bars
ant_post["rate_order"] = ["Anterior >> Posterior" if row["idmaxrate_ant"] < row["idmaxrate_pos"] else "Posterior >> Anterior" for i, row in ant_post.iterrows()]
ant_post["fc_order"] = ["Anterior >> Posterior" if row["idmaxfc_ant"] < row["idmaxfc_pos"] else "Posterior >> Anterior" for i, row in ant_post.iterrows()]


ant_post_count = pd.DataFrame([[mode, order, measure, len(ant_post.loc[(ant_post["mode"] == mode) & (ant_post[measure + "_order"] == order)])]
                  for mode in ['fixed', 'ab_rand', 'tau_rand', 'rand'] for measure in ["rate", "fc"] for order in set(ant_post[measure + "_order"])],
                              columns=["mode", "order", "measure", "count"])


fig = px.bar(ant_post_count, x="mode", y="count", color="order", facet_col="measure", facet_col_spacing=0.15,
             color_discrete_sequence=px.colors.qualitative.Pastel1)

fig.update_layout(template="plotly_white", width=600, height=400,
                  legend=dict(title="", ), yaxis=dict(title="Dominance (%)"),
                  xaxis1=dict(title=None, tickvals=["fixed", "ab_rand", "tau_rand", "rand"], tickangle=45,
                             ticktext=["Fixed", "A\uA7B5 random", "tau random", "Random"]),
                  xaxis2=dict(title=None, tickvals=["fixed", "ab_rand", "tau_rand", "rand"], tickangle=45,
                             ticktext=["Fixed", "A\uA7B5 random", "tau random", "Random"]))

fig["layout"]["annotations"][0]["text"]="Firing Rate"
fig["layout"]["annotations"][1]["text"]="Functional<br>Connectivity"

pio.write_html(fig, main_folder + simulations_tag + "\\AnteriorPosterior_stacked.html", auto_open=True)
pio.write_image(fig, main_folder + simulations_tag + "\\AnteriorPosterior_stacked.svg")




# # 2. Trying violin plots
# ant_post["trate_pos"] = [results.loc[idx, "time"] for idx in ant_post["idmaxrate_pos"]]
# ant_post["trate_ant"] = [results.loc[idx, "time"] for idx in ant_post["idmaxrate_ant"]]
# ant_post["tfc_pos"] = [results.loc[idx, "time"] for idx in ant_post["idmaxfc_pos"]]
# ant_post["tfc_ant"] = [results.loc[idx, "time"] for idx in ant_post["idmaxfc_ant"]]
#
#
# fig = px.box(ant_post, x="mode", y="trate_pos")
# fig.show("browser")