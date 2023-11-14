Code used in the paper [Cabrera-Álvarez et al. (2023)](https://doi.org/10.1101/2023.09.24.559180) A multiscale closed-loop neurotoxicity model of Alzheimer’s disease progression explains functional connectivity alterations



- ADpg_bestCircularAD_vCC.py is the **main script** of this project. It simulates the closed-loop neurotoxicity model with the default parameters used in the paper. Here we load empirical data, define simulation parameters, run the simulation and call visualization functions. All functions used here are defined in ADpg_functions. The execution of the script saves the output into a folder "results" and returns some plots to visualize the model's behaviour. 

- ADpg_functions.py gathers a set of functions to simulate and visualize the closed-loop model. Here, we include the definition of the model. 



Figures included in the paper were built using the code in folders. R1, R3, and R4 include code to simulate the model in an HPC and gather the data that is eventually plotted. R2 includes the script used to visualize the output of executing ADpg_bestCircularAD_vCC.py as shown in the paper.

|Folder| Figures |
|------|---------|
| R1 | Figure 2 |
| R2 | Figure 3, 4 |
| R3 | Figure 5, 5-1 |
| R4 | Figure 6, 7 |



# Maths underlying the models' code -
SC derived from 20 healthy controls served as the skeleton for the BNMs implemented in TVB [(Sanz-León et al. 2015)](https://doi.org/10.1016/j.neuroimage.2015.01.002). This implementation develops a previous integrative model proposed by [Alexandersen et al. (2023)](https://doi.org/10.1098/rsif.2022.0607).


## Jansen-Rit Model
Regional signals were simulated using Jansen-Rit (JR) NMMs [(Jansen & Rit, 1995)](https://doi.org/10.1007/BF00199471), a biologically inspired model of a cortical column capable of reproducing alpha oscillations through a system of second-order coupled~differential equations (see Table 1 for a description of parameters):

$$\dot y_{0_i} = y_{3_i}$$

$$\dot y_{1_i} = y_{4_i}$$

$$\dot y_{2_i} = y_{5_i}$$

$$\dot y_{3_i} = \frac{H_e}{\tau_e} \ S[y_{1_i} - y_{2_i}] - \frac{2}{ \tau_e}  y_{3_i} - \frac{1}{\tau_e^2}  y_{0_i}$$

$$\dot y_{4_i} = \frac{H_e}{\tau_e} \ \{ I_i(t) + C_{ep} S[C_{pe} \ y_{0_i}] \} - \frac{2}{\tau_e}  y_{4_i} - \frac{1}{\tau_e^2}  y_{1_i}$$

$$\dot y_{5_i} = \frac{H_i}{\tau_i} \ C_{ip} \ S[C_{pi} y_{0_i}] - \frac{2}{\tau_i}  y_{5_i} - \frac{1}{\tau_i^2}  y_{2_i}$$

Where:
$$S\left[v\right]\ =\ \frac{2 \cdot e_{0}}{1 + e^{r(v_0-v)}}$$

$$I_i(t) = \eta_i(t) + g\sum_{j=1}^{n} w_{ji} \ S[y_{1_j}(t - d_{ji}) - y_{2_j}(t - d_{ji})]$$

$\eta_i$ is defined as a local and independent Gaussian noise \(\eta_i(t) \sim \mathcal{N}(p, \sigma)\).


## Proteinopathy
Proteinopathy dynamics are described by the heterodimer model, one of the most common hypotheses that describe the prion-like spreading of toxic proteins. This hypothesis suggests that a healthy (properly folded) protein misfolds when it interacts with a toxic version of itself (misfolded; the prion/seed) following the latter's structure as a template [(Garzón et al. 2021)](10.1016/j.jtbi.2021.110797). Therefore, this model includes healthy and toxic versions of amyloid-beta ($A\beta$ / $A\beta_t$) and tau ($TAU$ / $TAU_t$) that are produced, cleared, transformed (from healthy to toxic), and propagated in the SC with N nodes. 

$$\dot {A\beta_i} = -\rho \sum_{j=1}^{N}L_{ij} \ A\beta_j +  prod_{A\beta} \ q_i^{(ha)} - clear_{A\beta} \ A\beta_i - trans_{A\beta} \ A\beta_i \ \tilde{A\beta_i}$$

$$\dot {\tilde{A\beta_i}} = -\rho \sum_{j=1}^{N}L_{ij} \ \tilde{A\beta_j} - clear_{\tilde{A\beta}} \ \tilde{A\beta_i} + trans_{A\beta} \ A\beta_i \ \tilde{A\beta_i}$$

$$\dot {T_i} = -\rho \sum_{j=1}^{N}L_{ij} \ T_j + prod_{T} - clear_{T} \ T_i - trans_{T} \ T_i \ \tilde{T_i} - syn \ \tilde{A\beta_i} \ T_i \ \tilde{T_i}$$

$$\dot {\tilde{T_i}} = -\rho \sum_{j=1}^{N}L_{ij} \ \tilde{T_j} \ q_i^{(ha)} - clear_{\tilde{T}} \ \tilde{T_i} + trans_{T} \ T_i \ \tilde{T_i} + syn \ \tilde{A\beta_i} \ T_i \ \tilde{T_i}$$

Where:
$$L_{ij}=-w_{ij}(t) + \delta_{ij} \sum_{j=1}^{N}{w_{ij}(t)}$$





## Interactions
The effect of toxic proteins on neural activity and vice-versa is mediated by three transfer functions (damage variables). These damage variables are used to update the values of the JR parameters. 
 
$$\dot q_i^{(\tilde{A\beta})} = c_q^{(\tilde{A\beta})} \ \tilde{A\beta}_i \ (1-q_i^{(\tilde{A\beta})})$$

$$\dot q_i^{(\tilde{T})} = c_q^{(\tilde{T})} \ \tilde{T}_i \ (1-q_i^{(\tilde{T})})$$

The following interactions were considered: $A\beta$ disrupts the reuptake of glutamate rising excitation through the amplitude of the excitatory PSP ($H_e$); $A\beta$ reduced the number of inhibitory GABAergic synapses modelled through a reduction in $C_{ip}$ the average number of synapses between inhibitory neurons to pyramidal cells; hp-tau reduces the number of dendritic spines in pyramidal cells modelled through a reduction in $C_{ip}$, $C_{ep}$, and $w_{ij}$ (interregional weights).

$$\dot{H_{e_i}} = c_{exc}^{(\tilde{A\beta})} \ q_i^{(\tilde{A\beta})} \ (H_{e_{max}} - H_{e_i})$$

$$\dot{C_{ip_i}} = -c_{inh}^{(\tilde{A\beta})} \ q_i^{(\tilde{A\beta})} \ (C_{ip_i} - C_{ip_{min}}) - c_{inh}^{(\tilde{T})} \ q_i^{(\tilde{T})} \ (C_{ip_i} - C_{ip_{min}})$$

$$\dot{C_{ep_i}} = -c_{exc}^{(\tilde{T})} \ q_i^{(\tilde{T})} \ (C_{ep_i} - C_{ep_{min}})$$

$$\dot w_{ij} = -c_{sc}^{\tilde{T}} \ (q_i^{(\tilde{T})} + q_j^{(\tilde{T})}) \ (w_{ij} - w_{ij_{min}})$$

Where: 
$$w_{ij_{min}} = w_{ij_0}  (1 - T_{SC_{max}})$$


Finally, we included the effect of hyperactivity on the enhanced production of $A\beta$ and the biased prion-like propagation of TAUt to hyperactive regions directly into proteinopathy equations. Note that the level of cellular activity was evaluated through the average firing rate of the pyramidal cells in a region, using the sigmoidal transformation of the inputs to pyramidal cells (i.e., $S[y_{1_i}(t) - y_{2_i}(t)]$ described for the JR model.

$$\dot q_i^{(ha)} = c_{q}^{(ha)} \ (- q_i^{(ha)} + \Delta ha_i) \ (q_{max}^{(ha)} - q_i^{(ha)}) \ q_i^{(ha)}$$

Where
$$\Delta ha_i = ha_i(t) / ha_{i_{0}}$$

