Code use in the paper Cabrera-Álvarez et al. (2023) A multiscale closed-loop neurotoxicity model of Alzheimer’s disease progression explains functional connectivity alterations



- ADpg_bestCircularAD_vCC.py is the **main script** of this project. It simulates the closed-loop neurotoxicity model with the default parameters used in the paper. Here we load empirical data, define simulation parameters, run the simulation and call visualization functions. All functions used here are defined in ADpg_functions. The execution of the script saves the output into a folder "results" and returns some plots to visualize the model's behaviour. 

- ADpg_functions.py gathers a set of functions to simulate and visualize the closed-loop model. Here, we include the definition of the model. 



Figures included in the paper were built using the code in folders. R1, R3, and R4 include code to simulate the model in an HPC and gather the data that is eventually plotted. R2 includes the script used to visualize the output of executing ADpg_bestCircularAD_vCC.py as shown in the paper.

|Folder| Figures |
|------|---------|
| R1 | Figure 1 |
| R2 | Figure 2, 3 |
| R3 | Figure 4, 4-1 |
| R4 | Figure 6, 7 |



# Maths underlying the models' code -
SC derived from 20 healthy controls served as the skeleton for the BNMs implemented in TVB [(Sanz-León et al. 2015)](https://doi.org/10.1016/j.neuroimage.2015.01.002). This implementation develops a previous integrative model proposed by [Alexandersen et al. (2023)](https://doi.org/10.1098/rsif.2022.0607).


## Jansen-Rit Model
Regional signals were simulated using Jansen-Rit (JR) NMMs [(Jansen & Rit, 1995)](https://doi.org/10.1007/BF00199471), a biologically inspired model of a cortical column capable of reproducing alpha oscillations through a system of second-order coupled~differential equations (see Table 1 for a description of parameters):

$$\dot y_{0_i}(t) = y_{3_i}(t)$$

$$\dot y_{1_i}(t) = y_{4_i}(t)$$

$$\dot y_{2_i}(t) = y_{5_i}(t)$$

$$\dot y_{3_i}(t) = \frac{H_e}{\tau_e} \cdot S[y_{1_i}(t) - y_{2_i}(t)] - \frac{2}{ \tau_e} \cdot y_{3_i}(t) - \frac{1}{\tau_e^2} \cdot y_{0_i}(t)$$

$$\dot y_{4_i}(t) = \frac{H_e}{\tau_e} \cdot (input(t) + C_{ep} \cdot S[C_{pe} \cdot y_{0_i}(t)]) - \frac{2}{\tau_e} \cdot y_{4_i}(t) - \frac{1}{\tau_e^2} \cdot y_{1_i}(t)$$

$$\dot y_{5_i}(t) = \frac{H_i}{\tau_i} \cdot (C_{ip} \cdot S[C_{pi} \cdot y_{0_i}(t)]) - \frac{2}{\tau_i} \cdot y_{5_i}(t) - \frac{1}{\tau_i^2} \cdot y_{2_i}(t)$$

Where:
$$S\left[v\right]\ =\ \frac{2 \cdot e_{0}}{1 + e^{r(v_0-v)}}$$

$$input(t) = p_i + \eta_i(t) + g\sum_{j=1}^{n} w_{ji} \cdot S[y_{1_j}(t - d_{ji}) - y_{2_j}(t - d_{ji})]$$


## Proteinopathy
Proteinopathy dynamics are described by the heterodimer model, one of the most common hypotheses that describe the prion-like spreading of toxic proteins. This hypothesis suggests that a healthy (properly folded) protein misfolds when it interacts with a toxic version of itself (misfolded; the prion/seed) following the latter's structure as a template [(Garzón et al. 2021)](10.1016/j.jtbi.2021.110797). Therefore, this model includes healthy and toxic versions of amyloid-beta ($A\beta$ / $A\beta_t$) and tau ($TAU$ / $TAU_t$) that are produced, cleared, transformed (from healthy to toxic), and propagated in the SC with N nodes. 

$$\dot {A\beta_i} = -\rho \sum_{j=1}^{N}L_{ij} \cdot A\beta_j +  prod_{A\beta} \cdot q_i^{ha} - clear_{A\beta} \cdot A\beta_i - trans_{A\beta} \cdot A\beta_i \cdot A\beta t_i$$

$$\dot {A\beta t_i} = -\rho \sum_{j=1}^{N}L_{ij} \cdot A\beta t_j - clear_{A\beta t} \cdot A\beta t_i + trans_{A\beta} \cdot A\beta_i \cdot A\beta t_i$$

$$\dot {TAU_i} = -\rho \sum_{j=1}^{N}L_{ij} \cdot TAU_j + prod_{T} - clear_{T} \cdot TAU_i - trans_{T} \cdot TAU_i \cdot TAUt_i - syn_{T} \cdot A\beta t \cdot TAU \cdot TAUt $$

$$\dot {TAUt_{i}} = -\rho \sum_{j=1}^{N}L_{ij} \cdot TAUt_{j} \cdot q_i^{ha} - clear_{\tau_{t}} \cdot TAU_{t_i} + trans_{T} \cdot TAU_i \cdot TAUt_{i} + syn_{T} \cdot A\beta t \cdot TAU \cdot TAUt$$

Where:
$$L_{ij}=-w_{ij}(t) + \delta_{ij} \sum_{j=1}^{N}{w_{ij}(t)}$$





## Interactions
The effect of toxic proteins over neural activity and viceversa is mediated by three transfer functions (damage variables). These damage variables are used to update the values of the JR parameters and to influence the dynamics of proteinopathy (note that hyperactivity damage - $q_i^{(ha)}$ was directly introduced in the proteinopathy equations). 
 
 $$\dot q_i^{(A\beta)} = A\beta t_{damrate} \cdot A\beta t_i (1-q_i^{(A\beta)})$$

$$\dot q_i^{(T)} = TAUt_{damrate} \cdot TAUt_i (1-q_i^{(T)})$$

$$\dot q_i^{(ha)} = ha_{damrate} \cdot \Delta ha_i$$

Where
$$\Delta ha_i = ha_i(t) / ha_{i_{0}}$$


 The following interactions were considered: $A\beta$ disrupts the reuptake of glutamate rising excitation through the amplitude of the excitatory PSP ($H_e$); $A\beta$ reduced the number of inhibitory GABAergic synapses modeled through a reduction in $C_{ip}$ the average number of synapses between inhibitory neurons to pyramidal cells; hp-tau reduces the number of dendritic spines in pyramidal cells modeled through a reduction in $C_{ip}$, $C_{ep}$, and $w_{ij}$ (interregional weights).

$$\dot{H_{e_i}} = c_{A\beta_{exc}} q_i^{(A\beta)} (H_{e_{max}} - H_{e_i})$$

$$\dot{C_{ip_i}} = -c_{A\beta_{inh}} q_i^{(A\beta)}(C_{ip_i} - C_{ip_{min}}) - c_{T_{inh}} q_i^{(T)} (C_{ip_i} - C_{ip_{min}})$$

$$\dot{C_{ep_i}} = c_{T_{exc}}  q_i^{(T)}  (C_{ep_i} - C_{ep_{min}})$$


$$\dot w_{ij} = -c_{T_{SC}} (q_i^{(T)} + q_j^{(T)}) (w_{ij} - w_{ij_{min}} )$$

$$w_{ij_{min}} = w_{ij_0}  (1 - T_{SC_{max}})$$


Finally, we included the effect of hyperactivity on the enhanced production of $A\beta$ and on the biased prion-like propagation of TAUt to hyperactive regions directly into proteinopathy equations.
