Code use in the paper Cabrera-Álvarez et al. (2023) A multiscale closed-loop neurotoxicity model of Alzheimer’s disease progression explains functional connectivity alterations



- ADpg_bestCircularAD_vCC.py is the **main script** of this project. It simulates the closed loop neurotoxicity model with the default parameters used in the paper. Here we load empirical data, define simulation parameters, run the simulation and call visualization functions. All functions used here are defined in ADpg_functions. The execution of the script saves the output into a folder "results" and returns some plots to visualize the behaviour of the model. 

- ADpg_functions.py gathers a set of functions used to simulate and visualize the closed-loop model. Here, we include the definition of the model. 



Figures included in the paper were built using the code in folders. R1, R3, and R4 include code to simulate the model in a HPC, and gather the data that is eventually plotted. R2 includes the script used to visualize the output of executing ADpg_bestCircularAD_vCC.py as shown in the paper.

R1 - Figure 1; R2 - Figures 2 and 3; R3 - Figures 4, 4-1 and 5; R4 - Figures 6 and 7






