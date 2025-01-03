# EZBattery
An easy-to-use redox flow battery simulation software  
## Instructions
EZBattery is an easy-to-use flow battery simulation software for predicting the concentrations, potentials, and cycling performance of redox flow batteries (RFBs) and their variants. Its generalizability and accuracy have been verified for inorganic (all vanadium), organic (DHPS-Ferricyanide), hybrid (Zinc-Iodine), and redox-targeting (Fe-S) RFBs by comparing software simulation results with laboratory-scale experiments. This software demonstrates excellent computational efficiency of less than 0.1 s per charge-discharge cycle. In addition to its generalizability, high accuracy, and high efficiency, the software also supports various physical and operation setups, including a switch option between 0-dimension and 2-dimension scheme, a support for both unit cell and long duration energy storage system simulations, a support for constant current and time-varying power operations, and flexible for using different physicochemical parameters for electrode, electrolyte, current collector, pump, and operational conditions. The software is also user-friendly by using standard input and output formats, automated model calibration and validation, and flexible control on the start and stop operating time and state of charge. The work is supported by the Pacific Northwest National Laboratory’s Energy Storage Materials Initiative and US DOE Office of Electricity’s Rapid Operational Validation Initiative.

The details about the models are available in the journal articles: 

Yunxiang Chen, Zhijie Xu, Chao Wang, Jie Bao, Brian Koeppel, Litao Yan, Peiyuan Gao, Wei Wang, (2021), Analytical modeling for redox flow battery design, Journal of Power Sources, 482 (15) 228817 (DOI https://doi.org/10.1016/j.jpowsour.2020.228817)      

Yunxiang Chen, Jie Bao, Zhijie Xu, Peiyuan Gao, Litao Yan, Soowhan Kim, Wei Wang, (2021), A two-dimensional analytical unit cell model for redox flow battery evaluation and optimization, Journal of Power Sources, 506 (15) 230192 (DOI https://doi.org/10.1016/j.jpowsour.2021.230192)    

Yunxiang Chen, Jie Bao, Zhijie Xu, Peiyuan Gao, Litao Yan, Soowhan Kim, Wei Wang (2023), A hybrid analytical and numerical model for cross-over and performance decay in a unit cell vanadium redox flow battery, Journal of Power Sources, 578 (15) 233210 (DOI https://doi.org/10.1016/j.jpowsour.2023.233210)  
   
## Installation
The following libraries are needed.  
* Python  
  Version 3.9 or newer are recommended.  
  It is highly recommended to install the python and the dependent packages through Conda. The user can find the anaconda package for various operation systems on https://www.anaconda.com/products/distribution#Downloads
* numpy (1.24.3)
* scipy (1.14.0)
* ismember (1.0.5)
* openpyxl (3.0.10)
  
Using different version of numpy and scipy as listed above may cause slightly different simualtion results.  

## Run the code
All the codes for the models are in folder EZBattery. Five Jupyter Notebooks are provided to demonstrate how to used the code. All the necessary data for the demos are in folder Data.
* Demo_Cell_Usage.ipynb : a demo to show the minimum requirement to run the model, with all vanadium flow battery as an example.
* Demo_Calibration_Vanadium.ipynb : simulate all vanadium redox flow battery.
* Demo_Calibration_Organic.ipynb : simulate organic elelctrolyte redox flow battery.
* Demo_Zn_system.ipynb : simulate Zn/I hybrid redox flow battery, and up-scale from cell to long duration energy storage system.
* Demo_Calibration_RedoxTargeting.ipynb : simulate redox targeting flow battery.

The users can follow the descrption in each Jupyter Notebook to run the model step by step.

## Cite as

@misc{Chen2024,
address = {Richland, WA, USA},
author = {Chen, Yunxiang and Bao, Jie and Gao, Peiyuan and Fu, Yucheng and Zeng, Chao and Xu, Zhijie and Kim, Soowhan and Yan, Litao and Wang, Wei and Jiang, Qixuan and Liu, Alvin and Louie, Tiffany and Yuan, Grace},
title = {{EZBattery: an easy-to-use flow battery simulation software}},
url = {https://github.com/pnnl/EZBattery },
year = {2024}
}

## Developers
Yunxiang Chen, Jie Bao, Peiyuan Gao, Yucheng Fu, Alasdair Crawford, Zhijie Xu, Chao Zeng, Soowhan Kim, Litao Yan, Wei Wang, Qixuan Jiang*, Alvin Liu*, Tiffany Louie*, Grace Yuan*  
Pacific Northwest National Laboratory, Richland, WA, USA  
 \* : Intern students

