#%%######################### Descriptions #####################################
# This code includes functions for 3 flow batteries,including all vanadium RFB#
# ,DHPS-Ferricyanide RFB, and Zinc-Iodine RFB. The code is developed by Dr.   #
# Yunxiang Chen @PNNL. Please contact yunxiang.chen@pnnl.gov for questions.   #
# Potential users are permitted to use for non-commercial purposes without    #
# modifying any part of the code. Please contact the author to obtain written #
# permission for commercial or modifying usage.                               # 

# Copyright (c) 2024 Yunxiang Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%% Required package for redox flow batteries.
import time, warnings, copy
import numpy as np                                                             # Version 1.24.3.
import numpy.matlib
import pandas as pd                                                            # Using 2.0.3.
from .EZBatteryUtilities import CrossOverRateUNIROVI, UpdateCVUNIROVI
from .mathm import linspace, conc, calcEnergy                          

#%% 
# RFB: a unified code for predicting voltage of three redox flow batteries,   %
# including all vanadium RFB, DHPS-Ferricyanide RFB, and Zinc-Iodine RFB. The #
# input of the code, PP, is determined from a StanDard Paramter File (SDPF),  #
# designed for each battery. Important output are summarized in Re. Please use#
# the following command to access information in Re: a) extracting all keys in#
# Re: Re.keys(); b) Extracting keys in Potenteials: Re['Potentials'].keys();  #
# c) Converting dataframe to array format: Re['Potentials'].values            #
# The code is developed by Yunxiang Chen, yunxiang.chen@pnnl.gov              #

def RFB(PP0):   
    # Mainbody.
    timestart=time.time()                                                      # Start time of the code.
    # Note: copy.copy() cannot avoid parameter interference.
    PP = copy.deepcopy(PP0)                                                    # Avoid parameter interactions.

    # Calculate and extract parameters from input parameters for later usage.
    Chemical = PP.Chemical
    Calculated = PP.Calculated
    Display = PP.Display.lower() == 'yes'                                      # If output information on screen (integer).
    t_start = PP.StartTime                                                     # Starting time.
    t_end = PP.EndTime                                                         # End of simulation time.
    cmin = PP.MininumConcentration                                             # Minimum concentration.

    # Checking if batter is Zinc or not.
    iszinc = 1 if 'Zinc' in Chemical['Name'] or\
        'Iodide' in Chemical['Name'] else 0
        
    # Identify active and non-active species.
    n = np.zeros(3,dtype=int)
    ixa = Chemical['Active']
    ixn = Chemical['NoneActive']

    # For active species (4 species).
    active_species = [Chemical["Name"][i] for i in ixa]
    C0C = Chemical['InitialConcentrationInCathode']                            # All initial concentrations in Cathode.
    C0A = Chemical['InitialConcentrationInAnode']
    C0C = np.array(C0C,dtype=float)
    C0A = np.array(C0A,dtype=float)
    CI0 = conc([C0C[i] for i in ixa],[C0A[i] for i in ixa])              # Concentrations 4(C/A),5(C/A),3(C/A),2(C/A).
    n[0] = len(CI0) // 2                                                       # Number of active species.

    # For non-active species (length(species) - 4).
    nonactive_species = [Chemical["Name"][i] for i in ixn]                     # For V: Water_P, Water_N, H_P, H_N;
    CS0 = conc([C0C[i] for i in ixn],[C0A[i] for i in ixn])              # For DHPS: Water_P, Water_N, OH_P, OH_N.
    n[1] = len(CS0) // 2                                                       # Number of non-active species.

    # For membrane species.
    membrane_species = []
    CM0 = []
    if PP.MembraneIonName not in Chemical['Name']:
        membrane_species.append(PP.MembraneIonName)
        CM0.extend([PP.MembraneIonInitialConcentrationInCathode, \
                    PP.MembraneIonInitialConcentrationInAnode])                # For membrane carrying ion;
    if PP.MembraneSolventName not in Chemical['Name']:
        membrane_species.append(PP.MembraneSolventName)
        CM0.extend([PP.MembraneSolventInitialConcentrationInCathode, \
                    PP.MembraneSolventInitialConcentrationInAnode])            # For water in negative.
    CM0 = np.array(CM0,dtype=float)
    n[2] = len(CM0) // 2                                                       # Number of membrane related species.

    # Parameters.
    VCutOffCharge = PP.CutOffVoltageCharge
    VCutOffDischarge = PP.CutOffVoltageDischarge
    if PP.BatteryMode.lower() != 'current':                                    # Using different cut off voltage.
        VCutOffCharge = PP.SystemCutOffVoltageCharge
        VCutOffDischarge = PP.SystemCutOffVoltageDischarge
        
    #Lh = PP.ElectrodeHeight;                                                  # Electrode height (m)
    dt = float(PP.TimeStep)
    t_wait = PP.ChargeRestTime
    isign_2D = Calculated['TwoD']
    I = PP.Current                                                             # Current (A).
    F = PP.Constants['Faraday']                                                # Faraday constant;
    VE = Calculated['ElectrodeVolume'];                                        # Electrode volume (m^3).
    NE = Chemical['ElectronNumber'][1]
    k = PP.Heterogeneity

    # Output information control
    formatSpec1 = 'Running a multi-cycle unit cell model.'
    formatSpec2 = 'Cycle %d: charge %.3f h, discharge %.3f h, CE: %.3f, Cumulative CE %.3f (%.1f%%).'
    formatSpec3 = 'Active species: {}.'
    #formatSpec3b = 'Active species: {0}, {1}, {2}, {3}.'
    formatSpec4 = 'Non active species: {}.'
    #formatSpec4b = 'Non active species: {0}, {1}.'
    formatSpec5 = 'Membrane species: {}.\n'
    #formatSpec5b = 'Membrane species: {0}.\n'
    formatSpec6 = 'Computation done, elapsed time %g seconds, average time per cycle %g seconds.\n';
    formatSpec7 = 'Computation done, elapsed time %g seconds.\n';
    msg = 'Negative concentration found, adjust time step or minimum concentration.';
    if Display: print(formatSpec1)
    if len(active_species)>0 and Display: print(formatSpec3.format(active_species))
    if len(nonactive_species)>0 and Display: print(formatSpec4.format(nonactive_species))
    if len(membrane_species)>0 and Display: print(formatSpec5.format(membrane_species))
    #print(formatSpec3.format(active_species[0],active_species[1],active_species[2],active_species[3]))
    #print(formatSpec4b.format(nonactive_species[0],nonactive_species[1]))
    #print(formatSpec5b.format(membrane_species[0]))

    # UCMCO model mesh/time/initializations.
    # Grid/time
    nc0 = PP.CycleNumber
    ni = PP.RefineTimeTimes                                                    # Refine times, minimum 1.
    nps = PP.RefineTimePoints                                                  # Number of points to refine, minimum 1.
    Nt = int(np.ceil((t_end - t_start) / dt)) + 1                              # Number of times (Nt>=2).
    if nps + 1 >= Nt: nps = max(Nt - 2,1)                                      # Refine points number should less than total time number.
    ti = linspace(t_start, t_end, Nt)                                    # Time seris.
    Ntt = Nt + nc0 * (2*nps * (ni - 1) + nps*ni) + nps*(ni-1) + 1              # Estimated maximum time step.

    # The Cka, Ckb, Ckc, Ckd are reordered as 4/5/3/2 in Calculated parameters.#
    # However, the code below is designed based on the order V2,3,4,5. We need #
    # to reorder the 4/5/3/2 to 2/3/4/5 to avoid changing too many codes.      # 
    order = np.array([3, 2, 0, 1])
    ixa = np.array(Chemical['Active'])[order]
    Ki = np.array(Chemical['PartitionCoefficient'])[ixa]
    Dm = np.array(Chemical['DiffusivityInMembrane'])[ixa]
    Lm = PP.MembraneThickness
    nv = 4

    # Initialization for active, non-active, and membrane species. CI         #
    # represnts: V2345 in tank (1-4) and V2345 in electrode outlet (5-8).     #               
    CI = np.full((int(n[0]) * 2, Ntt), np.nan)                                 # Vanadium concentrations at outlet for tank and electrode.
    CI[0:4, 0] = CI0[[7, 5, 0, 2]]                                             # Tank concentration at outlet for V2,3,4,5.
    CI[4:8, 0] = CI[0:4, 0] + Calculated['Ckd'][order] * isign_2D              # Electrode concentrations at outlet for V4,5,3,2.
    #eps0 = np.array([PP.CathodeElectrodePorosity,PP.AnodeElectrodePorosity])   # Initial porosity.
    eps = np.matlib.repmat(Calculated['Porosity'],Ntt,1)                       # Time varying porosity for cathode and anode.
    As = np.matlib.repmat(Calculated['SpecificArea'],Ntt,1)                    # Time varying specific area for cathode and anode.
    eps_neg0 = Calculated['Porosity'][1]                                       # Inital anode porosity.

    # V: CHW represents HP, WP, HN, WN for tank (1-4) and for electrode (5-8) #
    # DHPS:CHW represents OHP, WP, OHN, WN for tank (1-4) and for electrode (5-8).#
    CHW = np.full((int(n[1]) * 4, Ntt), np.nan)
    CHW[:, 0] = np.tile(CS0[[2, 0, 3, 1]], 2)                                  # Covert the order from WP, WN, HP, HN to HP WP HN WN.
    Nz = 1                                                                     # Nubmer of grid points along cell height direction.
    Vcell = np.full((Nz, Ntt), np.nan)                                         # Cell voltage and potential components at each time and electrode outlet.
    SOC_t = np.full(Ntt, np.nan)                                               # Calculated state of charge.
    ncc = np.full(Ntt, np.nan)                                                 # Cycle number.
    C = np.full((Nz, Ntt, 8 + 8), np.nan)                                      # Concentrations for V2/5 at cell center/wall (4+4), and proton/water (2+1).
    V = np.full((Nz, Ntt, 13), np.nan)                                         # Potentials for cell voltage (1) and components (6+6).
    S = np.full((300, 1 + 3 + 2 + 2 + 6), np.nan)                              # Statistics for each cycle.

    # Initial parameters controlling voltage cycles.
    t_charge = np.inf                                                          # The duration of charge for each cycle (s).
    t_cycle_old = t_start                                                      # The end time of previous discharge cycle (s).
    t_cycle = t_cycle_old
    sc = 3600                                                                  # Convert seconds to hour.

    # Calculating flux ratios if FluxCalculation is enabled.
    nm = PP.MembranePoints                                                     # Points along membrane thickness direction (1, no effect on voltage).
    FX = np.full((Ntt, nm, int(n[0]) * 3), np.nan)
    CM = np.full((Ntt, nm, int(n[0])), np.nan)
    CMA = np.full((Ntt, int(n[0])), np.nan)
    m1, r1, uct1, _, Ckb, _, co1, st1 = CrossOverRateUNIROVI(1, PP)            # Note: uct1 is always positive.
    m2, r2, uct2, _, _, _, co2, st2 = CrossOverRateUNIROVI(0, PP)
    m3, r3, uct3, _, _, _, co3, st3 = CrossOverRateUNIROVI(-1, PP)
    CO = [co1, co2, co3]

    # This is special case for Zinc battery.
    if iszinc: 
        m1[[0,1,4,5],:] = 0
        m2[[0,1,4,5],:] = 0
        m3[[0,1,4,5],:] = 0

    # Initialization the calculation.
    Status = PP.InitialStatus                                                  # Initial status.
    dt0 = PP.InitialTimeStep                                                   # Initial time step.
    if iszinc:
        eps[1,1] = eps_neg0 - I/F/NE/VE/CI[4,0]*dt0*Status*iszinc              # Update anode porosity.
        PP.AnodeElectrodePorosity = eps[0,1]                                   # Update parameters.
        As[1,:] = PP.Calculated['SpecificArea']                                # Save the updated specific area.

    # Computing new concentrations and voltages for step 2 using 0D model.
    TS = [t_cycle,t_start,t_start + dt0]
    CT0 = np.concatenate([CI[:, 0], [t_start]])
    if Status == 1:
        CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
            (Status, TS, CI[:, 0], \
              CHW[:, 0], PP, m1, r1, uct1, st1, CT0)
    elif Status == 0:
        CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
            (Status, TS, CI[:, 0], \
              CHW[:, 0], PP, m2, r2, uct2, st2, CT0)
    else:
        CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
            (Status, TS, CI[:, 0], \
              CHW[:, 0], PP, m3, r3, uct3, st3, CT0)      

    # Updating results for the first step.
    CI[:, 1] = CIN
    Vcell[:, 1] = VN[:, 0]
    SOC_t[1] = SOCN
    ncc[1] = 1
    V[:, 1, :] = VN
    C[:, 1, :] = CN
    CHW[:, 1] = CHWN
    nt0 = np.hstack([0, linspace(dt0 / dt, 1, nps * ni + 1)])            # Non-dimensional time data.
    nt1 = linspace(0, 1, nps * ni + 1)
    ti = np.hstack([ti[0] + (ti[nps] - ti[0]) * nt0, ti[nps + 1 :]])           # Add multiple points at the beginning.
    Nt = Nt  - (nps + 1) + nps*ni + 2  # Note: Nt>=Nt+1>=3.
    VNO = VN[0,0]; vre = 0
    nc = 1                                                                     # The index of cycle.

    # Used to control cases with unresonable parameter inputs.
    if Status == 1 and VNO>VCutOffCharge: nc = 1e6                             # Set large nc to disable the the loop below.
    if Status == -1 and VNO<VCutOffDischarge: nc = 1e6                         # Set large nc to disable the the loop below.

    # Updating cycling data.
    i = 2                                                                      # The third index.
    BattMode = PP.BatteryMode.lower() == 'power'
    #while i< Nt-nps-1 and i < Ntt and vre<=0.5 and nc <= PP.CycleNumber:
    while i< Nt and i < Ntt and vre<=10 and nc <= PP.CycleNumber:
        ncc[i] = nc
        if (BattMode and np.all(CIN>cmin)): Status = PP.InitialStatus          # Fix battery status if in power mode.
        if (BattMode and np.any(CIN<=cmin)): break                             # Early exit if very small concentrations found.                      
        
        if Status == 1:
            # Updating parameters.
            TS = [t_cycle, ti[i-1], ti[i]]                                     # Update time information: cycle start time, old, and new time.
            if iszinc:
                eps[i,1] = eps[i-1,1] - \
                    I/F/NE/VE/CI[4,i-1]*(TS[2]-TS[1])*iszinc                   # Update anode porosity.
                PP.AnodeElectrodePorosity = eps[i,1]                           # Update parameters.
            if iszinc: m1, r1, uct1, _, Ckb, _, co1, st1 = \
                CrossOverRateUNIROVI(1, PP)                                    # Note: uct1 is always positive.
            if iszinc: m1[[0,1,4,5],:] = 0
            
            # Updating concentration and voltage.
            CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
                (Status, TS, CI[:, i-1], CHW[:, i-1], PP, m1, r1, uct1, st1, CT0)

            # Checkign if all concentrations are valid.
            if np.any(CIN < 0): warnings.warn(msg); break
            if iszinc: 
                As[i,:] = PP.Calculated['SpecificArea']                        # Save the updated specific area.
                eps[i,:] = PP.Calculated['Porosity']                           # Save the updated specific area.
            CI[:, i] = CIN; V[:, i, :] = VN; C[:, i, :] = CN
            SOC_t[i] = SOCN; 
            Vcell[:, i] = VN[:, 0]  ; 
            CHW[:, i] = CHWN
            
            # Identifying final points.
            if not BattMode and Vcell[-1, i] > VCutOffCharge:
                tio = ti[i-1]                                                  # Old time.
                tin = ti[i]                                                    # New time.
                tif = (tio + tin) / 2                                          # Middle time.
                TS = [t_cycle, tio, tif, tin]
                CIO = CI[:, i-1]
                CHWO = CHW[:, i-1]
                if iszinc: 
                    eps[i,1] = eps[i-1,1] - \
                        I/F/NE/VE/CIO[4]*(TS[2]-TS[1])*Status*0*iszinc         # Update anode porosity.
                    PP.AnodeElectrodePorosity = eps[i,1]                       # Update parameters.
                    m1, r1, uct1, _, Ckb, _, co1, st1 = \
                        CrossOverRateUNIROVI(1, PP)                            # Note: uct1 is always positive.
                if iszinc: m1[[0,1,4,5],:] = 0
                
                CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
                    (Status, TS, CIO, CHWO, PP, m1, r1, uct1, st1, CT0)
                if iszinc: 
                    As[i,:] = PP.Calculated['SpecificArea']                    # Save the updated specific area.
                    eps[i,:] = PP.Calculated['Porosity']                       # Save the updated specific area.
                count = 1
                err = abs(VN[-1, 0] - VCutOffCharge)

                while err > PP.CutOffVoltageTolerance and \
                    count <= PP.DischargeRefineTimeNumber - 1 + 2:
                    count = count + 1
                    if VN[-1, 0] < VCutOffCharge:
                        tio = TS[2]; tin = TS[3]; tif = (tio + tin) / 2
                        CIO = CIN ; CHWO = CHWN
                    else:
                        tio = TS[1]; tin = TS[2]; tif = (tio + tin) / 2
                    TS = [t_cycle, tio, tif, tin]
                    if iszinc:
                        eps[i,1] = eps[i-1,1] - \
                            I/F/NE/VE/CIO[4]*(TS[2]-TS[1])*Status*iszinc       # Update anode porosity.
                        PP.AnodeElectrodePorosity = eps[i,1]                   # Update parameters.
                        m1, r1, uct1, _, Ckb, _, co1, st1 = \
                            CrossOverRateUNIROVI(1, PP)                        # Note: uct1 is always positive.
                    if iszinc: m1[[0,1,4,5],:] = 0
                  
                    CIN, VN, CN, SOCN, CHWN, _, _  = \
                        UpdateCVUNIROVI(Status, TS, CIO, CHWO, PP, m1, r1, uct1, st1, CT0)
                    if iszinc:
                        As[i,:] = PP.Calculated['SpecificArea']                # Save the updated specific area.
                        eps[i,:] = PP.Calculated['Porosity']                   # Save the updated specific area.
                    err = abs(VN[-1, 0] - VCutOffCharge)
                    
                ti[i] = tif
                CI[:, i] = CIN
                V[:, i, :] = VN
                C[:, i, :] = CN
                SOC_t[i] = SOCN
                Vcell[:, i] = VN[:, 0]
                CHW[:, i] = CHWN
                t_charge = ti[i] - t_cycle                                     # Recording the time of end charging.
                CT0 = np.concatenate([CIN, [ti[i]]])
                S[nc-1, [0, 1, 4, 6, 8, 9]] = [nc, t_charge, Vcell[-1, i]\
                                  - VCutOffCharge, count, t_cycle, ti[i]]
                Status = 0
                if PP.BatteryMode.lower() in 'power': Status = PP.InitialStatus# Fix battery status if in power mode.
                
                # Early termination if index exceeds the maximum size.
                if i + 1>=Nt: i = i + 1; break                                 

                # Add points if necessary.
                if ti[i] + t_wait < ti[i+1]:
                    ti = np.concatenate([ti[:i], ti[i] + nt1 * t_wait, ti[i+1:]])
                    Nt = Nt-1 + nps*ni + 1
                elif ti[i] + t_wait == ti[i+1]:
                    ti = np.concatenate([ti[:i], ti[i] + nt1 * t_wait, ti[i+2:]])
                    Nt = Nt-2 + nps*ni + 1
                else:
                    idk = np.where(ti[i+1:] < ti[i] + t_wait)[0][-1] + 1
                    ti = np.concatenate([ti[:i+idk], ti[i+idk] + \
                                          (ti[i] + t_wait - ti[i+idk]) * nt1, ti[i+idk+1:]]) 
                    Nt = Nt - 1 + nps * ni + 1
            if PP.FluxCalculation.lower() == 'yes':
                cav1_e = (CI[4:8, i] - Ckb[order]) * eps**k
                cav1_m = cav1_e * Ki
                flux1 = cav1_m * Dm / Lm
        
                for j in range(nv):
                    FX[i, :, (j-1)*3:(j*3)] = flux1[j] * co1['Componets'][:, (j-1)*3:(j*3)]
                    CM[i, :, j] = co1['Concentrations'][:, j] * cav1_m[j]
            
                CMA[i, :] = cav1_m * co1['ConcentrationCoeff']
        elif Status == 0:
            if ti[i] < t_charge + t_wait + t_cycle:
                if iszinc:
                    As[i,:] = PP.Calculated['SpecificArea']                    # Save the updated specific area.
                    eps[i,:] = PP.Calculated['Porosity']                       # Save the updated specific area.
                    m2, r2, uct2, _, _, _, co2, st2 = CrossOverRateUNIROVI(0, PP)
                if iszinc: m2[[0,1,4,5],:] = 0
                #TS = [t_cycle, ti[i - 1], ti[i]]
                CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(0, TS, CI[:, i - 1], CHW[:, i - 1], PP, m2, r2, uct2, st2, CT0)
                CI[:, i] = CIN
                V[:, i, :] = VN
                C[:, i, :] = CN
                SOC_t[i] = SOCN
                Vcell[:, i] = VN[:, 0]
                CHW[:, i] = CHWN
            else:
                #print(ti[i] - (t_charge + t_wait + t_cycle))                  # Will always be 0.
                TS = [t_cycle, ti[i - 1], ti[i]]
                if iszinc: 
                    eps[i,1] = eps[i-1,1] - \
                        I/F/NE/VE/CI[4,i-1]*(TS[2]-TS[1])*0*iszinc             # Update anode porosity.
                    PP.AnodeElectrodePorosity = eps[i,1]                       # Update parameters.
                    m2, r2, uct2, _, _, _, co2, st2 = CrossOverRateUNIROVI(0, PP)
                if iszinc: m2[[0,1,4,5],:] = 0
                CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(0, TS, CI[:, i - 1], CHW[:, i - 1], PP, m2, r2, uct2, st2, CT0)
                if iszinc:
                    As[i,:] = PP.Calculated['SpecificArea']                    # Save the updated specific area.
                    eps[i,:] = PP.Calculated['Porosity']                       # Save the updated specific area.
                CI[:, i] = CIN
                V[:, i, :] = VN
                C[:, i, :] = CN
                SOC_t[i] = SOCN
                Vcell[:, i] = VN[:, 0]
                CHW[:, i] = CHWN
                CT0 = np.concatenate((CIN, [ti[i]]))
                S[nc-1, [10, 11]] = [S[nc-1, 9], ti[i]]
                if i + nps >=Nt: i = i + 1; break
                ti = np.concatenate([ti[:i], ti[i] + (ti[i+nps] - ti[i]) * nt1, ti[i+nps+1:]])
                Nt = Nt - (nps+1) + ni*nps + 1
                Status = -1
                if PP.BatteryMode.lower() in 'power': Status = PP.InitialStatus# Fix battery status if in power mode.

            # Updating fluxes
            if PP.FluxCalculation.lower() == 'yes':
                cav2_e = (CI[4:8, i] - Ckb[order]) @ eps**k
                cav2_m = cav2_e * Ki
                flux2 = cav2_m * Dm / Lm
                for j in range(nv):
                    FX[i, :, (j - 1) * 3:(j - 1) * 3 + 3] = \
                        flux2[j] * co2.Componets[:, (j - 1) * 3:(j - 1) * 3 + 3]
                    CM[i, :, j] = co2.Concentrations[:, j] * cav2_m[j]
                CMA[i, :] = co2['ConcentrationCoeff'] * cav2_m
        else:
            tdc = t_cycle + t_charge + t_wait
            TS = [tdc, ti[i - 1], ti[i]]
            if iszinc: 
                eps[i,1] = eps[i-1,1] - \
                    I/F/NE/VE/CI[4,i-1]*(TS[2]-TS[1])*Status*iszinc            # Update anode porosity.
                PP.AnodeElectrodePorosity = eps[i,1]                           # Update parameters.
                m3, r3, uct3, _, _, _, co3, st3 = CrossOverRateUNIROVI(-1, PP)
            if iszinc: m3[[0,1,4,5],:] = 0
            CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(Status, TS, CI[:, i - 1], CHW[:, i - 1], PP, m3, r3, uct3, st3, CT0) 
            if iszinc:
                As[i,:] = PP.Calculated['SpecificArea']                        # Save the updated specific area.
                eps[i,:] = PP.Calculated['Porosity']                           # Save the updated specific area.
            CI[:, i] = CIN
            V[:, i, :] = VN
            C[:, i, :] = CN
            SOC_t[i] = SOCN
            Vcell[:, i] = VN[:, 0]
            CHW[:, i] = CHWN
            
            if not BattMode and Vcell[-1, i] < VCutOffDischarge:
                tio = ti[i - 1]
                tin = ti[i]
                tif = (tio + tin) / 2
                CIO = CI[:, i - 1]
                TS = [tdc, tio, tif]
                CHWO = CHW[:, i - 1]
                if iszinc: 
                    eps[i,1] = eps[i-1,1] - I/F/NE/VE/CIO[4]*(TS[2]-TS[1])*Status*iszinc  # Update anode porosity.
                    PP.AnodeElectrodePorosity = eps[i,1]                       # Update parameters.
                    m3, r3, uct3, _, _, _, co3, st3 = CrossOverRateUNIROVI(-1, PP)
                if iszinc: m3[[0,1,4,5],:] = 0
                CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(Status, TS, CIO, CHWO, PP, m3, r3, uct3, st3, CT0)
                if iszinc:
                    As[i,:] = PP.Calculated['SpecificArea']                    # Save the updated specific area.
                    eps[i,:] = PP.Calculated['Porosity']                       # Save the updated specific area.
             
                count = 1
                err = abs(VN[-1, 0] - VCutOffDischarge)
                while err > PP.CutOffVoltageTolerance and count <= PP.DischargeRefineTimeNumber - 1 + 2:
                    if VN[-1, 0] > VCutOffDischarge:
                        tio = tif
                        tif = (tio + tin) / 2
                        CIO = CIN
                        CHWO = CHWN
                    else:
                        tin = tif
                        tif = (tio + tin) / 2

                    TS = [tdc, tio, tif]
                    if iszinc:
                        eps[i,1] = eps[i-1,1] - I/F/NE/VE/CIO[4]*(TS[2]-TS[1])*Status*iszinc  # Update anode porosity.
                        PP.AnodeElectrodePorosity = eps[i,1]                   # Update parameters.
                        m3, r3, uct3, _, _, _, co3, st3 = CrossOverRateUNIROVI(-1, PP)
                    if iszinc: m3[[0,1,4,5],:] = 0
                    CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(Status, TS, CIO, CHWO, PP, m3, r3, uct3, st3, CT0)
                    if iszinc:
                        As[i,:] = PP.Calculated['SpecificArea']                # Save the updated specific area.
                        eps[i,:] = PP.Calculated['Porosity']                   # Save the updated specific area.
                    err = abs(VN[-1, 0] - VCutOffDischarge)
                    count = count + 1

                ti[i] = tif                                                    # Update time
                CI[:, i] = CIN
                V[:, i, :] = VN
                C[:, i, :] = CN
                CHW[:,i] = CHWN
                SOC_t[i] = SOCN
                Vcell[:, i] = VN[:, 0]
                CT0 = np.concatenate((CIN, [ti[i]]))
                S[nc-1, [2, 5, 12, 13]] = [tif - tdc, Vcell[-1, i] - VCutOffDischarge, tdc, tif]
                S[nc-1, 3] = S[nc-1, 2] / S[nc-1, 1]
                S[nc-1, 7] = count

                # Update information for the next cycle.
                if Display and PP.InitialStatus == 1:
                    print(formatSpec2 %(S[nc - 1, 0],S[nc - 1, 1]/sc,
                    S[nc - 1, 2]/sc,S[nc - 1, 2]/S[nc-1, 1],
                    S[nc - 1, 2]/S[0, 1],100 * ti[i]/ti[-1]))
                    
                t_cycle_old = ti[i]
                t_cycle = t_cycle_old
                
                if i + nps >=Nt: i = i + 1; break
                ti = np.concatenate([ti[:i], t_cycle + (ti[i + nps] - t_cycle) * nt1, ti[i + nps + 1:]])
                Nt = Nt - (nps + 1) + ni*nps + 1
                nc = nc + 1
                Status = 1
                if PP.BatteryMode.lower() in 'power': Status = PP.InitialStatus # Fix battery status if in power mode.
                
        # Updating fluxes
        if PP.FluxCalculation.lower() == 'yes':
            cav_e = np.dot(CI[4:8, i] - Ckb[order], eps ** k)
            cav_m = cav_e * Ki
            flux = cav_m * Dm / Lm
            for j in range(nv):
                FX[i, :, (j - 1) * 3:(j - 1) * 3 + 3] = \
                    flux[j] * CO[Status - 1]['Componets'][:, (j - 1) * 3:(j - 1) * 3 + 3]
                CM[i, :, j] = CO[Status - 1]['Concentrations'][:, j] * cav_m[j]

            CMA[i, :] = CO[Status - 1]['ConcentrationCoeff'] * cav_m
        vre = (VN[0,0] - VNO)/VNO
        i = i + 1
     
    # # Post-processing: removing the first points and Nan points.
    if i <= 2 or nc == 1e6: 
        Re = []
    else:
        rid = i - 1                                                            # Valid index.
        idx = np.where(np.any(CI[:,1:]<=cmin,axis = 0))[0] + 1 - 1             # Last index with concentration>cmin.
        if BattMode and len(idx)>0: rid = min([rid,idx[0]])                    # Remove the last points if in Power mode and very small concentration found.
        rid = rid + 1                                                          # 1 is Python special.
        ti = ti[1:rid]
        eps = eps[1:rid,:]
        As = As[1:rid,:]
        CI = CI[:, 1:rid].transpose()
        ncc = ncc[1:rid]
        V = np.transpose(V[:, 1:rid, :], (1, 2, 0)).reshape(rid-1,-1)
        C = np.transpose(C[:, 1:rid, :], (1, 2, 0)).reshape(rid-1,-1)
        CHW = CHW[:, 1:rid].transpose()
        SOC_t = SOC_t[1:rid]
        Vcell = np.transpose(Vcell[:, 1:rid])
        CMP = np.concatenate([ti.reshape(-1, 1), SOC_t.reshape(-1, 1), \
                              V, C[:, [12, 14, 13, 15]], ncc.reshape(-1, 1)], axis=1)
        S = S[~np.any(np.isnan(S), axis=1), :]
        #S = S[~np.any(S[:,1:3] == 0,axis=1),:]
        FX = []; CO = []; CM = []; CMA = []
        
        # Calculating other derived information.
        S = np.concatenate([S[:nc-1,:],np.ones((nc-1,3))*-9999],axis=1)
        if not BattMode and PP.PerformanceCalculation.lower() == 'yes':
            E = calcEnergy(PP,CMP,'rest') 
            if E.shape[0] == nc - 1: S = np.concatenate([S[:nc-1,:14],E[:nc-1,4:]],axis=1)

        # Output data in DataFrame format.
        pv_names = ['Time_s', 'SOC', 'Ec_V', 'Eeq_V', 'Eact_V', 'Econ_V', 'Eohm_V',\
                    'Epw_V', 'Edonnan_V', 'Eeq_p_V', 'Eeq_n_V','Eact_p_V','Eact_n_V',\
                    'Econ_p_V', 'Econ_n_V','CH_pe_mol_m3', 'CH_pe_mol_m3', \
                        'CW_pe_mol_m3', 'CW_ne_mol_m3', 'Cycle']
        CMP = pd.DataFrame(CMP,columns=pv_names)
        
        # S: cycle, charge_time, discharge_time, CE, charge cut off accuracy,
        # discharge cut off accurate, iter number charge, iter number discharge,
        # cycle start time, charge end time; charge end time, rest end time; rest
        # end time; discharge end time; charge energy, discharge energy (Wh).
        st_names = ['Cycle','Time_charge_s','Time_discharge_s','CE','Charge_cut_off_accuracy_V',\
                      'Discharge_cut_off_accuracy_V','Charge_iter_step','Discharge_iter_step',\
                          'Cycle_start_time_s','Charge_end_time_s','Charge_end_time_s',\
                              'Rest_end_time_s','Rest_end_time_s','Discharge_end_time_s',\
                                  'Charge_energy_Wh','Discharge_energy_Wh','EE']
        if S.shape[0] == 0: S = np.ones((1,len(st_names)),dtype=float)*-9999
        S = pd.DataFrame(S,columns=st_names)
        
        # Porosity and specific area.
        eps = pd.DataFrame(eps,columns=['Cathode','Anode'])
        As = pd.DataFrame(As,columns=['Cathode_m-1','Anode_m-1'])
        
        # CI: tanke concentration 2345; electrode outlet concentration 2345.
        ci_names = ['TankConcentrationV2_mol_m3','TankConcentrationV3_mol_m3',\
                  'TankConcentrationV4_mol_m3','TankConcentrationV5_mol_m3',\
                      'ElectrodeOutletConcentrationV2_mol_m3','ElectrodeOutletConcentrationV3_mol_m3',\
                          'ElectrodeOutletConcentrationV4_mol_m3','ElectrodeOutletConcentrationV5_mol_m3']
        CI = pd.DataFrame(CI,columns=ci_names)
        
        # CHW: tank HP, WP, HN, WN; electrode HP, WP, HN, WN.
        chw_names = ['PositiveTankProton_mol_m3','PositiveTankWater_mol_m3',\
                      'NegativeTankProton_mol_m3','NegativeTankWater_mol_m3',\
                      'PositiveElectrodeProton_mol_m3','PositiveElectrodeWater_mol_m3',\
                          'NegativeElectrodeProton_mol_m3','NegativeElectrodeWater_mol_m3']
        CHW = pd.DataFrame(CHW,columns=chw_names)
        Vcell = pd.DataFrame(Vcell,columns=['CellVoltage_V'])
        c_names = ['ElectrodeCenterlineV2_mol_m3','ElectrodeCenterlineV3_mol_m3',\
                    'ElectrodeCenterlineV4_mol_m3','ElectrodeCenterlineV5_mol_m3',\
                        'ElectrodeWallV2_mol_m3','ElectrodeWallV3_mol_m3',\
                            'ElectrodeWallV4_mol_m3','ElectrodeWallV5_mol_m3',\
                                'PositiveTankProton_mol_m3','PositiveTankWater_mol_m3',\
                                    'NegativeTankProton_mol_m3','NegativeTankWater_mol_m3',\
                                        'PositiveElectrodeProton_mol_m3','PositiveElectrodeWater_mol_m3',\
                                            'NegativeElectrodeProton_mol_m3','NegativeElectrodeWater_mol_m3']
        C = pd.DataFrame(C,columns=c_names)
        FX = pd.DataFrame(FX,columns=['Fluxes'])
        CM = pd.DataFrame(CM,columns=['MembraneConcentrations'])
        CMA = pd.DataFrame(CMA,columns=['MeanMembraneConcentrations'])
        CO = pd.DataFrame(CO,columns=['CrossOver'])
        
        # Save all data to a structure format.
        # CMP: Time, SOC, Vc, Veq, Vact, Vcon, Vohm, Vpw, Vdonna, Veq_p, Veq_n,
        # Vact_p,Vact_n, Vcon_p, Vcon_n, CH_p(electorde), CH_n, Cw_p, Cw_n, cycle.
        # S: cycle, charge_time, discharge_time, CE, charge cut off accuracy,
        # discharge cut off accurate, iter number charge, iter number discharge,
        # cycle start time, charge end time; charge end time, rest end time; rest
        # end time; discharge end time; charge energy, discharge energy (Wh).
        # CI: tanke concentration 2345; electrode outlet concentration 2345.
        # CHW: tank HP, WP, HN, WN; electrode HP, WP, HN, WN.
        # Vcell: total cell voltage.
        # C1-4: electrode centerline V2345; C5-8: electrode wall V2345; C9-12: tank
        # HP, WP, HN, WN; C13-16: electrode HP, WP, HN, WN.
        # V: Vc, Veq, Vact, Vcon, Vohm, Vpw, Vdonna, Veq_p, Veq_n, Vact_p,Vact_n,
        # Vcon_p, Vcon_n
        # E: cycle, charge time, discharge time, charge energy, discharge energy.
        # Efficiencies: cycle, CE, EE.
        # FX: flux at all membrane points.
        Re = {
        'Potentials': CMP,
        'Parameters': PP,
        'Statistics': S,
        'Porosity': eps,
        'SpecificArea': As,
        'InletConcentrations': CI,
        'ProtonWaterConcentrations': CHW,
        #'CellVoltage': Vcell,
        'ConcentrationCompoents': C,
        'Fluxes': FX,
        'MembraneConcentrations': CM,
        'MeanMembraneConcentrations': CMA,
        'CrossOver': CO
        }
        
        # Writing data to local disk.
        if hasattr(PP,'ParameterFilePath'):
            SDPF = PP.ParameterFilePath
            pv_path = SDPF.replace('.xlsx','_Potentials.csv')
            st_path = SDPF.replace('.xlsx','_Statistics.csv')
            Re['Potentials'].to_csv(pv_path,index=False)
            Re['Statistics'].to_csv(st_path,index=False)
        
    timeend = time.time()                                                      # End time of the code.
    tt = timeend-timestart
    if Display and nc==1: print(formatSpec7 %(tt))
    if Display and nc>1: print(formatSpec6 %(tt,tt/(nc-1)))
    return Re

#%% 
# ZIB is a unit cell model for Zinc-Iodine Flow Battery (ZIB). The output from#
# ZIB is equivalent to RFB is using the same Zinc battery paerameter file.    #
def ZIB(PP0):   
    # Mainbody.
    timestart=time.time()                                                      # Start time of the code.
    PP = copy.deepcopy(PP0)                                                    # Avoid parameter interactions.

    # Calculate and extract parameters from input parameters for later usage.
    Chemical = PP.Chemical
    Calculated = PP.Calculated
    Display = PP.Display.lower() == 'yes'                                      # If output information on screen (integer).
    t_start = PP.StartTime                                                     # Starting time.
    t_end = PP.EndTime                                                         # End of simulation time.
    cmin = PP.MininumConcentration                                             # Minimum concentration.
    
    # Identify active and non-active species.
    n = np.zeros(3,dtype=int)
    ixa = Chemical['Active']
    ixn = Chemical['NoneActive']
    
    # For active species (4 species).
    active_species = [Chemical["Name"][i] for i in ixa]
    C0C = Chemical['InitialConcentrationInCathode']                            # All initial concentrations in Cathode.
    C0A = Chemical['InitialConcentrationInAnode']
    C0C = np.array(C0C,dtype=float)
    C0A = np.array(C0A,dtype=float)
    CI0 = conc([C0C[i] for i in ixa],[C0A[i] for i in ixa])              # Concentrations 4(C/A),5(C/A),3(C/A),2(C/A).
    n[0] = len(CI0) // 2                                                       # Number of active species.
    
    # For non-active species (length(species) - 4).
    nonactive_species = [Chemical["Name"][i] for i in ixn]                     # For V: Water_P, Water_N, H_P, H_N;
    CS0 = conc([C0C[i] for i in ixn],[C0A[i] for i in ixn])              # For DHPS: Water_P, Water_N, OH_P, OH_N.
    n[1] = len(CS0) // 2                                                       # Number of non-active species.
    
    # For membrane species.
    membrane_species = []
    CM0 = []
    if PP.MembraneIonName not in Chemical['Name']:
        membrane_species.append(PP.MembraneIonName)
        CM0.extend([PP.MembraneIonInitialConcentrationInCathode, \
                    PP.MembraneIonInitialConcentrationInAnode])                # For membrane carrying ion;
    if PP.MembraneSolventName not in Chemical['Name']:
        membrane_species.append(PP.MembraneSolventName)
        CM0.extend([PP.MembraneSolventInitialConcentrationInCathode, \
                    PP.MembraneSolventInitialConcentrationInAnode])            # For water in negative.
    CM0 = np.array(CM0,dtype=float)
    n[2] = len(CM0) // 2                                                       # Number of membrane related species.
    
    # Parameters.
    VCutOffCharge = PP.CutOffVoltageCharge
    VCutOffDischarge = PP.CutOffVoltageDischarge
    if PP.BatteryMode.lower() != 'current':                                    # Using different cut off voltage.
        VCutOffCharge = PP.SystemCutOffVoltageCharge
        VCutOffDischarge = PP.SystemCutOffVoltageDischarge
        
    #Lh = PP.ElectrodeHeight;                                                  # Electrode height (m)
    dt = float(PP.TimeStep)
    t_wait = PP.ChargeRestTime
    isign_2D = Calculated['TwoD']
    I = PP.Current                                                             # Current (A).
    F = PP.Constants['Faraday']                                                # Faraday constant;
    VE = Calculated['ElectrodeVolume'];                                        # Electrode volume (m^3).
    NE = Chemical['ElectronNumber'][1]
    k = PP.Heterogeneity
    
    # Output information control
    formatSpec1 = 'Running a multi-cycle unit cell model.'
    formatSpec2 = 'Cycle %d: charge %.3f h, discharge %.3f h, CE: %.3f, Cumulative CE %.3f (%.1f%%).'
    formatSpec3 = 'Active species: {}.'
    #formatSpec3b = 'Active species: {0}, {1}, {2}, {3}.'
    formatSpec4 = 'Non active species: {}.'
    #formatSpec4b = 'Non active species: {0}, {1}.'
    formatSpec5 = 'Membrane species: {}.\n'
    #formatSpec5b = 'Membrane species: {0}.\n'
    formatSpec6 = 'Computation done, elapsed time %g seconds, average time per cycle %g seconds.\n';
    formatSpec7 = 'Computation done, elapsed time %g seconds.\n';
    msg = 'Negative concentration found, adjust time step or minimum concentration.';
    if Display: print(formatSpec1)
    if len(active_species)>0 and Display: print(formatSpec3.format(active_species))
    if len(nonactive_species)>0 and Display: print(formatSpec4.format(nonactive_species))
    if len(membrane_species)>0 and Display: print(formatSpec5.format(membrane_species))
    #print(formatSpec3.format(active_species[0],active_species[1],active_species[2],active_species[3]))
    #print(formatSpec4b.format(nonactive_species[0],nonactive_species[1]))
    #print(formatSpec5b.format(membrane_species[0]))
    
    # UCMCO model mesh/time/initializations.
    # Grid/time
    nc0 = PP.CycleNumber
    ni = PP.RefineTimeTimes                                                    # Refine times, minimum 1.
    nps = PP.RefineTimePoints                                                  # Number of points to refine, minimum 1.
    Nt = int(np.ceil((t_end - t_start) / dt)) + 1                              # Number of times.
    if nps + 1 >= Nt: nps = max(Nt - 2,1)                                      # Refine points number should less than total time number.
    ti = linspace(t_start, t_end, Nt)                                    # Time seris.
    Ntt = Nt + nc0 * (2*nps * (ni - 1) + nps*ni) + nps*(ni-1) + 1              # Estimated maximum time step.
    
    # The Cka, Ckb, Ckc, Ckd are reordered as 4/5/3/2 in Calculated parameters.#
    # However, the code below is designed based on the order V2,3,4,5. We need #
    # to reorder the 4/5/3/2 to 2/3/4/5 to avoid changing too many codes.      # 
    order = np.array([3, 2, 0, 1])
    ixa = np.array(Chemical['Active'])[order]
    Ki = np.array(Chemical['PartitionCoefficient'])[ixa]
    Dm = np.array(Chemical['DiffusivityInMembrane'])[ixa]
    Lm = PP.MembraneThickness
    nv = 4
    
    # Initialization for active, non-active, and membrane species. CI         #
    # represnts: V2345 in tank (1-4) and V2345 in electrode outlet (5-8).     #               
    CI = np.full((int(n[0]) * 2, Ntt), np.nan)                                 # Vanadium concentrations at outlet for tank and electrode.
    CI[0:4, 0] = CI0[[7, 5, 0, 2]]                                             # Tank concentration at outlet for V2,3,4,5.
    CI[4:8, 0] = CI[0:4, 0] + Calculated['Ckd'][order] * isign_2D              # Electrode concentrations at outlet for V4,5,3,2.
    eps = np.matlib.repmat(Calculated['Porosity'],Ntt,1)                       # Time varying porosity for cathode and anode.
    As = np.matlib.repmat(Calculated['SpecificArea'],Ntt,1)                    # Time varying specific area for cathode and anode.
    eps_neg0 = Calculated['Porosity'][1]                                       # Inital anode porosity.
 
    # V: CHW represents HP, WP, HN, WN for tank (1-4) and for electrode (5-8) #
    # DHPS:CHW represents OHP, WP, OHN, WN for tank (1-4) and for electrode (5-8).#
    CHW = np.full((int(n[1]) * 4, Ntt), np.nan)
    CHW[:, 0] = np.tile(CS0[[2, 0, 3, 1]], 2)                                  # Covert the order from WP, WN, HP, HN to HP WP HN WN.
    Nz = 1                                                                     # Nubmer of grid points along cell height direction.
    Vcell = np.full((Nz, Ntt), np.nan)                                         # Cell voltage and potential components at each time and electrode outlet.
    SOC_t = np.full(Ntt, np.nan)                                               # Calculated state of charge.
    ncc = np.full(Ntt, np.nan)                                                 # Cycle number.
    C = np.full((Nz, Ntt, 8 + 8), np.nan)                                      # Concentrations for V2/5 at cell center/wall (4+4), and proton/water (2+1).
    V = np.full((Nz, Ntt, 13), np.nan)                                         # Potentials for cell voltage (1) and components (6+6).
    S = np.full((300, 1 + 3 + 2 + 2 + 6), np.nan)                              # Statistics for each cycle.
    
    # Initial parameters controlling voltage cycles.
    t_charge = np.inf                                                          # The duration of charge for each cycle (s).
    t_cycle_old = t_start                                                      # The end time of previous discharge cycle (s).
    t_cycle = t_cycle_old
    nc = 1                                                                     # The index of cycle.
    sc = 3600                                                                  # Convert seconds to hour.
    
    # Calculating flux ratios if FluxCalculation is enabled.
    nm = PP.MembranePoints                                                     # Points along membrane thickness direction (1, no effect on voltage).
    FX = np.full((Ntt, nm, int(n[0]) * 3), np.nan)
    CM = np.full((Ntt, nm, int(n[0])), np.nan)
    CMA = np.full((Ntt, int(n[0])), np.nan)
    m1, r1, uct1, _, Ckb, _, co1, st1 = CrossOverRateUNIROVI(1, PP)            # Note: uct1 is always positive.
    m2, r2, uct2, _, _, _, co2, st2 = CrossOverRateUNIROVI(0, PP)
    m3, r3, uct3, _, _, _, co3, st3 = CrossOverRateUNIROVI(-1, PP)
    CO = [co1, co2, co3]
    
    # This is special case for Zinc battery.
    m1[[0,1,4,5],:] = 0
    m2[[0,1,4,5],:] = 0
    m3[[0,1,4,5],:] = 0
    
    # Initialization the calculation.
    Status = PP.InitialStatus                                                  # Initial status.
    dt0 =  PP.InitialTimeStep                                                  # Initial time step.
    eps[1,1] = eps_neg0 - I/F/NE/VE/CI[4,0]*dt0*Status                         # Update anode porosity.
    PP.AnodeElectrodePorosity = eps[0,1]                                       # Update parameters.
    As[1,:] = PP.Calculated['SpecificArea']                                    # Save the updated specific area.
    
    # Computing for the first time.
    TS = [t_cycle,t_start,t_start + dt0]
    CT0 = np.concatenate([CI[:, 0], [t_start]])
    if PP.InitialStatus == 1:
        CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
            (Status, TS, CI[:, 0], \
             CHW[:, 0], PP, m1, r1, uct1, st1, CT0)
    elif PP.InitialStatus == 0:
        CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
            (Status, TS, CI[:, 0], \
             CHW[:, 0], PP, m2, r2, uct2, st2, CT0)
    else:
        CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
            (Status, TS, CI[:, 0], \
             CHW[:, 0], PP, m3, r3, uct3, st3, CT0)      
            
    # Updating results for the first step.
    CI[:, 1] = CIN
    Vcell[:, 1] = VN[:, 0]
    SOC_t[1] = SOCN
    ncc[1] = 1
    V[:, 1, :] = VN
    C[:, 1, :] = CN
    CHW[:, 1] = CHWN
    nt0 = np.hstack([0, linspace(dt0 / dt, 1, nps * ni + 1)])       # This function is not consistant with Matlab.
    nt1 = linspace(0, 1, nps * ni + 1)
    ti = np.hstack([ti[0] + (ti[nps] - ti[0]) * nt0, ti[nps + 1 :]]) # Add multiple points at the beginning.
    Nt = Nt  - (nps + 1) + nps*ni + 2
    VNO = VN[0,0]; vre = 0
    
    # Updating cycling data.
    i = 2                                                                      # The second index.
    BattMode = PP.BatteryMode.lower() == 'power'
    #while i<Nt-nps - 1 and i <Ntt and vre<=0.5 and nc <= PP.CycleNumber:
    while i< Nt and i < Ntt and vre<=0.5 and nc <= PP.CycleNumber:
        ncc[i] = nc
        if (BattMode and np.all(CIN>cmin)): Status = PP.InitialStatus          # Fix battery status if in power mode.
        if (BattMode and np.any(CIN<=cmin)): break                             # Early exit if very small concentrations found.                      
        
        if Status == 1:
            # Updating parameters.
            TS = [t_cycle, ti[i-1], ti[i]]                                     # Update time information: cycle start time, old, and new time.
            eps[i,1] = eps[i-1,1] - I/F/NE/VE/CI[4,i-1]*(TS[2]-TS[1])          # Update anode porosity.
            PP.AnodeElectrodePorosity = eps[i,1]                               # Update parameters.
            m1, r1, uct1, _, Ckb, _, co1, st1 = CrossOverRateUNIROVI(1, PP)    # Note: uct1 is always positive.
            m1[[0,1,4,5],:] = 0
            
            # Updating concentration and voltage.
            CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
                (Status, TS, CI[:, i-1], CHW[:, i-1], PP, m1, r1, uct1, st1, CT0)
    
            # Checkign if all concentrations are valid.
            if np.any(CIN < 0): warnings.warn(msg); break
            As[i,:] = PP.Calculated['SpecificArea']                            # Save the updated specific area.
            eps[i,:] = PP.Calculated['Porosity']                               # Save the updated specific area.
            CI[:, i] = CIN; V[:, i, :] = VN; C[:, i, :] = CN
            SOC_t[i] = SOCN; 
            Vcell[:, i] = VN[:, 0]; 
            CHW[:, i] = CHWN
            
            # Identifying final points.
            #tioo = ti[i-1]; CIOO = CI[:,i-1]; CHWOO = CHW[:,i-1];             # Old time, concentrations.
            if Vcell[-1, i] > VCutOffCharge:
                tio = ti[i-1]                                                  # Old time.
                tin = ti[i]                                                    # New time.
                tif = (tio + tin) / 2                                          # Middle time.
                TS = [t_cycle, tio, tif, tin]
                CIO = CI[:, i-1]
                CHWO = CHW[:, i-1]
                eps[i,1] = eps[i-1,1] - I/F/NE/VE/CIO[4]*(TS[2]-TS[1])*Status*0# Update anode porosity.
                PP.AnodeElectrodePorosity = eps[i,1]                           # Update parameters.
                m1, r1, uct1, _, Ckb, _, co1, st1 = CrossOverRateUNIROVI(1, PP)# Note: uct1 is always positive.
                m1[[0,1,4,5],:] = 0
                
                CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI\
                    (Status, TS, CIO, CHWO, PP, m1, r1, uct1, st1, CT0)
                As[i,:] = PP.Calculated['SpecificArea']                        # Save the updated specific area.
                eps[i,:] = PP.Calculated['Porosity']                           # Save the updated specific area.
                count = 1
                err = abs(VN[-1, 0] - VCutOffCharge)
    
                while err > PP.CutOffVoltageTolerance and \
                    count <= PP.DischargeRefineTimeNumber - 1 + 2:
                    count = count + 1
                    if VN[-1, 0] < VCutOffCharge:
                        tio = TS[2]; tin = TS[3]; tif = (tio + tin) / 2
                        CIO = CIN ; CHWO = CHWN
                    else:
                        tio = TS[1]; tin = TS[2]; tif = (tio + tin) / 2
                    TS = [t_cycle, tio, tif, tin]
                    eps[i,1] = eps[i-1,1] - I/F/NE/VE/CIO[4]*(TS[2]-TS[1])*Status# Update anode porosity.
                    PP.AnodeElectrodePorosity = eps[i,1]                       # Update parameters.
                    m1, r1, uct1, _, Ckb, _, co1, st1 = CrossOverRateUNIROVI(1, PP)# Note: uct1 is always positive.
                    m1[[0,1,4,5],:] = 0
                  
                    CIN, VN, CN, SOCN, CHWN, _, _  = \
                        UpdateCVUNIROVI(Status, TS, CIO, CHWO, PP, m1, r1, uct1, st1, CT0)
                    As[i,:] = PP.Calculated['SpecificArea']                    # Save the updated specific area.
                    eps[i,:] = PP.Calculated['Porosity']                       # Save the updated specific area.
                    err = abs(VN[-1, 0] - VCutOffCharge)
                    
                ti[i] = tif
                CI[:, i] = CIN
                V[:, i, :] = VN
                C[:, i, :] = CN
                SOC_t[i] = SOCN
                Vcell[:, i] = VN[:, 0]
                CHW[:, i] = CHWN
                t_charge = ti[i] - t_cycle                                     # Recording the time of end charging.
                CT0 = np.concatenate([CIN, [ti[i]]])
                S[nc-1, [0, 1, 4, 6, 8, 9]] = [nc, t_charge, Vcell[-1, i]\
                                               - VCutOffCharge, count, t_cycle, ti[i]]
                Status = 0
                if PP.BatteryMode.lower() in 'power': Status = PP.InitialStatus# Fix battery status if in power mode.
                
                # Early termination if index exceeds the maximum size.
                if i + 1>=Nt: i = i + 1; break                                 

                # Add points if necessary.
                if ti[i] + t_wait < ti[i+1]:
                    ti = np.concatenate([ti[:i], ti[i] + nt1 * t_wait, ti[i+1:]])
                    Nt = Nt-1 + nps*ni + 1
                elif ti[i] + t_wait == ti[i+1]:
                    ti = np.concatenate([ti[:i], ti[i] + nt1 * t_wait, ti[i+2:]])
                    Nt = Nt-2 + nps*ni + 1
                else:
                    idk = np.where(ti[i+1:] < ti[i] + t_wait)[0][-1] + 1
                    ti = np.concatenate([ti[:i+idk], ti[i+idk] + \
                                         (ti[i] + t_wait - ti[i+idk]) * nt1, ti[i+idk+1:]]) 
                    Nt = Nt - 1 + nps * ni + 1
                
            if PP.FluxCalculation.lower() == 'yes':
                cav1_e = (CI[4:8, i] - Ckb[order]) * eps**k
                cav1_m = cav1_e * Ki
                flux1 = cav1_m * Dm / Lm
        
                for j in range(nv):
                    FX[i, :, (j-1)*3:(j*3)] = flux1[j] * co1['Componets'][:, (j-1)*3:(j*3)]
                    CM[i, :, j] = co1['Concentrations'][:, j] * cav1_m[j]
            
                CMA[i, :] = cav1_m * co1['ConcentrationCoeff']
        elif Status == 0:
            if ti[i] < t_charge + t_wait + t_cycle:
                As[i,:] = PP.Calculated['SpecificArea']                        # Save the updated specific area.
                eps[i,:] = PP.Calculated['Porosity']                           # Save the updated specific area.
                m2, r2, uct2, _, _, _, co2, st2 = CrossOverRateUNIROVI(0, PP)
                m2[[0,1,4,5],:] = 0
                #TS = [t_cycle, ti[i - 1], ti[i]]
                CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(0, TS, CI[:, i - 1], CHW[:, i - 1], PP, m2, r2, uct2, st2, CT0)
                CI[:, i] = CIN
                V[:, i, :] = VN
                C[:, i, :] = CN
                SOC_t[i] = SOCN
                Vcell[:, i] = VN[:, 0]
                CHW[:, i] = CHWN
            else:
                # ti[i] - (t_charge + t_wait + t_cycle)                        # Will always be 0.
                TS = [t_cycle, ti[i - 1], ti[i]]
                eps[i,1] = eps[i-1,1] - I/F/NE/VE/CI[4,i-1]*(TS[2]-TS[1])*0    # Update anode porosity.
                PP.AnodeElectrodePorosity = eps[i,1]                           # Update parameters.
                m2, r2, uct2, _, _, _, co2, st2 = CrossOverRateUNIROVI(0, PP)
                m2[[0,1,4,5],:] = 0
                CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(0, TS, CI[:, i - 1], CHW[:, i - 1], PP, m2, r2, uct2, st2, CT0)
                As[i,:] = PP.Calculated['SpecificArea']                        # Save the updated specific area.
                eps[i,:] = PP.Calculated['Porosity']                           # Save the updated specific area.
                CI[:, i] = CIN
                V[:, i, :] = VN
                C[:, i, :] = CN
                SOC_t[i] = SOCN
                Vcell[:, i] = VN[:, 0]
                CHW[:, i] = CHWN
                CT0 = np.concatenate((CIN, [ti[i]]))
                S[nc-1, [10, 11]] = [S[nc-1, 9], ti[i]]
                ti = np.concatenate([ti[:i], ti[i] + (ti[i+nps] - ti[i]) * nt1, ti[i+nps+1:]])
                Nt = Nt - (nps+1) + ni*nps + 1
                Status = -1
                if PP.BatteryMode.lower() in 'power': Status = PP.InitialStatus# Fix battery status if in power mode.

            # Updating fluxes
            if PP.FluxCalculation.lower() == 'yes':
                cav2_e = (CI[4:8, i] - Ckb[order]) @ eps**k
                cav2_m = cav2_e * Ki
                flux2 = cav2_m * Dm / Lm
                for j in range(nv):
                    FX[i, :, (j - 1) * 3:(j - 1) * 3 + 3] = flux2[j] * co2.Componets[:, (j - 1) * 3:(j - 1) * 3 + 3]
                    CM[i, :, j] = co2.Concentrations[:, j] * cav2_m[j]
                CMA[i, :] = co2['ConcentrationCoeff'] * cav2_m
        else:
            tdc = t_cycle + t_charge + t_wait
            TS = [tdc, ti[i - 1], ti[i]]
            eps[i,1] = eps[i-1,1] - I/F/NE/VE/CI[4,i-1]*(TS[2]-TS[1])*Status   # Update anode porosity.
            PP.AnodeElectrodePorosity = eps[i,1]                               # Update parameters.
            m3, r3, uct3, _, _, _, co3, st3 = CrossOverRateUNIROVI(-1, PP)
            m3[[0,1,4,5],:] = 0
            CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(Status, TS, CI[:, i - 1], CHW[:, i - 1], PP, m3, r3, uct3, st3, CT0) 
            As[i,:] = PP.Calculated['SpecificArea']                            # Save the updated specific area.
            eps[i,:] = PP.Calculated['Porosity']                               # Save the updated specific area.
            CI[:, i] = CIN
            V[:, i, :] = VN
            C[:, i, :] = CN
            SOC_t[i] = SOCN
            Vcell[:, i] = VN[:, 0]
            CHW[:, i] = CHWN
            
            if Vcell[-1, i] < VCutOffDischarge:
                tio = ti[i - 1]
                tin = ti[i]
                tif = (tio + tin) / 2
                CIO = CI[:, i - 1]
                TS = [tdc, tio, tif]
                CHWO = CHW[:, i - 1]
                eps[i,1] = eps[i-1,1] - I/F/NE/VE/CIO[4]*(TS[2]-TS[1])*Status  # Update anode porosity.
                PP.AnodeElectrodePorosity = eps[i,1]                           # Update parameters.
                m3, r3, uct3, _, _, _, co3, st3 = CrossOverRateUNIROVI(-1, PP)
                m3[[0,1,4,5],:] = 0
                CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(Status, TS, CIO, CHWO, PP, m3, r3, uct3, st3, CT0)
                As[i,:] = PP.Calculated['SpecificArea']                        # Save the updated specific area.
                eps[i,:] = PP.Calculated['Porosity']                           # Save the updated specific area.
             
                count = 1
                err = abs(VN[-1, 0] - VCutOffDischarge)
                while err > PP.CutOffVoltageTolerance and count <= PP.DischargeRefineTimeNumber - 1 + 2:
                    if VN[-1, 0] > VCutOffDischarge:
                        tio = tif
                        tif = (tio + tin) / 2
                        CIO = CIN
                        CHWO = CHWN
                    else:
                        tin = tif
                        tif = (tio + tin) / 2
    
                    TS = [tdc, tio, tif]
                    eps[i,1] = eps[i-1,1] - I/F/NE/VE/CIO[4]*(TS[2]-TS[1])*Status  # Update anode porosity.
                    PP.AnodeElectrodePorosity = eps[i,1]                       # Update parameters.
                    m3, r3, uct3, _, _, _, co3, st3 = CrossOverRateUNIROVI(-1, PP)
                    m3[[0,1,4,5],:] = 0
                    CIN, VN, CN, SOCN, CHWN, _, _ = UpdateCVUNIROVI(Status, TS, CIO, CHWO, PP, m3, r3, uct3, st3, CT0)
                    As[i,:] = PP.Calculated['SpecificArea']                    # Save the updated specific area.
                    eps[i,:] = PP.Calculated['Porosity']                       # Save the updated specific area.
                    err = abs(VN[-1, 0] - VCutOffDischarge)
                    count = count + 1
    
                ti[i] = tif                                                    # Update time
                CI[:, i] = CIN
                V[:, i, :] = VN
                C[:, i, :] = CN
                CHW[:,i] = CHWN
                SOC_t[i] = SOCN
                Vcell[:, i] = VN[:, 0]
                CT0 = np.concatenate((CIN, [ti[i]]))
                S[nc-1, [2, 5, 12, 13]] = [tif - tdc, Vcell[-1, i] - VCutOffDischarge, tdc, tif]
                S[nc-1, 3] = S[nc-1, 2] / S[nc-1, 1]
                S[nc-1, 7] = count
    
                # Update information for the next cycle.
                if Display and PP.InitialStatus == 1:
                    print(formatSpec2 %(S[nc - 1, 0],S[nc - 1, 1]/sc,
                    S[nc - 1, 2]/sc,S[nc - 1, 2]/S[nc-1, 1],
                    S[nc - 1, 2]/S[0, 1],100 * ti[i]/ti[-1]))
                    
                t_cycle_old = ti[i]
                t_cycle = t_cycle_old
                ti = np.concatenate([ti[:i], t_cycle + (ti[i + nps] - t_cycle) * nt1, ti[i + nps + 1:]])
                Nt = Nt - (nps + 1) + ni*nps + 1
                nc = nc + 1
                Status = 1
                if PP.BatteryMode.lower() in 'power': Status = PP.InitialStatus  # Fix battery status if in power mode.

        # Updating fluxes
        if PP.FluxCalculation.lower() == 'yes':
            cav_e = np.dot(CI[4:8, i] - Ckb[order], eps ** k)
            cav_m = cav_e * Ki
            flux = cav_m * Dm / Lm
            for j in range(nv):
                FX[i, :, (j - 1) * 3:(j - 1) * 3 + 3] = \
                    flux[j] * CO[Status - 1]['Componets'][:, (j - 1) * 3:(j - 1) * 3 + 3]
                CM[i, :, j] = CO[Status - 1]['Concentrations'][:, j] * cav_m[j]
    
            CMA[i, :] = CO[Status - 1]['ConcentrationCoeff'] * cav_m
        vre = (VN[0,0] - VNO)/VNO
        i = i + 1

    #%% Post-processing.
    #rid = min([i, Nt])
    #idx = np.where(np.any(CI<=cmin,axis = 0))[0]
    #if BattMode and len(idx)>0: rid = min([rid,idx[0]])                       # Remove the last points if in Power mode and very small concentration found.
    rid = i - 1                                                                # Valid index.
    idx = np.where(np.any(CI[:,1:]<=cmin,axis = 0))[0] + 1 - 1                 # Last index with concentration>cmin.
    if BattMode and len(idx)>0: rid = min([rid,idx[0]])                        # Remove the last points if in Power mode and very small concentration found.
    rid = rid + 1                                                              # 1 is Python special.
    ti = ti[1:rid]
    eps = eps[1:rid,:]
    As = As[1:rid,:]
    CI = CI[:, 1:rid].transpose()
    ncc = ncc[1:rid]
    V = np.transpose(V[:, 1:rid, :], (1, 2, 0)).reshape(rid-1,-1)
    C = np.transpose(C[:, 1:rid, :], (1, 2, 0)).reshape(rid-1,-1)
    CHW = CHW[:, 1:rid].transpose()
    SOC_t = SOC_t[1:rid]
    Vcell = np.transpose(Vcell[:, 1:rid])
    CMP = np.concatenate([ti.reshape(-1, 1), SOC_t.reshape(-1, 1), \
                          V, C[:, [12, 14, 13, 15]], ncc.reshape(-1, 1)], axis=1)
    S = S[~np.any(np.isnan(S), axis=1), :]
    FX = []; CO = []; CM = []; CMA = []

    # Calculating other derived information.
    E = np.zeros((0,7),dtype=float)
    if not BattMode and PP.PerformanceCalculation.lower() == 'yes':
        E = calcEnergy(PP,CMP,'rest') 
        S = np.concatenate([S[:nc-1,:],E[:nc-1,4:]],axis=1)
    
    # Output data in DataFrame format.
    pv_names = ['Time_s', 'SOC', 'Ec_V', 'Eeq_V', 'Eact_V', 'Econ_V', 'Eohm_V',\
                'Epw_V', 'Edonnan_V', 'Eeq_p_V', 'Eeq_n_V','Eact_p_V','Eact_n_V',\
                'Econ_p_V', 'Econ_n_V','CH_pe_mol_m3', 'CH_pe_mol_m3', \
                    'CW_pe_mol_m3', 'CW_ne_mol_m3', 'Cycle']
    CMP = pd.DataFrame(CMP,columns=pv_names)
    
    # S: cycle, charge_time, discharge_time, CE, charge cut off accuracy,
    # discharge cut off accurate, iter number charge, iter number discharge,
    # cycle start time, charge end time; charge end time, rest end time; rest
    # end time; discharge end time; charge energy, discharge energy (Wh).
    st_names = ['Cycle','Time_charge_s','Time_discharge_s','CE','Charge_cut_off_accuracy_V',\
                  'Discharge_cut_off_accuracy_V','Charge_iter_step','Discharge_iter_step',\
                      'Cycle_start_time_s','Charge_end_time_s','Charge_end_time_s',\
                          'Rest_end_time_s','Rest_end_time_s','Discharge_end_time_s',\
                              'Charge_energy_Wh','Discharge_energy_Wh','EE']
    if S.shape[0] == 0: S = np.ones((1,len(st_names)),dtype=float)*-9999
    S = pd.DataFrame(S,columns=st_names)

    # Porosity and specific area.
    eps = pd.DataFrame(eps,columns=['Cathode','Anode'])
    As = pd.DataFrame(As,columns=['Cathode_m-1','Anode_m-1'])
    
    # CI: tanke concentration 2345; electrode outlet concentration 2345.
    ci_names = ['TankConcentrationV2_mol_m3','TankConcentrationV3_mol_m3',\
              'TankConcentrationV4_mol_m3','TankConcentrationV5_mol_m3',\
                  'ElectrodeOutletConcentrationV2_mol_m3','ElectrodeOutletConcentrationV3_mol_m3',\
                      'ElectrodeOutletConcentrationV4_mol_m3','ElectrodeOutletConcentrationV5_mol_m3']
    CI = pd.DataFrame(CI,columns=ci_names)

    # CHW: tank HP, WP, HN, WN; electrode HP, WP, HN, WN.
    chw_names = ['PositiveTankProton_mol_m3','PositiveTankWater_mol_m3',\
                  'NegativeTankProton_mol_m3','NegativeTankWater_mol_m3',\
                  'PositiveElectrodeProton_mol_m3','PositiveElectrodeWater_mol_m3',\
                      'NegativeElectrodeProton_mol_m3','NegativeElectrodeWater_mol_m3']
    CHW = pd.DataFrame(CHW,columns=chw_names)
    Vcell = pd.DataFrame(Vcell,columns=['CellVoltage_V'])
    c_names = ['ElectrodeCenterlineV2_mol_m3','ElectrodeCenterlineV3_mol_m3',\
               'ElectrodeCenterlineV4_mol_m3','ElectrodeCenterlineV5_mol_m3',\
                   'ElectrodeWallV2_mol_m3','ElectrodeWallV3_mol_m3',\
                       'ElectrodeWallV4_mol_m3','ElectrodeWallV5_mol_m3',\
                           'PositiveTankProton_mol_m3','PositiveTankWater_mol_m3',\
                               'NegativeTankProton_mol_m3','NegativeTankWater_mol_m3',\
                                   'PositiveElectrodeProton_mol_m3','PositiveElectrodeWater_mol_m3',\
                                       'NegativeElectrodeProton_mol_m3','NegativeElectrodeWater_mol_m3']
    C = pd.DataFrame(C,columns=c_names)
    FX = pd.DataFrame(FX,columns=['Fluxes'])
    CM = pd.DataFrame(CM,columns=['MembraneConcentrations'])
    CMA = pd.DataFrame(CMA,columns=['MeanMembraneConcentrations'])
    CO = pd.DataFrame(CO,columns=['CrossOver'])

    # Save all data to a structure format.
    # CMP: Time, SOC, Vc, Veq, Vact, Vcon, Vohm, Vpw, Vdonna, Veq_p, Veq_n,
    # Vact_p,Vact_n, Vcon_p, Vcon_n, CH_p(electorde), CH_n, Cw_p, Cw_n, cycle.
    # S: cycle, charge_time, discharge_time, CE, charge cut off accuracy,
    # discharge cut off accurate, iter number charge, iter number discharge,
    # cycle start time, charge end time; charge end time, rest end time; rest
    # end time; discharge end time; charge energy, discharge energy (Wh).
    # CI: tanke concentration 2345; electrode outlet concentration 2345.
    # CHW: tank HP, WP, HN, WN; electrode HP, WP, HN, WN.
    # Vcell: total cell voltage.
    # C1-4: electrode centerline V2345; C5-8: electrode wall V2345; C9-12: tank
    # HP, WP, HN, WN; C13-16: electrode HP, WP, HN, WN.
    # V: Vc, Veq, Vact, Vcon, Vohm, Vpw, Vdonna, Veq_p, Veq_n, Vact_p,Vact_n,
    # Vcon_p, Vcon_n
    # E: cycle, charge time, discharge time, charge energy, discharge energy.
    # Efficiencies: cycle, CE, EE.
    # FX: flux at all membrane points.
    Re = {
    'Potentials': CMP,
    'Parameters': PP,
    'Statistics': S,
    'Porosity': eps,
    'SpecificArea': As,
    'InletConcentrations': CI,
    'ProtonWaterConcentrations': CHW,
    #'CellVoltage': Vcell,
    'ConcentrationCompoents': C,
    'Fluxes': FX,
    'MembraneConcentrations': CM,
    'MeanMembraneConcentrations': CMA,
    'CrossOver': CO
    }
    
    # Writing data to local disk.
    if hasattr(PP,'ParameterFilePath'):
        SDPF = PP.ParameterFilePath
        pv_path = SDPF.replace('.xlsx','_Potentials.csv')
        st_path = SDPF.replace('.xlsx','_Statistics.csv')
        Re['Potentials'].to_csv(pv_path,index=False)
        Re['Statistics'].to_csv(st_path,index=False)
        
    timeend = time.time()                                                      # End time of the code.
    tt = timeend-timestart
    if Display and nc==1: print(formatSpec7 %(tt))
    if Display and nc>1: print(formatSpec6 %(tt,tt/(nc-1)))
    return Re