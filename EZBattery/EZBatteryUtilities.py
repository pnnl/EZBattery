#%%######################### Descriptions #####################################
# This file includes utilities for flow battery modeling.The code is developed#
# by Dr. Yunxiang Chen @PNNL, please contact yunxiang.chen@pnnl.gov for       #  
# questions. Potential users are permitted to use for non-commercial purposes #
# without modifying any part of the code. Please contact the author to obtain #
# written permission for commercial or modifying usage.                       # 
            
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
             
#%% Loading required packages.
import numpy as np
import warnings, copy
import pandas as pd
from ismember import ismember
from .mathm import cross_over_rhs
from .mathm import BV

#%%
# This code is designed for both inorganic (all vanadium) and organic         #
# (DHP family) redox flow batteries. The code reads various parameters of     #
# redox flow batteries in the format of PP and output data in format of       #
# of Re which includes cell voltage, potential components, concentrations,    #
# and statistic information. The code is derived based on a 2D analytical     #
# modeling for potentials and concentrations.                                 #
############### The Code is developed by Yunxiang Chen @PNNL. #################
def CrossOverRateUNIROVI(Status, P):
    # This function calculates the coefficient matrix and source term of the  #
    # ODE system for cross-over and self-discharge effects.                   # 
    
    # Extracting parameters for further usage.                   
    Constants = P.Constants                                                    # Datatype: float dict.
    Calculated = P.Calculated
    Chemical = P.Chemical
    
    psi = P.MembraneSolventElectroOsmoticCoeffcient                            # Water electro-osmotic coefficient.
    lambda_val = P.MembraneSolventContentCoefficient                           # Water content coefficient.
    cf = float(P.MembraneChargeSiteConcentration)
    Lm = P.MembraneThickness
    VTP = P.CathodeTankVolume
    VTN = P.AnodeTankVolume
    W_pos = P.CathodePumpRate
    W_neg = P.AnodePumpRate
    I = P.Current
    F = Constants['Faraday']
    R = Constants['GasConstant']
    T = P.Temperature
    k = P.Heterogeneity
    nm = P.MembranePoints
    isign_2D = Calculated['TwoD']
    
    fluxcal = P.FluxCalculation
    symbolic = P.Symbolic
    if P.SDPFVersion.lower() == 'v4':
        fluxcal = P.FluxCalculation.lower() == 'yes'
        symbolic = P.Symbolic.lower() == 'yes'
    
    # Other parameters directly derived from input parameters P.
    eps_pos = Calculated['Porosity'][0]                                        # Cathode electrode porosity
    eps_neg = Calculated['Porosity'][1]                                        # Anode electrode porosity
    Sigma_m = Calculated['ElectronicConductivityMembrane']                     # Derived membrane conductivity (S/m)
    VE = Calculated['ElectrodeVolume']
    Ae = Calculated['ElectrodeArea']
    NP = Chemical['ElectronNumber'][0]
    NE = Chemical['ElectronNumber'][1]
    
    order = np.array([3, 2, 0, 1])
    ixa = np.array(Chemical['Active'])[order]
    zv = np.array(Chemical['Valence'],dtype=float)[ixa]
    Dm = np.array(Chemical['DiffusivityInMembrane'],dtype=float)[ixa]
    cc = np.array(Chemical['Coefficient'],dtype=float)[ixa]                    # Coefficients of vanadium ions V2345.
    Ki = np.array(Chemical['PartitionCoefficient'],dtype=float)[ixa]
    Cka = Calculated['Cka'][order] * isign_2D
    Ckb = Calculated['Ckb'][order] * isign_2D
    sn = [NE, NE, NP, NP]
    
    # Modified on 10/13/2023 for Zinc battery.
    sa_n = [1, -1, -1, 1]
    sa_m = [0, 0, 0, 0]
    if 'Zinc' in Chemical['Name'] or 'Iodide' in Chemical['Name']:
        sa_m = [-1, 1, 0, 0]                                                   # For Zinc RFB.
    sa = np.array(sa_n) + np.array(sa_m)
    
    # Further calculation.
    I0 = I / Ae
    nv = len(zv)
    
    pct = np.zeros((nm, 3 * nv))
    cmx = np.zeros((nm, nv))
    g = np.zeros(nv)
    uc = np.zeros(nv)
    s = np.zeros(nv)
    f = np.zeros(nv)
    CO = {}
    dx0 = 1e-16
    x = np.linspace(dx0, 1 - dx0, nm)
    x = x.reshape((-1,1))

    if type(P.DecayMode) != str:
        P.DecayMode = 'none'
    
    if P.DecayMode.lower() == 'none':
        uc[:] = np.zeros(nv)
        pct[:, 0:3:2] = 0
        cmx[:] = 0
    elif P.DecayMode.lower() == 'diffusion':
        f = np.array([1, 1, -1, -1])
        uc[:] = Dm / Lm * f
        pct[:, 0:3:2] = np.tile(f, (nm, 1))
        g = np.array([0.5, 0.5, 0.5, 0.5])
        cmx[:] = np.concatenate((1-x, 1-x, x, x),axis=1)
    elif P.DecayMode.lower() == 'all':
        if Status == 0:
            f = np.array([1, 1, -1, -1])
            uc = f * Dm / Lm
            pct[:, [0, 3]] = np.tile(f[:2], (nm, 1))
            pct[:, [6, 9]] = np.tile(f[2:], (nm, 1))
            g = np.array([0.5, 0.5, 0.5, 0.5])
            s = np.array([0, 0, 0, 0])
            cmx[:] = np.concatenate((1-x, 1-x, x, x),axis=1)
        else:
            if symbolic:
                raise ValueError('Symbolic calculation is not supported in Python.')
                # d1 = 40 # Adjust the precision if needed
                # getcontext().prec = d1
                # pct = np.zeros((nm, 3 * nv), dtype=object)
                # cmx = np.zeros((nm, nv), dtype=object)
                # g = np.zeros(nv, dtype=object)
                # #xi_m = np.array(zv * F / Sigma_m / R / T, dtype=object)
                # xi_m1 = Decimal(zv[0]) * Decimal(F) / Decimal(Sigma_m) / Decimal(R) / Decimal(T)
                # xi_m2 = Decimal(zv[1]) * Decimal(F) / Decimal(Sigma_m) / Decimal(R) / Decimal(T)
                # xi_m3 = Decimal(zv[2]) * Decimal(F) / Decimal(Sigma_m) / Decimal(R) / Decimal(T)
                # xi_m4 = Decimal(zv[3]) * Decimal(F) / Decimal(Sigma_m) / Decimal(R) / Decimal(T)
                # #xi_c = np.array(psi / Dm / F / lambda_val / cf, dtype=object)
                # xi_c1 = Decimal(psi)/Decimal(Dm[0])/Decimal(F)/Decimal(lambda_val)/Decimal(cf)
                # s = (xi_m + xi_c) * Lm * I0 * np.sign(Status)
                # f = np.zeros(nv, dtype=object)
                # # pct = np.zeros((nm, 3 * nv))
                # # cmx = np.zeros((nm, nv))
                # # g = np.zeros(nv)
                # # xi_m = np.array(zv * F / Sigma_m / R / T)
                # # xi_c = np.array(psi / Dm / F / lambda_val / cf)
                # s = (xi_m + xi_c) * Lm * I0 * np.sign(Status)
                # #f = np.array([s[0:2] / (np.exp(s[0:2]) - 1), s[2:4] / (np.exp(-s[2:4]) - 1)])
                # # f[0:2] = s[0:2] / (np.exp(s[0:2]) - 1)
                # # f[2:4] = s[2:4] / (np.exp(-s[2:4]) - 1)
                # uc = f * Dm / Lm
            else:
                xi_m = zv * F / Sigma_m / R / T
                xi_c = psi / Dm / F / lambda_val / cf
                s = (xi_m + xi_c) * Lm * I0 * np.sign(Status)
                #f = np.array([s[0:2] / (np.exp(s[0:2]) - 1), s[2:4] / (np.exp(-s[2:4]) - 1)])
                f[0:2] = s[0:2] / (np.exp(s[0:2]) - 1)
                f[2:4] = s[2:4] / (np.exp(-s[2:4]) - 1)
                uc = f * Dm / Lm
                
            if fluxcal:
                #if symbolic:
                rm = xi_m / (xi_m + xi_c)
                pct23 = s[0:2] * np.exp(s[0:2] * (1 - x)) / (np.exp(s[0:2]) - 1)
                pct[:, [0, 3]] = pct23                                         # Non-dimensional Diffusion contribution
                pct[:, [1, 4]] = (f[0:2] - pct23) * rm[0:2]                    # Non-dimensional Migration contribution
                pct[:, [2, 5]] = (f[0:2] - pct23) * (1 - rm[0:2])              # Non-dimensional Convection contribution
                pct45 = s[2:4] * np.exp(-s[2:4] * x) / (np.exp(-s[2:4]) - 1)
                pct[:, [6, 9]] = pct45
                pct[:, [7, 10]] = (f[2:4] - pct45) * rm[2:4]
                pct[:, [8, 11]] = (f[2:4] - pct45) * (1 - rm[2:4])
                g[0:2] = (np.exp(s[0:2]) - 1 - s[0:2]) / (s[0:2] * (np.exp(s[0:2]) - 1))
                g[2:4] = (1 - np.exp(-s[2:4]) - s[2:4]) / (s[2:4] * (np.exp(-s[2:4]) - 1))
                cmx[:, 0:2] = (np.exp(s[0:2] * (1 - x)) - 1) / (np.exp(s[0:2]) - 1)
                cmx[:, 2:4] = (np.exp(-x * s[2:4]) - 1) / (np.exp(-s[2:4]) - 1)
    else:
        raise ValueError('Wrong decay mode.')
    
    if np.any(uc[:2] < 0) or np.any(uc[2:4] > 0):
        raise ValueError('Wrong flux.')
    
    uct = np.abs(uc) * Ki                                                      # Consistent with Eq 22 in JPS paper.
    
    CO = {
        'MembranePoints': x,
        'CrossOverRate': uc,
        'Componets': pct,
        'ConcentrationCoeff': g,
        'NonDimensionalFlux': f,
        'NonDimesionParameter': s,
        'Concentrations': cmx
    }
    
    q = np.array([[1, 0, 1, 2],
                  [0, 1, -2, -3],
                  [-3, -2, 1, 0],
                  [2, 1, 0, 1]])
    W = np.array([W_neg, W_neg, W_pos, W_pos])
    VT = np.array([VTN, VTN, VTP, VTP])
    eps_np = np.array([eps_neg, eps_neg, eps_pos, eps_pos])
    S1 = -W / VT * Cka
    S2 = (cc*sa*I/F/sn + W * Cka + Ae * np.dot(q, uct * Ckb * eps_np ** k)) / (eps_np * VE)
    M22 = np.zeros((4, 4))
    M22[0, :] = (-(W_neg + eps_neg ** k * Ae * uct[0]), 0, -eps_pos ** k * Ae * uct[2], -2 * eps_pos ** k * Ae * uct[3]) / (eps_np * VE)
    M22[1, :] = (0, -(W_neg + eps_neg ** k * Ae * uct[1]), 2 * eps_pos ** k * Ae * uct[2], 3 * eps_pos ** k * Ae * uct[3]) / (eps_np * VE)
    M22[2, :] = (3 * eps_neg ** k * Ae * uct[0], 2 * eps_neg ** k * Ae * uct[1], -(W_pos + eps_pos ** k * Ae * uct[2]), 0) / (eps_np * VE)
    M22[3, :] = (-2 * eps_neg ** k * Ae * uct[0], -Ae * eps_neg ** k * uct[1], 0, -(W_pos + eps_pos ** k * Ae * uct[3])) / (eps_np * VE)
    
    M = np.block([[-np.diag(W / VT), np.diag(W / VT)], [np.diag(W / eps_np / VE), M22]])
    S = np.concatenate((S1, S2)) * Status
    uct = np.abs(uct)
    
    # For high electrode tank volume ratio cases
    ST = sa * I / F / VT / sn * Status

    return M, S, uct, Cka, Ckb, Sigma_m, CO, ST
 
#%%
# This function updates the concentrations and cell voltage (potentials )     #
# at a new time step TS(3) based on the concentrations and parameters at      #
# the previous time step TS(2) for a cycle starting at TS(1). The output      #
# variables are: CIN: Tank (1-4, 2345), Electrode outlet (5-8, 2345); VN:     #
# Potentials: Total, Eq, act, con, ohm, OH/H&W, Donna, Eq_P, Eq_N, Act_P,     # 
# Act_N, Con_P, Con_N; CN: Centerline (1-4),Wall(5-8),OH/H Water (tank:       #
# 9-12, electrode 13-16).                                                     #
def UpdateCVUNIROVI(Status, TS, CIO, CHWO, P, M, S, uc, ST, CT0):
    # Extracting parameters
    tio = TS[1]
    tin = TS[2]
    t_start = P.StartTime

    # Local parameters
    Calculated = P.Calculated
    Chemical = P.Chemical
    Constants = P.Constants
    iszinc = 1 if 'Zinc' in Chemical['Name'] or 'Iodide' in Chemical['Name'] else 0

    T = P.Temperature
    R = Constants['GasConstant']
    F = Constants['Faraday']
    alpha_pos = P.CathodeReactionTransferCoefficient
    alpha_neg = P.AnodeReactionTransferCoefficient
    W_pos = P.CathodePumpRate
    W_neg = P.AnodePumpRate
    Lh = P.ElectrodeHeight 
    Lw = P.ElectrodeWidth
    nh = P.MembraneIonDragCoefficient                                          # Proton draf coefficient (1).
    nd = P.MembraneSolvantDragCoefficient                                      # Water drag coefficieint (1).
    VSMALL = P.MininumConcentration
    k = P.Heterogeneity
    E01 = Chemical['StandardPotential'][1]
    E02 = Chemical['StandardPotential'][0]
    NP = Chemical['ElectronNumber'][0]
    NE = Chemical['ElectronNumber'][1]
    addDonnan = Calculated['Donnan']
    isign_2D = Calculated['TwoD']
    VTP = P.CathodeTankVolume
    VTN = P.AnodeTankVolume
    VE = Calculated['ElectrodeVolume']
    eps_pos = Calculated['Porosity'][0]
    eps_neg = Calculated['Porosity'][1]
    Over_potential_ohmic = Calculated['OhmicLossCharge']
    beta_pos = P.CathodeMassTransportCoefficient
    beta_neg = P.AnodeMassTransportCoefficient
    I = P.Current * abs(Status)
    Ae = Lh * Lw
    order = np.array([3, 2, 0, 1])
    
    if tio == t_start: isign_2D = 0                                            # Using 0D model for the first time step.
    Ick_pos = Calculated['Ick'][0]
    Ick_neg = Calculated['Ick'][1]
    dc_pos = Calculated['Dc'][0] * isign_2D / beta_pos
    dc_neg = Calculated['Dc'][1] * isign_2D / beta_neg
    
    # Calculate cross-over mass exchange rate.
    CkbHW = Status * np.array(Calculated['Cka'])[order] * isign_2D
    ixa = np.array(Chemical['Active'])[order]
    cc = np.array(Chemical['Coefficient'])[ixa]      
    
    # Update concentrations at outlet for tank and electrode.
    dti = tin - tio
    if np.all(Calculated['ElectrodeTankRatio']<1):
        if P.SolverType.lower() == 'odeexplicit':
            K = np.zeros((len(CIO), 4),dtype = float)
            K[:, 0] = dti * cross_over_rhs(M, CIO, S)
            K[:, 1] = dti * cross_over_rhs(M, CIO + 1/2 * K[:, 0], S)
            K[:, 2] = dti * cross_over_rhs(M, CIO + 1/2 * K[:, 1], S)
            K[:, 3] = dti * cross_over_rhs(M, CIO + K[:, 2], S)
            if P.Solver.lower() == 'rk4':
                CIN = CIO + 1/6 * K @ np.array([1, 2, 2, 1])
            else:
                CIN = CIO + K[:, 0]
        else:
            CIN = np.copy(CIO)
            if iszinc:
                idx = [2,3,6,7]
                M4 = np.identity(4,dtype = float) - M[idx,:][:,idx]*dti
                CIN[idx] = np.linalg.solve(M4, CIO[idx] + dti*S[idx])
            else:
                M8 = np.identity(8,dtype = float) - M*dti
                CIN = np.linalg.solve(M8, CIO + dti*S)
    else:
        warnings.warn('Electrolyte tank volume ratio is larger than ' + str(1*100) + '%')
        CIN = np.zeros(len(CIO),dtype=float)
        CIN[0:4] = CT0[0:4] + (tin - CT0[8]) * ST
        CIN[4:8] = CIN[0:4] + Calculated['Ckd'][order] * isign_2D
        
    # Update concentration at the outlet for active species (4).
    CIC = CIN[4:] - Calculated['Ckc'][order] * isign_2D *Status
    CIC = np.maximum(CIC, [dc_neg, dc_neg, dc_pos, dc_pos])
    
    # Set a minimum concentration and maintain a mass conservation. (Notes:
    #if any(CIN < VSMALL) and any(CIN != 0.0):
    # if any(CIN < VSMALL):
    #     CIN = np.maximum(CIN, VSMALL)
    
    # Another way to changing very small value.
    CIN_sum = CIN[[0,2,4,6]] + CIN[[1,3,5,7]]
    CIN_sum = CIN_sum[[0,0,1,1,2,2,3,3]]
    ida = np.array([[0,2,4,6],[1,3,5,7]]).transpose()
    nida = 4
    if np.any(CIN < VSMALL):
        #for k in range(ida.shape[0]):
        for k in range(nida):
            idx = ida[k,:]
            tfk = CIN[idx] < VSMALL
            if np.any(tfk):
                CIN[idx[tfk]] = VSMALL
                CIN[idx[~tfk]] = CIN_sum[idx[~tfk]] - CIN[idx[tfk]]
    
    # Update SOC and non-dimensional coefficients.
    C23_i = CIN[0] + CIN[1]/cc[1]*cc[0]
    C45_i = CIN[3] + CIN[2]/cc[2]*cc[3]
    #C23_i = np.sum(CIN[0:2])
    #C45_i = np.sum(CIN[2:4])
    if P.SOCDefineType.lower() == 'mean':
        SOCN = np.mean([CIN[0] / C23_i, CIN[3] / C45_i])
    elif P.SOCDefineType.lower() == 'maximum':
        SOCN = np.max([CIN[0] / C23_i, CIN[3] / C45_i])
    elif P.SOCDefineType.lower() == 'minimum':
        SOCN = np.min([CIN[0] / C23_i, CIN[3] / C45_i])
    elif P.SOCDefineType.lower() == 'anode':
        SOCN = CIN[0] / C23_i
    elif P.SOCDefineType.lower() == 'cathode':
        SOCN = CIN[3] / C45_i
    else:
        raise 'Invalid SOC define type.'
        
    # Calculate concentrations and potentials for negative electrode.
    if abs(alpha_neg - 0.5) <= 1e-4:
        tmp_neg_eta = Ick_neg / 2 / np.sqrt(CIN[4] * CIN[5])
        Act_potential_neg_i = -R * T / alpha_neg / (NE * F) * np.log(tmp_neg_eta + np.sqrt(1 + tmp_neg_eta**2)) * Status
    else:
        tmp_neg_eta = Ick_neg / (CIN[4]**alpha_neg * CIN[5]**(1 - alpha_neg))
        Act_potential_neg_i = -R * T / (NE * F) * BV(alpha_neg, tmp_neg_eta) * Status
    
    if Status == 1:
        if CIC[1] <= dc_neg:
            CIC[1] = dc_neg + VSMALL
        Con_potential_neg_i = -R * T / alpha_neg / (NE * F) * np.log(1 / (1 - dc_neg / CIC[1]))
    else:
        if CIC[0] <= dc_neg:
            CIC[0] = dc_neg + VSMALL
        Con_potential_neg_i = R * T / alpha_neg / (NE * F) * np.log(1 / (1 - dc_neg / CIC[0]))

    Equil_potential_neg_i = E01 + R * T / (NE * F) * np.log(CIN[5]**cc[1] / CIN[4]**cc[0])
    
    # Calculate concentrations and potentials for positive electrode.
    if abs(alpha_pos - 0.5) <= 1e-4:
        tmp_pos_eta = Ick_pos / 2 / np.sqrt(CIN[7] * CIN[6])
        Act_potential_pos_i = R * T / alpha_pos / (NP * F) * np.log(tmp_pos_eta + np.sqrt(1 + tmp_pos_eta**2)) * Status
    else:
        tmp_pos_eta = Ick_pos / (CIN[7]**alpha_pos * CIN[6]**(1 - alpha_pos))
        Act_potential_pos_i = R * T / (NP * F) * BV(alpha_pos, tmp_pos_eta) * Status
    
    if Status == 1:
        if CIC[2] <= dc_pos:
            CIC[2] = dc_pos + VSMALL
        Con_potential_pos_i = -R * T / alpha_pos / (NP * F) * np.log(1 - dc_pos / CIC[2])
    else:
        if CIC[3] <= dc_pos:
            CIC[3] = dc_pos + VSMALL
        Con_potential_pos_i = R * T / alpha_pos / (NP * F) * np.log(1 - dc_pos / CIC[3])
    
    # Updating equilibrium poential based on battery type.
    if iszinc: 
        Equil_potential_pos_i = E02 + R * T / (NP * F) * np.log(CIN[7]**cc[3] / CIN[6]**2)
    else:
        Equil_potential_pos_i = E02 + R * T / (NP * F) * np.log(CIN[7]**cc[3] / CIN[6]**cc[2])

    # Calculating concentrations for non-active species.
    # For vanadium:  CHWO is an 8x1 array: row 1-4 is the concentrations for
    # tank for proton (P), water (P), proton (N), and water (N); row 5-8 is
    # the concentrations for electrode with the same order.
    # For DHPS: CHWO is also 8x1: row 1-4 is the concentrations for tank for
    # hydroxide (P), water (P), hydroxide (N), and water (N). Similar for 5-8.
    sv = P.MembraneSolventDirection
    
    if P.ProtonMode.lower() == 'unsteady':
        nv = len(CHWO) // 2
        #deleted W = np.array([P.CathodePumpRate, P.CathodePumpRate, P.AnodePumpRate, P.AnodePumpRate])
        W = np.array([W_pos, W_pos, W_neg, W_neg])
        # deleted VT = np.array([P.CathodeTankVolume, P.CathodeTankVolume, P.AnodeTankVolume, P.AnodeTankVolume])
        VT = np.array([VTP, VTP, VTN, VTN])
        # deleted  MM = np.zeros((len(CHWO), len(CHWO)))
        # deleted MM[:nv, :nv] = np.diag(-W / VT)
        # deleted MM[:nv, nv:] = np.diag(W / VT)
        # deleted MM[nv:, nv:] = np.diag(W / (P.ElectrodePorosity * Calculated['ElectrodeVolume']))
        eps_pn = np.array([eps_pos, eps_pos, eps_neg, eps_neg])
        MM = np.block([[np.diag(-W / VT), np.diag(W / VT)],
                       [np.diag(W / eps_pn  / VE), -np.diag(W / eps_pn / VE)]])
        S1 = np.zeros(nv)
        
        if P.ChemicalType.lower() == 'inorganic/inorganic':
            S2a = np.array([2, -1, 0, 0]) # Normal reaction for non-active species.
            # deleted S2m = np.array([-P.MembraneIonDragCoefficient, -P.MembraneSolvantDragCoefficient, P.MembraneIonDragCoefficient, P.MembraneSolvantDragCoefficient]) * sv
            S2m = np.array([-nh, -nd, nh, nd]) * sv  # Transport across membrane of non-active species.
            S2c = np.array([-2 * uc[0] * (CIN[4] - CkbHW[0]), 
                            uc[0] * (CIN[4] - CkbHW[0]), 
                            -2 * uc[2] * (CIN[6] - CkbHW[2]) - 4 * uc[3] * (CIN[7] - CkbHW[3]), 
                            uc[2] * (CIN[6] - CkbHW[2]) + 2 * uc[3] * (CIN[7] - CkbHW[3])]) * (eps_pn ** k * Ae / eps_pn / VE)
        elif P.ChemicalType.lower() == 'inorganic/organic':
            S2a = np.array([0, 0, 2, -2])
            S2m = np.array([-nh, -nd, nh, nd]) * sv * 0
            S2c = np.array([0, 0, 0, 0])
        
        sn = np.array([NP, NP, NE, NE])
        S2 = (S2a + S2m) * I / F / sn / eps_pn / VE * Status + S2c
        SS = np.concatenate((S1,S2),axis=0)
        
        # Solving ODE for water/proton concentrations.
        dti = tin - tio
        if P.SolverType.lower() == 'odeexplicit':
            K = np.zeros((len(CHWO), 4),dtype=float)
            K[:, 0] = dti * cross_over_rhs(MM, CHWO, SS)
            K[:, 1] = dti * cross_over_rhs(MM, CHWO + 1/2 * K[:, 0], SS)
            K[:, 2] = dti * cross_over_rhs(MM, CHWO + 1/2 * K[:, 1], SS)
            K[:, 3] = dti * cross_over_rhs(MM, CHWO + K[:, 2], SS)
            if P.Solver.lower() == 'rk4':
                CHWN = CHWO + 1/6 * K @ np.array([1, 2, 2, 1])
            else:
                CHWN = CHWO + K[:, 0]
        else:
            M8HW = np.identity(8,dtype = float) - MM*dti
            CHWN = np.linalg.solve(M8HW, CHWO + dti*SS) 
    else:
        CHWN = CHWO
        
    # Set a minimum concentration and maintain a mass conservation. (Notes:
    if np.any(CHWN < VSMALL):
        CHWN = np.maximum(CHWN, VSMALL)
    
    # Computing potentials from non-active species.
    if P.ChemicalType.lower() == 'inorganic/inorganic':
        OCV_potential_donnan_i = addDonnan * R * T / (NP * F) * np.log(CHWN[4] / CHWN[6])
        OCV_potential_proton_i = R * T / (NP * F) * np.log(CHWN[4]**2 / CHWN[5]) - 0
        
        # For Zinc battery, do not consider charge carrier potential for now.
        #if any([name in Chemical['Name'] for name in ['Zinc', 'Iodide']]):
        if iszinc:
            OCV_potential_donnan_i = 0
            OCV_potential_proton_i = 0
    elif P.ChemicalType.lower() == 'inorganic/organic':
        OCV_potential_donnan_i = 0
        OCV_potential_proton_i = -R * T / (NE * F) * np.log(CHWN[6]**2 / CHWN[7]**2)
    else:
        OCV_potential_donnan_i = 0
        OCV_potential_proton_i = 0

    Ohmic_potential_i = Over_potential_ohmic * Status

    # Ensemble total cell voltage and potential components.
    Nz = 1
    VN = np.zeros((Nz, 13))
    VN[:, 0] = Equil_potential_pos_i - Equil_potential_neg_i + \
               (Act_potential_pos_i - Act_potential_neg_i) + \
               (Con_potential_pos_i - Con_potential_neg_i) + \
               OCV_potential_donnan_i + OCV_potential_proton_i + \
               Ohmic_potential_i
    VN[:, 1] = Equil_potential_pos_i - Equil_potential_neg_i
    VN[:, 2] = Act_potential_pos_i - Act_potential_neg_i
    VN[:, 3] = Con_potential_pos_i - Con_potential_neg_i
    VN[:, 4:7] = np.tile(np.array([Ohmic_potential_i, OCV_potential_proton_i, OCV_potential_donnan_i]), (Nz, 1))
    VN[:, 7] = Equil_potential_pos_i
    VN[:, 8] = Equil_potential_neg_i
    VN[:, 9] = Act_potential_pos_i
    VN[:, 10] = Act_potential_neg_i
    VN[:, 11] = Con_potential_pos_i
    VN[:, 12] = Con_potential_neg_i

    # Ensemble concentrations.
    CN = np.zeros((Nz, 16))
    CN[:, 0:8] = np.concatenate([CIC, CIN[4:8]])
    CN[:, 8:16] = np.tile(CHWN, (Nz, 1))
    return CIN,VN,CN,SOCN,CHWN,M,S 

#%%
# This function updates initial concentrations and porosity based on given SOC#
def UpdateInitialConcentration(PP0):
    PP = copy.deepcopy(PP0)
    
    # Updating concentrations based on given SOC for Vanadium.
    SOC = PP.SOC
    Status = PP.InitialStatus
    
    Chemical = PP.Chemical
    iszinc = 1 if 'Zinc' in Chemical['Name'] or\
        'Iodide' in Chemical['Name'] else 0
    
    order = np.array([3, 2, 0, 1])
    ixa = np.array(Chemical['Active'])[order]
    cc = np.array(Chemical['Coefficient'],dtype=float)[ixa]  
    C40 = PP.CathodeReductant1InitialConcentrationInCathode
    C50 = PP.CathodeOxidant1InitialConcentrationInCathode
    C20 = PP.AnodeReductant1InitialConcentrationInAnode
    C30 = PP.AnodeOxidant1InitialConcentrationInAnode
    C450 = C50 + C40/cc[2]*cc[3]
    C230 = C20 + C30/cc[1]*cc[0]
    C5i = SOC*C450
    C4i = (1-SOC)*C450*cc[2]/cc[3]
    C2i = SOC*C230
    C3i = (1-SOC)*C230*cc[1]/cc[0]
    
    # Updating concentrations for cathode.
    PP.CathodeReductant1InitialConcentrationInCathode = C4i
    PP.CathodeOxidant1InitialConcentrationInCathode = C5i
    
    # Updating concentrations for anode.
    if not iszinc:
        PP.AnodeReductant1InitialConcentrationInAnode = C2i
        PP.AnodeOxidant1InitialConcentrationInAnode = C3i
    else:
        eps0 = PP.AnodeElectrodePorosity                                       # Assuming eps is at SOC 0.
        VTP = PP.CathodeTankVolume
        CZN = PP.AnodeReductant1InitialConcentrationInAnode
        VE = PP.Calculated['ElectrodeVolume']
        if Status == 1:
            epsn = eps0 - (SOC-0)*VTP*C450/VE/CZN
        elif Status == -1:
            epsn = eps0 - (0.9-0)*VTP*C450/VE/CZN + (0.9-SOC)*VTP*C450/VE/CZN  # Assuming SOC charge from 0 to 1.
        else:
            raise 'Wrong status.'
        PP.AnodeElectrodePorosity = epsn

    return PP

#%%
# This function loads epxeriment data further usage.                          #
def LoadExperiment(PP):
    formatSpec1 = 'Loading experiment data from %s.'
    formatSpec2 = 'Keyword: %s not found, skipping data loading.'
    Display = PP.Display.lower() == 'yes'                                      # If output information on screen (integer).
    if Display: print(formatSpec1 %(PP.Experiment))
    
    Exp = pd.read_excel(PP.Experiment,1)                                       # Loading experiment data sheet number 2.
    keys = list(Exp.keys())
    user_keys = ['Test_Time(s)','Voltage(V)','Current(A)','Cycle_Index']       # This is required key words in experiment.
    tfx,locc = ismember(user_keys,keys)
    F = []
    if not np.all(tfx):
        idx = np.where(~tfx)[0]
        user_keys_no = np.array(user_keys)[idx]
        user_keys_miss = ', '.join(user_keys_no)
        if Display: print(formatSpec2 %(user_keys_miss))
    else:
        F = Exp.iloc[:,locc].to_numpy()
        Exp = []                                                               # Clear data to save memory.
        ncs = PP.ExperimentStartCycle
        nce = PP.ExperimentStartCycle - 1 + PP.CycleNumber                                                
        #nce = PP.ExperimentEndCycle
        F = F[F[:,3]>=ncs,:]
        F = F[F[:,3]<=nce,:]
        ids = np.where(F[:,2] >0.01)[0][0]                                     # The first point with non-zero current.
        ide = np.where(F[:,2] <-0.01)[0][-1]
        F = F[ids:ide+1,:]
        F[:,0] = F[:,0] - F[0,0]
        F[:,3] = F[:,3] - F[0,3] + 1

    return F
