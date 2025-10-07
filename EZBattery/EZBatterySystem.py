#%%######################### Descriptions #####################################
# This file includes functions for calculating power loss, shunt loss, total  #
# loss, search current for a given SOC and power, and search current for      #
# mulitple SOC and power. The code is initially developed by Jie Bao @PNNL    #
# (jie.bao@pnnl.gov), and later modified by Yunxiang Chen @PNNL               #
# (yunxiang.chen@pnnl.gov) to integrate the initial code to cell level redox  #
# flow batteries. Alasdair Crawford @PNNL provides initial idea and review of #
# the shunt and pump power functions. Please contact yunxiang.chen@pnnl.gov   #
# or jie.bao@pnnl.gov for feedbacks or questions.Potential users are permitted# 
# to use for non-commercial purposes without modifying any part of the code.  #
# Please contact the author to obtain written permission for commercial or    #
# modifying applications.                                                     # 

# Copyright (c) 2024 Yunxiang Chen, Jie Bao
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

#%% Python functions.
import math, copy, time
import numpy as np
from .EZBatteryCell import RFB
from .EZBatteryUtilities import UpdateInitialConcentration
from .mathm import numCount

#%% Functions for cell system integration.
def shuntPowerLoss(PP0,try_v,I,Rce):
    # New way for parameter transfer.
    PP = copy.deepcopy(PP0)
    electrolyteCond = PP.CathodeElectrolyteConductivity
    manifoldRadius = PP.ManifoldRadius
    manifoldLength = PP.ManifoldLength
    flowChannelThickness = PP.FlowChannelThickness
    flowChannelWidth = PP.FlowChannelWidth
    flowChannelLength = PP.FlowChannelLength
    n = PP.CellNumber
    
    # Further calculation.
    manifoldArea = math.pi * manifoldRadius**2
    flowChannelArea = flowChannelThickness*flowChannelWidth
    Rm = manifoldLength/manifoldArea/electrolyteCond
    Rch = flowChannelLength/flowChannelArea/electrolyteCond

    flowChannels = [-(n - 1) / 2 + i for i in range(n)]                        #indexing for flow channels
    manifolds = [-(n-2) / 2 + i for i in range(n-1)]                           #indexing for manifolds
    
    #Ik=(VNom+I*Rce)/(4*Rce+Rm) #characteristic current, amps
    Ik=(try_v)/(4*Rce+Rm)                                                      # characteristic current, amps
    r=math.sqrt((Rch)/(4*Rce+Rm))
    
    powerLoss = 0                                                              # W, this gets each loss term added onto
    for i in flowChannels:
        channelCurrent=Ik/r*(math.sinh(i/r)/math.cosh(n/2/r))                  #A, current thru each flow channel
        powerLoss += 4*Rch*channelCurrent**2
    for i in manifolds:
        manifoldCurrent=Ik*(1-math.cosh(i/r)/math.cosh(n/2/r))                 #A, current thru each manifold
        powerLoss += 4*Rm*manifoldCurrent**2
    return powerLoss, PP

#%%
def pumpPowerLoss(PP0):
    PP = copy.deepcopy(PP0)
    # Internal parameter transfer.
    pumpEff = PP.PumpEfficiency
    cellHeight = PP.ElectrodeHeight
    cellWidth = PP.ElectrodeWidth
    cellThickness = PP.ElectrodeThickness
    channelDepth = PP.ChannelDepth
    channelWidth = PP.ChannelWidth
    nChannel = PP.ChannelNumber
    K = PP.Calculated['ElectrodePermeability'][1]*PP.SystemEletrodePermeabilityAdjust                              # Using Cozney-Karman equation.
    visc = PP.CathodeElectrolyteViscosity
    interdigitationWidth = PP.InterdigitationWidth
    Q = PP.CathodePumpRate
    n = PP.CellNumber
    interdigitation = PP.CellType.lower() == 'ID'                              # Checking if flow channel is Interdigitated.
    
    # Further calculation.
    Dh = 2 * channelDepth * channelWidth / (channelDepth + channelWidth)       # Hydraulic diameter (m).
    channelArea = channelDepth * channelWidth                                  # Channel area (m2).
    if interdigitation:
        segments = cellWidth / interdigitationWidth                            #1 (yes this is continuous instead of discrete, makes optimisation easier)
        Pdrop= (Q/(cellThickness*cellHeight))*visc*cellWidth/K/segments+\
            32*cellWidth*Q*visc/(channelArea*Dh*Dh*segments)                   #Pa, first term is cell drop, second term is channel drop
    else:
        Pdrop=(Q/(cellThickness*cellWidth))*visc*cellHeight/K+\
            2*32*cellWidth*Q*visc/(channelArea*Dh*Dh*nChannel)                 #Pa, first term is cell drop, second term is channel drop
        
    powerLoss= 2 * Pdrop*Q*n/pumpEff
    return powerLoss, PP

#%%
def systemPower(PP0):
    PP = copy.deepcopy(PP0)
    
    #timestart=time.time()
    d_current = 0.1
    n= PP.CellNumber
    Status = PP.InitialStatus
    try_current = PP.Current
    
    # Updating initial concentrations and porosity based on given paraemters.
    PP = UpdateInitialConcentration(PP)
    
    # Calculating cell potential at a given current.
    PP.Current = try_current
    Re = RFB(PP)

    # Calculating cell resistance.
    #PP1 = copy.deepcopy(PP)
    #PP1.Current = try_current - d_current
    PP.Current = try_current - d_current
    Re1 = RFB(PP)
    #PP2 = copy.deepcopy(PP)
    #PP2.Current = try_current + d_current
    PP.Current = try_current + d_current
    Re2 = RFB(PP)
    PP.Current = try_current

    # Calculating system power if Re is not empty.
    tt_p,try_v,shunt_powerloss,pump_powerloss = (-1e7,-9999,-9999,-9999)
    #print('len()',len(Re['Potentials']['Ec_V']),len(Re1['Potentials']['Ec_V']),len(Re2['Potentials']['Ec_V']))
    if len(Re)>0 and len(Re1)>0 and len(Re2)>0:
        try_v= Re['Potentials']['Ec_V'].iloc[-1]
        Rce = abs((Re2['Potentials']['Ec_V'].iloc[-1] - \
                   Re1['Potentials']['Ec_V'].iloc[-1])/(2*d_current))

        # Computing pump loss and shut loss.
        shunt_powerloss,_ = shuntPowerLoss(PP,try_v,try_current,Rce)
        pump_powerloss,_ = pumpPowerLoss(PP)
    
        if Status == -1: tt_p = try_v*try_current * n - shunt_powerloss-pump_powerloss -PP.SystemAuxPower # approximated extra auxiliary power
        if Status == 1: tt_p = try_v*try_current *n + shunt_powerloss+pump_powerloss   +PP.SystemAuxPower # approximated extra auxiliary power
        
    #timeend=time.time()
    #print("total computation time (s)", timeend-timestart)
    return tt_p,try_v,shunt_powerloss,pump_powerloss, Re, PP

def searchCurrent(PP0):
    fmt1 = 'Step %d: current %3.2fA, voltage %3.2fV, power %3.2fW,' + \
        ' pump %3.2fW, shunt %3.2fW, error %.1e, negative.'
    fmt2 = 'Step %d: current %3.2fA, voltage %3.2fV, power %3.2fW,' + \
        ' pump %3.2fW, shunt %3.2fW, error %.1e, steady.'
    fmt3 = 'Step %d: current %3.2fA, voltage %3.2fV, power %3.2fW,' + \
        ' pump %3.2fW, shunt %3.2fW, error %.1e, success.'
    fmt4 = 'Step %d: current %3.2fA, voltage %3.2fV, power %3.2fW,' + \
        ' pump %3.2fW, shunt %3.2fW, error %.1e'
    
    PP = copy.deepcopy(PP0)
    power = PP.Power
    #PP.InitialStatus = np.sign(power)
    current_low = PP.SearchCurrentMinimum
    current_high = PP.SearchCurrentMaximum
    max_iteration = PP.MaximumSearchStep
    tolerance = PP.SearchCurrentConvergeTolerance
    (low, high) = (current_low, current_high)
    DispCurrent = PP.DisplayCurrentSearch.lower() == 'yes'
    DispIteration = PP.DisplayCurrentSearchIteration.lower() == 'yes'
    
    nv = 7
    nmin = 3
    hists = np.zeros((0,nv))
    Re = []
    for i in range(max_iteration):
        current_mid = (low + high)/2.0
        PP.Current = current_mid
        power_mid,v,spl,ppl, Re, PPC  = systemPower(PP)
        err = abs((power_mid - power)/power)
        x0 = np.array([i+1,current_mid, v, power_mid, ppl, spl, err]).reshape((1,nv))
        hists = np.append(hists,x0, axis=0)
        
        if v<0 or power_mid <-1e6:
            if DispCurrent: print(fmt1 %(i+1,current_mid, v, power_mid, ppl, spl, err))
            (current_mid,v,spl,ppl) = (np.nan, np.nan, np.nan, np.nan)
            break
        else:
            if abs(power_mid - power) <= tolerance*abs(power):
                if DispCurrent: print(fmt3 %(i+1,current_mid, v, power_mid, ppl, spl, err))
                break
            else:
                [st,na,val] = numCount(hists[:,1],tol=1e-3)
                if np.any(na>=nmin): 
                    if DispCurrent: print(fmt2 %(i+1,current_mid, v, power_mid, ppl, spl, err))
                    (current_mid,v,spl,ppl) = (np.nan, np.nan, np.nan, np.nan)
                    break
                else:
                    if DispCurrent and DispIteration: 
                        print(fmt4 %(i+1,current_mid, v, power_mid, ppl, spl, err))
                    if power_mid < power:
                        low = current_mid
                    else:
                        high = current_mid
                    
    return current_mid,v,spl,ppl,Re,PPC

#%% Search current, voltage and power losses at multiple SOC and power.       #
def searchSystem(PP0):
    PP = copy.deepcopy(PP0)
    timestart=time.time()
    fmt0 = '\nStarting to search current and voltage for a battery system: %s.'
    fmt1 = '%s search %d: Power = %3.2f W and SOC = %3.2f (%d/%d, %3.2f%%)'
    # Define target SOC and Power series.
    nPower = PP.PowerNumber
    nSOC = PP.SOCNumber
    Power = np.linspace(PP.PowerMinimum,PP.PowerMaximum,PP.PowerNumber)
    SOC = np.linspace(PP.SOCStart, PP.SOCEnd,PP.SOCNumber)
    ntotal = nPower*nSOC
    DispSystem = PP.DisplaySystemSearch.lower() == 'yes'
    Status = 'Charging' if PP.InitialStatus == 1 else 'Discharging' 
    print(fmt0 %(Status))

    I = np.full((nPower,nSOC),np.nan)
    V = np.full((nPower,nSOC),np.nan)
    SOCs = np.full((nPower,nSOC),np.nan)
    Powers = np.full((nPower,nSOC),np.nan)
    PowersShunt = np.full((nPower,nSOC),np.nan)
    PowersPump = np.full((nPower,nSOC),np.nan)
    Porosity = np.full((nPower,nSOC),np.nan)
    SpecificArea = np.full((nPower,nSOC),np.nan)
    
    # Search results.
    count = 1
    for i in range(nPower):   
        PP.Power = Power[i]
        for j in range(nSOC):
            if i>0 and j == 0: print('')
            PP.SOC = SOC[j]
            if DispSystem: print(fmt1 %(Status, count, Power[i], SOC[j], count, ntotal, count/ntotal*100))
            current_mid,v,spl,ppl,re2,ppc = searchCurrent(PP)
            I[i,j] = current_mid
            V[i,j] = v
            SOCs[i,j] = SOC[j]
            Powers[i,j] = Power[i]
            PowersShunt[i,j] = spl
            PowersPump[i,j] = ppl
            Porosity[i,j] = ppc.Calculated['Porosity'][1]
            SpecificArea[i,j] = ppc.Calculated['SpecificArea'][1]
            count = count + 1
            
    Re = {
    'SOC': SOCs,
    'Power_W': Powers,
    'Current_A': I,
    'Voltage_V': V,
    'ShuntPowerLoss_W': PowersShunt,
    'PumpPowerLoss_W': PowersPump,
    'Porosity': Porosity,
    'SpecificArea': SpecificArea
    }
    
    timeend=time.time()
    print("Computation done, total time %3.2f seconds" % (timeend-timestart))
    return Re