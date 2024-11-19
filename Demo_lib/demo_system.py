import os
import sys
import copy
import numpy as np
import pandas as pd
import argparse
import math
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.ticker import FormatStrFormatter
from EZBattery import BatteryParameter, EZBatteryCell, RFB, searchCurrent, searchSystem, systemPower, LoadExperiment, Calibration, CellLossFunction
#%%
def one_year_RPT_run(nyear,PPS):
    # WorkDir = ''
    # SDPFName = 'Data/SDPF_Vanadium_20240718_V5_large_system_v2.xlsx'
    # SDPFPath = SDPFName
    # PP0 = BatteryParameter(SDPFPath)
    PP1 = copy.deepcopy(PPS)
    PP1.BatteryMode = 'power'
    # PP1.StartTime = 0
    # PP1.EndTime = 1
    # PP1.TimeStep = 1
    # PP1.RefineTimePoints = 1
    PP1.Display = 'no'
    PP1.DisplayCurrentSearch = 'no'
    PP1.DisplayCurrentSearchIteration = 'no'
    PP1.SearchCurrentConvergeTolerance = 5e-5
    #PP1.CellNumber = 50
    PP1.SearchCurrentMinimum = 0.001
    PP1.SearchCurrentMaximum = 200
    soc_lower_bond = 0.15
    soc_upper_bond = 0.85
    soc_tol = 2.5e-5
    faraday_c = 96_485.3321
    ini_anode_conc = PPS.AnodeOxidant1InitialConcentrationInAnode * 0.997**nyear
    ini_cathode_conc = PPS.CathodeReductant1InitialConcentrationInCathode * 1.003**nyear
    PP1.MembraneElectronicConductivity = PPS.MembraneElectronicConductivity * 0.992**nyear
    PP1.CathodeReactionRateConstant = PPS.CathodeReactionRateConstant * 0.992**nyear
    PP1.AnodeReactionRateConstant = PPS.AnodeReactionRateConstant * 0.992**nyear

    syn_power = np.zeros(20)
    syn_power[0:10] = -4330
    syn_power[10:20] = 4330
    dt_syn_time = 1
    syn_time = np.arange(0, 20, dt_syn_time)
    dt = 0.01
    syn_socs = [0.15]
    syn_vs = []
    syn_shunts = []
    syn_pumps = []
    syn_currents = []
    syn_hourss = [0]
    syn_act_power = []
    ii = 0
    re_run = False
    while syn_hourss[-1] < syn_time[-1]:
        ii_old = ii
        ii = math.floor(syn_hourss[-1] / dt_syn_time)
        if syn_power[ii] > 0:
            PP1.InitialStatus = -1 # discharge
        else:
            PP1.InitialStatus = 1 # charge
            
        if not re_run:
            dt = 0.25
        else:
            dt = dt / 4
            #if dt<1e-3: print('reduce dt to ',dt,' for matching the SOC ',soc_lower_bond,' or ',soc_upper_bond, 'with tol ',soc_tol)

        PP1.Power = abs(syn_power[int(syn_hourss[-1])])
        #if ii != ii_old:
            #print("{:<10.3f}(hours) {:<5} {:<10.3f}(W) {:<10.5f}(SOC) {:<10.3f}(dt s)".format(syn_hourss[-1], ii, syn_power[ii], syn_socs[-1], dt), flush=True)
            #print("Hours {:<10.3f},  Power {:<10.3f}(W), SOC {:<10.5f}, dt {:<6.3f}(hrs)".format(syn_hourss[-1], syn_power[ii], syn_socs[-1], dt), flush=True)
      

        PP1.SOC = syn_socs[-1]
        C450 = ini_cathode_conc
        C4i = (1 - PP1.SOC) * C450
        C5i = PP1.SOC * C450
        PP1.CathodeReductant1InitialConcentrationInCathode = C4i
        PP1.CathodeOxidant1InitialConcentrationInCathode = C5i
        C230 = ini_anode_conc
        C3i = (1 - PP1.SOC) * C230
        C2i = PP1.SOC * C230
        PP1.AnodeOxidant1InitialConcentrationInAnode = C3i
        PP1.AnodeReductant1InitialConcentrationInAnode = C2i
        if (syn_socs[-1] <= 0.15 and PP1.InitialStatus == -1) or (syn_socs[-1] >= 0.85 and PP1.InitialStatus == 1):
            idc = 0
            vdc = 0
            psldc = 0
            ppldc = 0
        else:
            idc, vdc, psldc, ppldc, Re1dc, ppdc = searchCurrent(PP1)
        if PP1.InitialStatus == -1 : # discharge
            new_soc = (syn_socs[-1]*ini_anode_conc*PP1.AnodeTankVolume*PP1.CellNumber - dt*3600*idc/faraday_c/PP1.AnodeElectronNumber*PP1.CellNumber) / PP1.AnodeTankVolume / PP1.CellNumber / ini_anode_conc
            if new_soc - soc_lower_bond < -soc_tol:
                re_run = True
            else:
                syn_socs.append(new_soc)
                re_run = False
        else: # charge
            new_soc = (syn_socs[-1]*ini_anode_conc*PP1.AnodeTankVolume*PP1.CellNumber + dt*3600*idc/faraday_c/PP1.AnodeElectronNumber*PP1.CellNumber) / PP1.AnodeTankVolume / PP1.CellNumber / ini_anode_conc
            if new_soc - soc_upper_bond > soc_tol:
                re_run = True
            else: 
                syn_socs.append(new_soc)
                re_run = False
        if not re_run:
            syn_vs.append(vdc)
            syn_shunts.append(psldc)
            syn_pumps.append(ppldc)
            if PP1.InitialStatus == -1: 
                syn_currents.append(idc)
            else:
                syn_currents.append(-idc)
            if len(syn_vs)<=1:
                syn_hourss.append(syn_hourss[-1] + dt)
            else:
                if syn_vs[-2]==0 and syn_vs[-1]!=0 :
                    syn_hourss[-1]=syn_time[ii]
                    #syn_hourss.append(syn_time[ii])
                    syn_hourss.append(syn_hourss[-1] + dt )
                    #syn_hourss[-2]=syn_time[ii]
                    #print('reset system time to ',syn_hourss[-3],syn_hourss[-2],syn_hourss[-1],syn_socs[-3],syn_socs[-2],syn_socs[-1],syn_vs[-3],syn_vs[-2],syn_vs[-1])
                else:
                    syn_hourss.append(syn_hourss[-1] + dt)
            if idc == 0 and vdc == 0:
                syn_act_power.append(0)
            else:
                syn_act_power.append(syn_vs[-1] * syn_currents[-1] * PP1.CellNumber - syn_shunts[-1] - syn_pumps[-1] - PP1.SystemAuxPower) 
            
    syn_hoursss = np.copy(syn_hourss[:-1])
    syn_currents = np.asarray(syn_currents)
    charge_time = syn_hoursss[syn_currents < 0]
    discharge_time = syn_hoursss[syn_currents > 0]
    total_charge_time = charge_time[-1] - charge_time[0]
    total_discharge_time = discharge_time[-1] - discharge_time[0]
    print('year ', nyear, flush=True)
    print('total charge time :', total_charge_time, " (h)", flush=True)
    print('total discharge time :', total_discharge_time, " (h)", flush=True)
    print('RTE :', total_discharge_time / total_charge_time, " (-)", flush=True)
    print('capacity :', total_discharge_time * abs(syn_power[-1]) / 1e3, " (KWh)", flush=True)
    
    # Collecting the results into a DataFrame
    results = {
        'year': nyear,
        'syn_socs': syn_socs,
        'syn_vs': syn_vs,
        'syn_shunts': syn_shunts,
        'syn_pumps': syn_pumps,
        'syn_currents': syn_currents.tolist(),
        'syn_hourss': syn_hourss,
        'syn_act_power': syn_act_power,
    }
    return results
#%%
def one_year_run(nnyear,PPS,data_year,start_time,result_file_base):
    result_file=result_file_base+'_year_'+str(nnyear)+'.csv'
    result_out=open(result_file,'w')
    data_power=data_year['Power (kW)'].to_numpy()
    nyear=nnyear-2025
    PP1 = copy.deepcopy(PPS)
    PP1.BatteryMode = 'power'
    PP1.StartTime = 0
    PP1.EndTime = 1
    PP1.TimeStep = 1
    PP1.RefineTimePoints = 1
    PP1.Display = 'no'
    PP1.DisplayCurrentSearch = 'no'
    PP1.DisplayCurrentSearchIteration = 'no'
    PP1.SearchCurrentConvergeTolerance = 5e-5
    PP1.SearchCurrentMinimum = 0.001
    PP1.SearchCurrentMaximum = 200
    soc_lower_bond = 0.15
    soc_upper_bond = 0.85
    soc_tol = 2.5e-5
    faraday_c = 96_485.3321
    ini_anode_conc = PPS.AnodeOxidant1InitialConcentrationInAnode * 0.997**nyear
    ini_cathode_conc = PPS.CathodeReductant1InitialConcentrationInCathode * 1.003**nyear
    ini_MembraneElectronicConductivity = PPS.MembraneElectronicConductivity * 0.992**nyear
    ini_CathodeReactionRateConstant = PPS.CathodeReactionRateConstant * 0.992**nyear
    ini_AnodeReactionRateConstant = PPS.AnodeReactionRateConstant * 0.992**nyear
    hr_drate1=0.997**(1/365/24)
    hr_drate2=0.992**(1/365/24)
    hr_drate3=1.003**(1/365/24)

    syn_power = np.copy(data_power)*1e3
    dt_syn_time = 1
    syn_time = np.arange(0, len(syn_power), dt_syn_time)
    dt = 0.25
    syn_socs = [0.85]
    syn_vs = []
    syn_shunts = []
    syn_pumps = []
    syn_currents = []
    syn_hourss = [0]
    syn_act_power = []
    ii = 0
    re_run = False
    ncyc = 0
    while syn_hourss[-1] < syn_time[-1]:
        ii_old = ii
        ii = math.floor(syn_hourss[-1] / dt_syn_time)
        if syn_power[ii] > 0:
            PP1.InitialStatus = -1 # discharge
        else:
            PP1.InitialStatus = 1 # charge
            
        if not re_run:
            dt = min(dt*2,0.25) #0.25
        else:
            dt = dt / 4
        PP1.Power = np.abs(syn_power[ii])
        PP1.SOC = syn_socs[-1]
        C450 = ini_cathode_conc*hr_drate3**(syn_hourss[-1])
        C4i = (1 - PP1.SOC) * C450
        C5i = PP1.SOC * C450
        PP1.CathodeReductant1InitialConcentrationInCathode = C4i
        PP1.CathodeOxidant1InitialConcentrationInCathode = C5i
        C230 = ini_anode_conc*hr_drate1**(syn_hourss[-1])
        C3i = (1 - PP1.SOC) * C230
        C2i = PP1.SOC * C230
        PP1.AnodeOxidant1InitialConcentrationInAnode = C3i
        PP1.AnodeReductant1InitialConcentrationInAnode = C2i
        PP1.MembraneElectronicConductivity = ini_MembraneElectronicConductivity*hr_drate2**(syn_hourss[-1])
        PP1.CathodeReactionRateConstant = ini_CathodeReactionRateConstant*hr_drate2**(syn_hourss[-1])
        PP1.AnodeReactionRateConstant = ini_AnodeReactionRateConstant*hr_drate2**(syn_hourss[-1])
        if (syn_socs[-1] <= 0.15 and PP1.InitialStatus == -1) or (syn_socs[-1] >= 0.85 and PP1.InitialStatus == 1 or PP1.Power==0):
            idc = 0
            vdc = 0
            psldc = 0
            ppldc = 0
        else:
            idc, vdc, psldc, ppldc, Re1dc, ppdc = searchCurrent(PP1)
        if PP1.InitialStatus == -1 : # discharge
            new_soc = (syn_socs[-1]*ini_anode_conc*PP1.AnodeTankVolume*PP1.CellNumber - dt*3600*idc/faraday_c/PP1.AnodeElectronNumber*PP1.CellNumber) / PP1.AnodeTankVolume / PP1.CellNumber / ini_anode_conc
            if new_soc - soc_lower_bond < -soc_tol:
                re_run = True
            else:
                syn_socs.append(new_soc)
                re_run = False
        else: # charge
            new_soc = (syn_socs[-1]*ini_anode_conc*PP1.AnodeTankVolume*PP1.CellNumber + dt*3600*idc/faraday_c/PP1.AnodeElectronNumber*PP1.CellNumber) / PP1.AnodeTankVolume / PP1.CellNumber / ini_anode_conc
            if new_soc - soc_upper_bond > soc_tol:
                re_run = True
            else: 
                syn_socs.append(new_soc)
                re_run = False
        if not re_run:
            syn_vs.append(vdc)
            syn_shunts.append(psldc)
            syn_pumps.append(ppldc)
            if PP1.InitialStatus == -1: 
                syn_currents.append(idc)
            else:
                syn_currents.append(-idc)
            if len(syn_vs)<=1:
                syn_hourss.append(syn_hourss[-1] + dt)
            else:
                if syn_vs[-2]==0 and syn_vs[-1]!=0 :
                    syn_hourss[-1]=syn_time[ii]
                    #syn_hourss.append(syn_time[ii])
                    syn_hourss.append(syn_hourss[-1] + dt)
                else:
                    syn_hourss.append(syn_hourss[-1] + dt)
            if idc == 0 and vdc == 0:
                syn_act_power.append(0)
            else:
                syn_act_power.append(syn_vs[-1] * syn_currents[-1] * PP1.CellNumber - syn_shunts[-1] - syn_pumps[-1] - 200) # 200w is the extra power loss
        real_time=start_time+pd.Timedelta(hours=syn_hourss[-1])
        if  len(syn_socs)>10:
            if syn_socs[-1]>syn_socs[-2] and abs(syn_socs[-2]-soc_lower_bond)<=soc_tol and abs(syn_socs[-1]-soc_lower_bond)>soc_tol and len(syn_socs)>10:
                ncyc=ncyc+1
            if syn_socs[-1]>syn_socs[-2] and abs(syn_socs[-2]-soc_lower_bond)>soc_tol and syn_socs[-2]<syn_socs[-3] and len(syn_socs)>10:
                ncyc=ncyc+1
        if math.floor(syn_hourss[-1])==math.ceil(syn_hourss[-1]) and int(syn_hourss[-1])%8==0:
            print(real_time, ",    Hours {:<10.3f}, Cycles {:<5},  Power {:<10.3f}(W), SOC {:<10.5f}, dt {:<6.3f}(hrs)".format(syn_hourss[-1], ncyc, syn_power[ii], syn_socs[-1], dt), flush=True)
        #print(real_time,',',syn_hourss[-1],',',syn_act_power[-1],',',syn_shunts[-1],',',syn_pumps[-1],',',syn_currents[-1],',',syn_vs[-1],',',syn_socs[-1],flush=True)
        print(real_time,',',syn_hourss[-1],',',ncyc,',',syn_act_power[-1],',',syn_shunts[-1],',',syn_pumps[-1],',',syn_currents[-1],',',syn_vs[-1],',',syn_socs[-1],file=result_out,flush=True)

    results = {
        'year': nyear,
        'syn_socs': syn_socs,
        'syn_vs': syn_vs,
        'syn_shunts': syn_shunts,
        'syn_pumps': syn_pumps,
        'syn_currents': syn_currents.tolist(),
        'syn_hourss': syn_hourss,
        'syn_act_power': syn_act_power,
    }
    return results
#%%
def loss_function_conc(x,PPstm,socs,currents,volts,soc_diff,hours_diff):
    faraday_c = 96_485.3321
    hours_diff=np.asarray(hours_diff)
    soc_diff=np.asarray(soc_diff)
    currents=np.asarray(currents)
    test_soc_diff=abs(hours_diff*3600*currents/faraday_c/PPstm.AnodeElectronNumber*PPstm.CellNumber / PPstm.AnodeTankVolume / PPstm.CellNumber / (1600*x[0]))
    loss=np.sqrt(np.mean((abs(soc_diff)-test_soc_diff)**2))
    return(loss)
#%%
def loss_function_mat(x,PPstm,socs,currents,volts,soc_diff,hours_diff,x1):
    param=copy.deepcopy(PPstm)
    faraday_c = 96_485.3321
    param.Display = 'no'
    param.DisplayCurrentSearch = 'no'
    param.DisplayCurrentSearchIteration = 'no'    
    param.MembraneElectronicConductivity=PPstm.MembraneElectronicConductivity*x[0]
    param.CathodeReactionRateConstant = PPstm.CathodeReactionRateConstant*x[0]
    param.AnodeReactionRateConstant = PPstm.AnodeReactionRateConstant*x[0]   
    C230 = 1600*x1
    hours_diff=np.asarray(hours_diff)
    soc_diff=np.asarray(soc_diff)
    currents=np.asarray(currents)
    test_val=[]
    true_val=np.copy(volts)
    for i in range(len(socs)):
        param.SOC=socs[i]
        param.Current = abs(currents[i])
        C3i = (1 -  param.SOC) * C230
        C2i =  param.SOC * C230
        C4i = (1 -  param.SOC) * C230
        C5i =  param.SOC * C230
        param.CathodeReductant1InitialConcentrationInCathode = C4i
        param.CathodeOxidant1InitialConcentrationInCathode = C5i
        param.AnodeOxidant1InitialConcentrationInAnode = C3i
        param.AnodeReductant1InitialConcentrationInAnode = C2i
        if currents[i]>0: param.InitialStatus=-1 #discharge
        if currents[i]<0: param.InitialStatus=1 #charge
        power,v,spl,ppl, Re, PPC  = systemPower(param)
        test_val.append(v)
    test_val=np.array(test_val)
    loss=np.sqrt(np.mean((true_val-test_val)**2))
    return(loss)
#%%
def callback(xk, convergence):
    print(f"Current best solution: {xk}, Convergence: {convergence}")
    sys.stdout.flush()
#%%
def differential_evolution_optimize(loss_function, bounds, args):
    res = optimize.differential_evolution(
        loss_function,
        bounds,
        args=args,
        maxiter=10,
        popsize=10,
        workers=1,
        disp=True,
        callback=callback
    )
    return res