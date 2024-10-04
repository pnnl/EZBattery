import os, copy
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm
import multiprocessing
from EZBattery import BatteryParameter
from EZBattery import EZBatteryCell, RFB, searchCurrent, searchSystem, systemPower

def loss_function(x, param, socs, currents, true_power):
    C40 = 3000 
    C50 = 0

    param.AnodeOxidant1InitialConcentrationInAnode = x[0]
    param.MembraneElectronicConductivity = x[1]

    test_power = []
    true_power_cut = []
    interval = 50
    for i in range(1, len(socs), interval):
        if abs(currents[i]) > 1e-8:
            param.SOC = socs[i]
            param.Current = abs(currents[i])
            C450 = C40 / 3 + C50
            C4i = (1 - param.SOC) * C450 * 3
            C5i = param.SOC * C450
            param.CathodeReductant1InitialConcentrationInCathode = C4i
            param.CathodeOxidant1InitialConcentrationInCathode = C5i
            if currents[i] > 0: param.InitialStatus = -1  # discharge
            if currents[i] < 0: param.InitialStatus = 1   # charge
            power, v, spl, ppl, Re, PPC = systemPower(param)
            if currents[i] < 0: power = -power
            test_power.append(power)
            true_power_cut.append(true_power[i])

    true_power_cut = np.array(true_power_cut)
    test_power = np.array(test_power)
    loss = np.sqrt(np.mean((true_power_cut - test_power) ** 2))
    #print('loss', loss, 'x', x[0], x[1], flush=True)
    return loss

def callback(xk, convergence):
    print(f"Current best solution: {xk}, Convergence: {convergence}")
    #print("Flushing output...")
    sys.stdout.flush()

def differential_evolution_optimize(loss_function, bounds, args):
    res = optimize.differential_evolution(
        loss_function,
        bounds,
        args=args,
        maxiter=25,
        popsize=120,
        workers=40,
        disp=True,
        callback=callback
    )
    return res

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Use 'spawn' if on Windows
    WorkDir = os.getcwd() + os.sep + 'data' 
    SDPFName = 'SDPF_ZIB_V5.xlsx'
    SDPFPath = WorkDir + os.sep + SDPFName
    PP0 = BatteryParameter(SDPFPath)
    PP1 = copy.deepcopy(PP0)
    PP1.BatteryMode = 'power'
    PP1.StartTime = 0
    PP1.EndTime = 1
    PP1.TimeStep = 1
    PP1.RefineTimePoints = 1
    PP1.Display = 'no'
    PP1.DisplayCurrentSearch = 'yes'
    PP1.DisplayCurrentSearchIteration = 'no'
    PP1.SearchCurrentConvergeTolerance = 1e-3
    PP1.CellNumber = 60
    PP1.ElectrodeHeight = 0.25
    PP1.ElectrodeWidth = 0.25
    PP1.CathodeTankVolume = 6e-3
    PP1.AnodeTankVolume = 4e-3
    PP1.CathodePumpRate = 2e-6
    PP1.AnodePumpRate = 2e-6
    PP1.Power = 30
    PP1.SearchCurrentMinimum = 0.001
    PP1.SearchCurrentMaximum = 200
    PP1.SOC = 0.6

    loadfile = WorkDir + os.sep + 'one_week_syn_data_2_deg.csv'
    one_week_syn_data = pd.read_csv(loadfile)
    result_file=WorkDir + os.sep + 'diagnosis_results_2_deg_parallel.csv'
    result_out=open(result_file,'w')
    bounds = [(300, 450), (0.5, 1.5)]
    for i in range(7):
        true_batch = one_week_syn_data[np.floor(one_week_syn_data['Time(h)'] / 24) == i]
        print('diagnostic day #', i, 'size of data', true_batch.shape, flush=True)

        res = dict()
        PP2 = copy.deepcopy(PP1)
        print(true_batch)

        result = differential_evolution_optimize(
            loss_function, bounds, (PP2, np.array(true_batch['SOC']), np.array(true_batch['current(A)']), np.array(true_batch['Power_sys(W)']))
        )
        print('results x:', result['x'][0], result['x'][1],file=result_out,flush=True)
        bounds = [(result['x'][0] * 0.85, result['x'][0] * 1.15), (result['x'][1] * 0.85, result['x'][1] * 1.15)]
    result_out.close()
    
