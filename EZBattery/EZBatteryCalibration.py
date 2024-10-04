#%%######################### Descriptions #####################################
# This file includes tools for cell model calibration.The code is developed   #
# by Yunxiang Chen @PNNL. Please contact yunxiang.chen@pnnl.gov for questions.#
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

#%% Loading required packages.
import copy, time
import numpy as np                                                             # numpy 1.24.3
from scipy import optimize                                                     # scipy 1.14.0 
import pandas as pd 
from .EZBatteryCell import RFB                                                 
from .mathm import numCount

#%% Cost function between model predicted voltage and experiment data.        #
def CellLossFunction(x,PP0,FT):
    PP = copy.deepcopy(PP0)
    FT = FT[FT[:,2] != 0,:]                                                    # Remove all data points during rest.

    # Output format control.
    formatSpec1 = 'Parameter: %3.2f, %3.2f, %3.2e, %3.2e'
    formatSpec2 = 'Parameter: %3.2f, %3.2f, %3.2e, %3.2e; Loss: %4.1f, %4.1f mV, %3.1f s'
    formatSpec3 = 'Parameter: %3.2f, %3.2f, %3.3f, %3.3f, %3.3f, %3.3f.'
    formatSpec4 = 'Parameter: %3.2f, %3.2f, %3.3f, %3.3f, %3.3f, %3.3f; Loss: %4.1f, %4.1f mV, %3.1f s, %3.1f s, %3.1f s'
    
    Display = PP.CalibrationDisplay.lower() == 'yes'
    DisplayDebug = PP.CalibrationDisplayDebug.lower() == 'yes'
    ncs = PP.CalibrationCycleStart
    nce = PP.CalibrationCycleEnd
    PP.CycleNumber = nce - ncs + 1
    FT = FT[FT[:,3]>=ncs,:]
    FT = FT[FT[:,3]<=nce,:]
    FT0 = FT.copy()
    beta = PP.CalibrationTimeWeight
    
    # Default cost function for time and voltage.
    delta_t = 1e6; delta_v = 1000*1000; delta_ta = 1e6; delta_tc = 1e6         # unit: s and mV.
    delta_tv = delta_v + delta_t *beta
    if PP.CalibrationMode.lower() == 'single':
        # Updating 4 parameters, based on  Chen Y. et al, JPS, 506, 230192, 2021.
        # https://doi.org/10.1016/j.jpowsour.2021.230192
        PP.CathodeMassTransportCoefficient = x[0]
        PP.AnodeMassTransportCoefficient = x[0]
        PP.MembraneIonDragCoefficient = x[1]
        PP.CathodeReactionRateConstant = x[2]
        PP.AnodeReactionRateConstant = x[3]
        
        if Display and DisplayDebug: print(formatSpec1 %(x[0],x[1],x[2],x[3]))  
        if FT.shape[0] == 0: 
            raise 'Experiment data is empty, please check.'
        else:
            te_start = FT[0,0]; te_end =  FT[-1,0]
            
            # Updating result based on given parameters.
            Re = RFB(PP)
            if len(Re)>0:
                CMP = Re['Potentials'].to_numpy()
                CMP = CMP[abs(CMP[:,6])>0,:]                                   # Remove data points during rest.
                CMP = CMP[:,[0,2,6,19]]
                CMP0 = CMP.copy()
                CMP[:,0] = CMP[:,0] - CMP[0,0] + te_start                      # Aligning the starting time.
                CMP[:,3] = CMP[:,3] - 1 + ncs
                tm_start = te_start
                tm_end = CMP[-1,0]
                delta_ts = abs(tm_start  - te_start)
                delta_te = abs(tm_end - te_end)
                delta_t = delta_te
                
                # Computing rms of voltage for single cycle.
                st0,na0,val0 = numCount(np.sign(CMP[:,2]))
                st9,na9,val9 = numCount(np.sign(FT[:,2]))
                if len(st0) == len(st9):
                    delta_v = 0
                    for i in range(len(st0)):
                        cmp = CMP[st0[i]:st0[i]+na0[i],:]
                        ft = FT[st9[i]:st9[i]+na9[i],:]
                        cmp[:,0] = cmp[:,0] - cmp[0,0]
                        ft[:,0] = ft[:,0] - ft[0,0]
                        tmax = min(cmp[-1,0],ft[-1,0])
                        
                        if PP.CalibrationInterpolation.lower() == 'model':
                            cmp = cmp[cmp[:,0]<=tmax,:]
                            tx = cmp[:,0]
                            ym = cmp[:,1]
                            ye = np.interp(tx,ft[:,0],ft[:,1])    
                        elif PP.CalibrationInterpolation.lower() == 'experiment':
                            ft = ft[ft[:,0]<=tmax,:]
                            tx = ft[:,0]
                            ye = ft[:,1]
                            ym = np.interp(tx,cmp[:,0],cmp[:,1])
                        elif PP.CalibrationInterpolation.lower() == 'uniform':
                            nmax = max(ft.shape[0],cmp.shape[0])
                            tx = np.linspace(0,tmax,nmax)
                            ye = np.interp(tx,ft[:,0],ft[:,1])
                            ym = np.interp(tx,cmp[:,0],cmp[:,1])
                        elif PP.CalibrationInterpolation.lower() == 'combined':
                            tx = np.sort(np.unique(np.concatenate((ft[:,0],cmp[:,0]),axis=0)))
                            ye = np.interp(tx,ft[:,0],ft[:,1])
                            ym = np.interp(tx,cmp[:,0],cmp[:,1])
                        else:
                            raise 'Wrong interpolation type: model, experiment, uniform, and combined.'
                        delta_v = delta_v + np.sqrt(np.mean((ym - ye)**2))
                    delta_v =  delta_v/len(st0)*1e3                            # convert to mV.  
                else:
                    delta_v = 1000*1e3
                delta_tv = delta_v + delta_t*beta 
 
            if Display: print(formatSpec2 %(x[0],x[1],x[2],x[3],delta_tv, delta_v,delta_t))  
            if PP.CalibrationType.lower() == 'debug': 
                Rec = {'Cost':delta_tv,'Experiment':FT0,'Model':CMP0}
            elif PP.CalibrationType.lower() == 'run':
                Rec = delta_tv                                                 # Optimizing for voltage for single cycle.
            else:
                raise 'Wrong calibration type: run, debug.'
    elif PP.CalibrationMode.lower() == 'multiple':
        # Updating 6 parameters, based on Chen Y. et al, JPS, 578, 233210, 2023.
        # https://doi.org/10.1016/j.jpowsour.2023.233210
        PP.CathodeMassTransportCoefficient = x[0]
        PP.AnodeMassTransportCoefficient = x[0]
        PP.MembraneIonDragCoefficient = x[1]
        PP.AnodeReductant1PartitionCoefficient = x[2]
        PP.AnodeOxidant1PartitionCoefficient = x[3]
        PP.CathodeReductant1PartitionCoefficient = x[4]
        PP.CathodeOxidant1PartitionCoefficient = x[5]
        
        if Display and DisplayDebug: print(formatSpec3 %(x[0],x[1],x[2],x[3],x[4],x[5]))
        if FT.shape[0] == 0: 
            raise 'Experiment data is empty, please check.'
        else:
            te_start = FT[0,0]
            
            # Updating result based on given parameters.
            Re = RFB(PP)
            if len(Re)>0:
                CMP = Re['Potentials'].to_numpy()
                CMP = CMP[abs(CMP[:,6])>0,:]                                   # Remove data points during rest.
                CMP = CMP[:,[0,2,6,19]]
                CMP[:,0] = CMP[:,0] - CMP[0,0] + te_start                      # Aligning the starting time.
                CMP[:,3] = CMP[:,3] - 1 + ncs
                CMP0 = CMP.copy()
                nm = min(CMP[-1,3],FT[-1,3])
                CMP = CMP[CMP[:,3]<=nm,:]
                FT = FT[FT[:,3]<=nm,:]
                
                st1,na1,val1 = numCount(CMP[:,3])
                st2,na2,val2 = numCount(FT[:,3])
                delta_ts = np.sqrt(np.mean((CMP[st1,0] - FT[st2,0])**2))
                delta_te = np.sqrt(np.mean((CMP[st1 + na1 - 1,0] - FT[st2 + na2 - 1,0])**2))
                delta_t = abs(CMP[st1[-1] + na1[-1] - 1,0] - FT[st2[-1] + na2[-1] - 1,0])
                delta_t = delta_t/len(st1)
                alpha = 1
                delta_ta = ((1-alpha)*delta_ts + alpha*delta_te)/len(st1)

                st3,na3,val3 = numCount(np.sign(CMP[:,2]))
                st4,na4,val4 = numCount(np.sign(FT[:,2]))
                if len(st3) == len(st4):
                    tm = CMP[st3+na3-1,0] - CMP[st3,0]
                    te = FT[st4+na4-1,0] - FT[st4,0]
                    delta_tc = np.sqrt(np.mean((tm - te)**2))
                    
                    delta_v = 0
                    for i in range(len(st3)):
                        cmp = CMP[st3[i]:st3[i]+na3[i],:]
                        ft = FT[st4[i]:st4[i]+na4[i],:]
                        cmp[:,0] = cmp[:,0] - cmp[0,0]
                        ft[:,0] = ft[:,0] - ft[0,0]
                        tmax = min(cmp[-1,0],ft[-1,0])
                        
                        if PP.CalibrationInterpolation.lower() == 'model':
                            cmp = cmp[cmp[:,0]<=tmax,:]
                            tx = cmp[:,0]
                            ym = cmp[:,1]
                            ye = np.interp(tx,ft[:,0],ft[:,1])    
                        elif PP.CalibrationInterpolation.lower() == 'experiment':
                            ft = ft[ft[:,0]<=tmax,:]
                            tx = ft[:,0]
                            ye = ft[:,1]
                            ym = np.interp(tx,cmp[:,0],cmp[:,1])
                        elif PP.CalibrationInterpolation.lower() == 'uniform':
                            nmax = max(ft.shape[0],cmp.shape[0])
                            tx = np.linspace(0,tmax,nmax)
                            ye = np.interp(tx,ft[:,0],ft[:,1])
                            ym = np.interp(tx,cmp[:,0],cmp[:,1])
                        elif PP.CalibrationInterpolation.lower() == 'combined':
                            tx = np.sort(np.unique(np.concatenate((ft[:,0],cmp[:,0]),axis=0)))
                            ye = np.interp(tx,ft[:,0],ft[:,1])
                            ym = np.interp(tx,cmp[:,0],cmp[:,1])
                        else:
                            raise 'Wrong interpolation type: model, experiment, uniform, and combined.'
                        delta_v = delta_v + np.sqrt(np.mean((ym - ye)**2))
                    delta_v =  delta_v/len(st3)*1e3   
                else:
                    delta_tc = 1e6; delta_v = 1000*1e3
            delta_tv = delta_v + delta_t*beta
             
            if Display: print(formatSpec4 %(x[0],x[1],x[2],\
            x[3],x[4],x[5],delta_tv, delta_v, delta_t,delta_ta,delta_tc))
            
            if PP.CalibrationType.lower() == 'debug': 
                Rec = {'Cost':delta_tv,'Experiment':FT0,'Model':CMP0}
            elif PP.CalibrationType.lower() == 'run':
                Rec = delta_tv
            else:
                raise 'Wrong calibration type: run, debug.'
    else:
        raise 'Wrong calibration type: single or multiple.'
        
    return Rec

#%% Cost function between model predicted voltage and experiment data.        #
def CellLossFunction20240720(x,PP0,FT):
    PP = copy.deepcopy(PP0)
    
    # Output format control.
    formatSpec1 = 'Parameter: %3.3f, %3.3f, %3.3e, %3.3e'
    formatSpec2 = 'Parameter: %3.3f, %3.3f, %3.3f, %3.3f, %3.3f, %3.3f.'
    formatSpec3 = 'Parameter: %3.3f, %3.3f, %3.3e, %3.3e; Loss: %3.0f s, %4.4f V'
    formatSpec4 = 'Parameter: %3.3f, %3.3f, %3.3f, %3.3f, %3.3f, %3.3f; Loss: %3.0f s, %3.0f s'
    
    Display = PP.CalibrationDisplay.lower() == 'yes'
    DisplayDebug = PP.CalibrationDisplayDebug.lower() == 'yes'

    ncs = PP.CalibrationCycleStart
    nce = PP.CalibrationCycleEnd
    
    PP.CycleNumber = nce - ncs + 1
    FT = FT[FT[:,3]>=ncs,:]
    FT = FT[FT[:,3]<=nce,:]
    
    # Default cost function for time and voltage.
    delta_t = 1e6; delta_v = 100
    if PP.CalibrationMode.lower() == 'single':
        # Updating 4 parameters, based on  Chen Y. et al, JPS, 506, 230192, 2021.
        # https://doi.org/10.1016/j.jpowsour.2021.230192
        PP.CathodeMassTransportCoefficient = x[0]
        PP.AnodeMassTransportCoefficient = x[0]
        PP.MembraneIonDragCoefficient = x[1]
        PP.CathodeReactionRateConstant = x[2]
        PP.AnodeReactionRateConstant = x[3]
        
        if Display and DisplayDebug: print(formatSpec1 %(x[0],x[1],x[2],x[3]))  
        if FT.shape[0] == 0: 
            raise 'Experiment data is empty, please check.'
        else:
            te_start = FT[0,0]; te_end =  FT[-1,0]
            
            # Updating result based on given parameters.
            Re = RFB(PP)
            if len(Re)>0:
                CMP = Re['Potentials'].to_numpy()
                CMP = CMP[abs(CMP[:,6])>0,:]                                   # Remove data points during rest.
                CMP = CMP[:,[0,2,6,19]]
                CMP[:,0] = CMP[:,0] - CMP[0,0] + te_start                      # Aligning the starting time.
                tm_start = te_start
                tm_end = CMP[-1,0]
                delta_ts = abs(tm_start  - te_start)
                delta_te = abs(tm_end - te_end)
                alpha = 0.5
                delta_t = (1-alpha)*delta_ts + alpha*delta_te
                
                # Computing rms of voltage for single cycle.
                tmax = min(tm_end, te_end)
                if PP.CalibrationInterpolation.lower() == 'model':
                    CMP = CMP[CMP[:,0]<=tmax,:]
                    tx = CMP[:,0]
                    ym = CMP[:,1]
                    ye = np.interp(tx,FT[:,0],FT[:,1])
                elif PP.CalibrationInterpolation.lower() == 'experiment':
                    FT = FT[FT[:,0]<=tmax,:]
                    tx = FT[:,0]
                    ye = FT[:,1]
                    ym = np.interp(tx,CMP[:,0],CMP[:,1])
                elif PP.CalibrationInterpolation.lower() == 'uniform':
                    nmax = max(FT.shape[0],CMP.shape[0])
                    tx = np.linspace(te_start,tmax,nmax)
                    ye = np.interp(tx,FT[:,0],FT[:,1])
                    ym = np.interp(tx,CMP[:,0],CMP[:,1])
                elif PP.CalibrationInterpolation.lower() == 'combined':
                    tx = np.sort(np.unique(np.concatenate((FT[:,0],CMP[:,0]),axis=0)))
                    ye = np.interp(tx,FT[:,0],FT[:,1])
                    ym = np.interp(tx,CMP[:,0],CMP[:,1])
                else:
                    raise 'Wrong interpolation type: model, experiment, uniform, and combined.'
                delta_v = np.sqrt(np.mean((ym - ye)**2))                       # Cost function.
            
            if Display: print(formatSpec3 %(x[0],x[1],x[2],x[3],delta_t,delta_v))  
            if PP.CalibrationType.lower() == 'debug': 
                XY = np.zeros((len(tx),3))
                XY[:,0] = tx; XY[:,1] = ye; XY[:,2] = ym
                Rec = (delta_v,FT,CMP,XY)
            elif PP.CalibrationType.lower() == 'run':
                Rec = delta_v                                                  # Optimizing for voltage for single cycle.
            else:
                raise 'Wrong calibration type: run, debug.'
    elif PP.CalibrationMode.lower() == 'multiple':
        # Updating 6 parameters, based on Chen Y. et al, JPS, 578, 233210, 2023.
        # https://doi.org/10.1016/j.jpowsour.2023.233210
        PP.CathodeMassTransportCoefficient = x[0]
        PP.AnodeMassTransportCoefficient = x[0]
        PP.MembraneIonDragCoefficient = x[1]
        PP.AnodeReductant1PartitionCoefficient = x[2]
        PP.AnodeOxidant1PartitionCoefficient = x[3]
        PP.CathodeReductant1PartitionCoefficient = x[4]
        PP.CathodeOxidant1PartitionCoefficient = x[5]
        
        if Display and DisplayDebug: print(formatSpec2 %(x[0],x[1],x[2],x[3],x[4],x[5]))
        if FT.shape[0] == 0: 
            raise 'Experiment data is empty, please check.'
        else:
            te_start = FT[0,0]
            
            # Updating result based on given parameters.
            Re = RFB(PP)
            if len(Re)>0:
                CMP = Re['Potentials'].to_numpy()
                CMP = CMP[abs(CMP[:,6])>0,:]                                   # Remove data points during rest.
                CMP = CMP[:,[0,2,6,19]]
                CMP[:,0] = CMP[:,0] - CMP[0,0] + te_start                      # Aligning the starting time.
                st1,na1,val1 = numCount(CMP[:,3])
                st2,na2,val2 = numCount(FT[:,3])
                nm = min(len(st1),len(st2))
                st1 = st1[:nm]; na1 = na1[:nm]; val1 = val1[:nm]
                st2 = st2[:nm]; na2 = na2[:nm]; val2 = val2[:nm]
                delta_ts = np.sqrt(np.mean((CMP[st1,0] - FT[st2,0])**2))
                delta_te = np.sqrt(np.mean((CMP[st1 + na1 - 1,0] - FT[st2 + na2 - 1,0])**2))
                alpha = 0.5
                delta_ta = (1-alpha)*delta_ts + alpha*delta_te
                delta_t = abs(CMP[st1[-1] + na1[-1] - 1,0] - FT[st2[-1] + na2[-1] - 1,0])

            if Display: print(formatSpec4 %(x[0],x[1],x[2],x[3],x[4],x[5],delta_t,delta_ta))
            if PP.CalibrationType.lower() == 'debug': 
                Rec = (delta_t,FT,CMP)
            elif PP.CalibrationType.lower() == 'run':
                Rec = delta_t
            else:
                raise 'Wrong calibration type: run, debug.'
    else:
        raise 'Wrong calibration type: single or multiple.'
        
    return Rec

#%%      
def CellLossFunction20240715(x,PP0,FT):
    PP = copy.deepcopy(PP0)
    
    # Output format control.
    formatSpec1 = 'Parameter: %3.3f, %3.3f, %3.3e, %3.3e'
    formatSpec2 = 'Parameter: %3.3f, %3.3f, %3.3f, %3.3f, %3.3f, %3.3f.'
    formatSpec3 = 'Parameter: %3.3f, %3.3f, %3.3e, %3.3e; Loss: %4.4f V, %3.0f s'
    formatSpec4 = 'Parameter: %3.3f, %3.3f, %3.3f, %3.3f, %3.3f, %3.3f; Loss: %4.4f V, %3.0f s'
    
    Display = PP.CalibrationDisplay.lower() == 'yes'
    DisplayDebug = PP.CalibrationDisplayDebug.lower() == 'yes'

    ncs = PP.CalibrationCycleStart
    nce = PP.CalibrationCycleEnd
    
    PP.CycleNumber = nce - ncs + 1
    FT = FT[FT[:,3]>=ncs,:]
    FT = FT[FT[:,3]<=nce,:]
    
    # Default cost function for time and voltage.
    delta_t = 1e6; delta_v = 10; XY = []
    if PP.CalibrationMode.lower() == 'single':
        # Updating 4 parameters, based on  Chen Y. et al, JPS, 506, 230192, 2021.
        # https://doi.org/10.1016/j.jpowsour.2021.230192
        PP.CathodeMassTransportCoefficient = x[0]
        PP.AnodeMassTransportCoefficient = x[0]
        PP.MembraneIonDragCoefficient = x[1]
        PP.CathodeReactionRateConstant = x[2]
        PP.AnodeReactionRateConstant = x[3]
    elif PP.CalibrationMode.lower() == 'multiple':
        # Updating 6 parameters, based on Chen Y. et al, JPS, 578, 233210, 2023.
        # https://doi.org/10.1016/j.jpowsour.2023.233210
        PP.CathodeMassTransportCoefficient = x[0]
        PP.AnodeMassTransportCoefficient = x[0]
        PP.MembraneIonDragCoefficient = x[1]
        PP.AnodeReductant1PartitionCoefficient = x[2]
        PP.AnodeOxidant1PartitionCoefficient = x[3]
        PP.CathodeReductant1PartitionCoefficient = x[4]
        PP.CathodeOxidant1PartitionCoefficient = x[5]
    else:
        raise 'Wrong calibration type: single or multiple.'
    
    if Display and DisplayDebug:
        if PP.CalibrationType.lower() == 'single':
            print(formatSpec1 %(x[0],x[1],x[2],x[3]))  
        else:
            print(formatSpec2 %(x[0],x[1],x[2],x[3],x[4],x[5]))
    
    # Updating cost.
    if FT.shape[0] == 0: raise 'Experiment data is empty, please check.'
    
    if FT.shape[0]>0:
        te_start = FT[0,0]
        te_end =  FT[-1,0]
        
        # Updating result based on given parameters.
        Re = RFB(PP)
        if len(Re)>0:
            CMP = Re['Potentials'].to_numpy()
            CMP = CMP[abs(CMP[:,6])>0,:]                                       # Remove data points during rest.
            CMP = CMP[:,[0,2]]
            CMP[:,0] = CMP[:,0] - CMP[0,0]*0 + te_start                        # Aligning the starting time.
            tp_end = CMP[-1,0]
            delta_t = abs(tp_end - te_end)                                     # Time difference.
    
            # Computing voltage difference by interpolating
            tmax = min(tp_end, te_end)
            if PP.CalibrationInterpolation.lower() == 'model':
                CMP = CMP[CMP[:,0]<=tmax,:]
                tx = CMP[:,0]
                ym = CMP[:,1]
                ye = np.interp(tx,FT[:,0],FT[:,1])
            elif PP.CalibrationInterpolation.lower() == 'experiment':
                FT = FT[FT[:,0]<=tmax,:]
                tx = FT[:,0]
                ye = FT[:,1]
                ym = np.interp(tx,CMP[:,0],CMP[:,1])
            elif PP.CalibrationInterpolation.lower() == 'uniform':
                nmax = max(FT.shape[0],CMP.shape[0])
                tx = np.linspace(te_start,tmax,nmax)
                ye = np.interp(tx,FT[:,0],FT[:,1])
                ym = np.interp(tx,CMP[:,0],CMP[:,1])
            elif PP.CalibrationInterpolation.lower() == 'combined':
                tx = np.sort(np.unique(np.concatenate((FT[:,0],CMP[:,0]),axis=0)))
                ye = np.interp(tx,FT[:,0],FT[:,1])
                ym = np.interp(tx,CMP[:,0],CMP[:,1])
            else:
                raise 'Wrong interpolation type: model, experiment, uniform, and combined.'
            
            delta_v = np.sqrt(np.mean((ym - ye)**2))                           # Cost function.

        # Print information control.
        if Display:
            if PP.CalibrationType.lower() == 'single':
                print(formatSpec3 %(x[0],x[1],x[2],x[3],delta_v,delta_t))  
            else:
                print(formatSpec4 %(x[0],x[1],x[2],x[3],x[4],x[5],delta_v,delta_t))

        # Return data control 
        if PP.CalibrationType.lower() == 'debug': 
            XY =  np.zeros((len(tx),3))
            XY[:,0] = tx; XY[:,1] = ye; XY[:,2] = ym
            Rec = (delta_v,FT,CMP,XY)
        elif PP.CalibrationType.lower() == 'run':
            Rec = delta_v
        else:
            raise 'Wrong calibration type: run, debug.'
    return Rec

#%% Used for differentiable evolution to print parameters.
def CallBack(xk, convergence):
    print(f"Solution: {xk}, Convergence: {convergence}")

#%%
# This function call differential evolution to calibrated model parameters.   #
def Calibration(PP0,F):
    PP = copy.deepcopy(PP0)
    PP.CalibrationType =  'run'                                                # Calibration only works with "run" mode.

    # Output info control.
    formatSpec1 = 'EZBattery calibration for cycle %s.'
    formatSpec2 = 'EZBattery calibration for cycles %s - %s'
    print('')
    
    starttime = time.time()
    FT = F[F[:,2] != 0,:]                                                      # Remove all data points during rest.
    seed_id = 123                                                              # For reproducibility.
    npp = PP.NumberOfProcessor                                                 # Number of CPU cores to use.
    popsize = PP.PopulationSize                                                # Number of popsize.
    maxIter = PP.MaxIteration                                                  # Maximum iteration step.
    disp = PP.DisplayDifferentialEvolution.lower() == 'yes'                    # If output information on screen (integer).
    ncs = PP.CalibrationCycleStart
    nce = PP.CalibrationCycleEnd   
    
    # Auto-calibration for a single cycle.
    if PP.CalibrationMode.lower() == 'single':
        print(formatSpec1 %(str(ncs)))
        bounds = PP.BoundsSingle
        args = (PP,FT)
        Res = optimize.differential_evolution(CellLossFunction, bounds=bounds,\
        args=args, maxiter=maxIter,popsize=popsize,workers=npp,disp=disp,\
            callback=CallBack,updating='deferred',seed=np.random.seed(seed_id))
    elif PP.CalibrationMode.lower() == 'multiple':
        print(formatSpec2 %(str(ncs),str(nce)))
        bounds = PP.BoundsMultiple
        if hasattr(PP,'CalibratedReactionRateConstant'):
            k_pn = PP.CalibratedReactionRateConstant
            PP.CathodeReactionRateConstant = k_pn[0]
            PP.AnodeReactionRateConstant = k_pn[1]
        if hasattr(PP,'CalibratedMassTransportCoefficient'):
            beta = PP.CalibratedMassTransportCoefficient
            bounds[0] = (beta,beta)
        if hasattr(PP,'CalibratedMembraneIonDragCoefficient'):
            nd = PP.CalibratedMembraneIonDragCoefficient
            bounds[1] = (nd,nd)
        args = (PP,FT)
        Res = optimize.differential_evolution(CellLossFunction, bounds=bounds,\
        args=args, maxiter=maxIter,popsize=popsize,workers=npp,disp=disp,\
            callback=CallBack,updating='deferred',seed=np.random.seed(seed_id))
    else:
        raise 'Wrong calibration type: single or multiple.'
    
    # Output data to file.
    out = np.concatenate((Res.x,[Res.fun]),axis=0)
    if hasattr(PP,'CalibrationOutputPath'):
        SDPF = PP.CalibrationOutputPath
        if PP.CalibrationMode.lower() == 'single':
            outname = SDPF.replace('.xlsx','_SingleCycle_' + str(ncs) + '_' + str(nce) + '_' + str(npp) + '.csv')
            vnames = ['MassTransportCoefficient','MembraneIonDragCoefficient',\
                      'PCathodeReactionRateConstant','AnodeReactionRateConstant','Cost']
        elif PP.CalibrationMode.lower() == 'multiple':
            outname = SDPF.replace('.xlsx','_MultipleCycle_' + str(ncs) + '_' + str(nce) + '_' + str(npp) + '.csv')
            vnames = ['MassTransportCoefficient','MembraneIonDragCoefficient',\
            'AnodeReductant1PartitionCoefficient','AnodeOxidant1PartitionCoefficient',\
            'CathodeReductant1PartitionCoefficient','CathodeOxidant1PartitionCoefficient','Cost']
        out = out.reshape((1,len(vnames)))
        out = pd.DataFrame(out,columns=vnames)
        out.to_csv(outname,index=False)

    # Output final information.
    endtime = time.time()
    print('\nFinal cost: ',Res.fun)
    print('Final parameter: ',Res.x)
    
    # Print full cost information.
    PP.CalibrationType = PP0.CalibrationType
    PP.CalibrationDisplay = 'yes'
    PP.CalibrationDisplayDebug = 'no'
    Rec = CellLossFunction(Res.x,PP,FT)
    Res.loss = Rec
    print('Calibration time: ',endtime - starttime, ' s.\n')
    return Res