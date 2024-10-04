#%%######################### Descriptions #####################################
# This code is developed by Dr. Yunxiang Chen @PNNL over multiple years for   #
# multiple projecs. Potential users are permitted to use for non-commercial   #
# purposes without modifying any part of the code. Please contact the author  #
# to obtain written permission for commercial or modifying usage.             #

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

#%% Required packages.
import warnings
import numpy as np
from scipy.optimize import fsolve

#%% Calculating the start, number, and value of unique values in an array.    #
# A could be an array with N*0 and N*1 dimension.                             #
# Performance: 0.04 s for 1M data, 0.22 s for 10M data, too slow for 100M data#
def numCount(A, tol=0):
    Idx = np.arange(A.shape[0])
    df = np.abs(np.diff(A, axis=0)) > tol
    df = np.insert(df, 0, True)
    df = np.insert(df, len(df), True)
    CheckI = np.where(df)[0]
    st = Idx[CheckI[:-1]]
    na = np.diff(CheckI)
    val = A[st]
    return st, na, val

#%% The latest version of the function used to calculate battery performance  #
# metrics based on input battery parameters and voltage data. Column names:   #
# cycle number, charge time, discharge time, charge energy, discharge energy, #
# Energy efficiency.                                                          #
def calcEnergy(P,V, option='rest'):
    V = V[np.where(~np.any(np.isnan(V),axis=1))[0],:]
    st,na,_ = numCount(V[:,-1])
    nc = len(st)
    # Cycle id, charging time(s), discharge time (s), CE, charge energy (Wh), #
    # discharge energy (Wh), EE.                                              #
    E = np.full((nc,7),np.nan)
    I = P.Current
    for i in range(nc):
        vi = V[st[i] - 1 + range(na[i]) + 1,:]
        if np.sum(vi[:,6]<0)<=2 or \
            abs(vi[-1,2] - P.CutOffVoltageDischarge)>0.01: continue
        # For charging.
        E[i,0] = i + 1
        idc = np.where(vi[:,6]>0)[0]
        if len(idc)>=2:
            vic = vi[idc,:]
            E[i,1] = vic[-1,0] - vic[0,0]
            E[i,4] =  np.trapz(vic[:,2],vic[:,0])*I/3600 
        # For discharging.
        if option.lower() in 'rest':                                           # Including the last point in rest in discharging.
            idr = np.where(vi[:,6]==0)[0]
            iddc = np.where(vi[:,6]<0)[0]
            idx = np.concatenate((np.array([idr[-1]]),iddc))
        else:
            idx = np.where(vi[:,6]<0)[0]
        if len(idx)>=3:
            vidc = vi[idx,:]
            E[i,2] = vidc[-1,0] - vidc[0,0]
            E[i,5] =  np.trapz(vidc[:,2],vidc[:,0])*I/3600  
    E[:,3] = E[:,2]/E[:,1]
    E[:,6] = E[:,5]/E[:,4]
    E = E[np.logical_not(np.isnan(E[:,0])),:]
    return E

#%% Combine to Lists to an array format.
def conc(L1,L2):
    L=[]
    for i in range(len(L1)):
        L.append(L1[i])
        L.append(L2[i])
    return np.array(L)

#%% Converting object data type to array.
def obj2arr(arr, typ):
    """
    Convert an object array of same-sized arrays to a normal 3D array
    with dtype=typ. This is a workaround as numpy doesn't realize that
    the object arrays are numpy arrays of the same legth, so just using
    array.astype(typ) fails. Technically works if the items are numbers
    and not arrays, but then `arr.astype(typ)` should be used.
    """
    full_shape = (*arr.shape, *np.shape(arr.flat[0]))
    return np.vstack(arr.flatten()).astype(typ).reshape(full_shape)

#%% Redefining linspace to be consistant with Matlab.
def linspace(start,stop, npoints):
    values = np.zeros(npoints,dtype=float)
    if npoints == 1:
        values[0] = stop
    else:
        values[:] = np.linspace(start, stop, npoints)
    return values

#%% Solving for over-potential from Bulter-Volmer equation.
def BV(alpha, c0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not isinstance(alpha, (list, np.ndarray)):
            alpha = [alpha]
        if not isinstance(c0, (list, np.ndarray)):
            c0 = [c0]
        x0 = np.zeros((len(alpha), len(c0)))
        for i in range(len(alpha)):
            for j in range(len(c0)):
                alpha_i = alpha[i]
                c0_j = c0[j]
                bvx = lambda x: np.exp((1 - alpha_i) * x) - np.exp(-alpha_i * x) - c0_j
                x0[i, j] = fsolve(bvx, 0.1 * np.sign(c0_j))
    return x0

#%%
def cross_over_rhs(M, X, S):
    RHS = np.dot(M, X) + S
    return RHS

#%% For testing purposes.
if __name__ == "__main__":
    x0 = BV(0.4,1e6)
    print(x0)