#%% Notes of usage of .self in Python.
# The Chemical, Calculated, compute_chemical are re-culated each time when a  #
# fixed parameter is updated. If 10 fixed parameters are updated, then the    #
# recalculation is conducted for 10 times.                                    #

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
import pandas as pd
import numpy as np
from .mathm import obj2arr

#%%
class BatteryParameter:
    def __init__(self, filepath=None):
        self._cache = {}
        self.filepath = filepath
        self.Constants = self.set_Constants()
        if filepath:
            self.read_and_process_data()
            
    def set_Constants(self):
        return {
            'KozenyCarmanConstant': 5.55,
            'Faraday': 96485.3329,
            'GasConstant': 8.31446261815324,
            'EntranceLength': 0.033
        }
    
    def __setattr__(self, key, value):
        # Invalidate cache if a property that affects Chemical or Calculated is changed
        if key not in ['_cache', 'Constants'] and key in self.__dict__:
        # if  key in self.__dict__:
            self._cache.pop('Chemical', None)
            self._cache.pop('Calculated', None)
        super().__setattr__(key, value)
        
    def read_and_process_data(self):
        if self.filepath is None:
            raise ValueError("Filepath not set")

        # Read data from the specified Excel file
        pp = pd.read_excel(self.filepath, sheet_name='property_parameter')
        # Filter out rows that are not parameters
        keywords = ['parameter', 'cell', 'chemical', 'material', 'membrane',
                    'operation', 'thermodynamics', 'additional']
        
        pp = pp[~pp.iloc[:, 0].str.lower().isin(keywords)].drop_duplicates(subset=pp.columns[0], keep='first')

        # Set attributes based on the data read from the file
        for i in range(len(pp)):
            key = pp.iloc[i, 0]  # Standardize the attribute name
            value = pp.iloc[i, 2]
            setattr(self, key, value)
            
    @property
    def Chemical(self):
        if 'Chemical' not in self._cache:
            self._cache['Chemical'] = self._compute_chemical()
        return self._cache['Chemical']

    @property
    def Calculated(self):
        if 'Calculated' not in self._cache:
            self._cache['Calculated'] = self._compute_calculated()
        return self._cache['Calculated']

    def _compute_chemical(self):
        key1 = ['CathodeReductant', 'CathodeOxidant', 'AnodeOxidant', 'AnodeReductant']
        key2 = ['Name', 'Coefficient', 'Valence', 'Diffusivity', 'InitialConcentrationInCathode',
                'InitialConcentrationInAnode', 'CrossMembraneRate', 'DiffusivityInMembrane', 'PartitionCoefficient']
    
        # Initialize the Chemical dictionary with sub-dictionaries for each component
        Chemical = {component: {prop: [] for prop in key2} for component in key1}
        
        # Populate the Chemical dictionary
        for component in key1:
            for prop in key2:
                for k in range(1, 101):
                    attribute_name = f"{component}{k}{prop}"
                    if hasattr(self, attribute_name):
                        value = getattr(self, attribute_name)
                        Chemical[component][prop].append(value)
    
        # If maintaining separate top-level lists for each property is required
        for prop in key2:
            Chemical[prop] = []
            for component in key1:
                Chemical[prop].extend(Chemical[component][prop])
                
        Chemical['Index'] = []
        for component in key1:
            # Only consider the length of the first property list for each component
            if key2[0] in Chemical[component]:  # Check if the first property exists for the component
                Chemical['Index'].extend([key1.index(component) + 1] * len(Chemical[component][key2[0]]))

        idx = list(range(len(Chemical['Index'])))
        unique_elements, ia = [], []
        for i, element in enumerate(Chemical['Index']):
            if element not in unique_elements:
                unique_elements.append(element)
                ia.append(i)
    
        Chemical['Active'] = ia
        Chemical['NoneActive'] = [x for x in idx if x not in ia]
    
        # Update 'ElectronNumber' and 'StandardPotential' based on existing attributes
        Chemical['ElectronNumber'] = [getattr(self, 'CathodeElectronNumber', None), 
                                      getattr(self, 'AnodeElectronNumber', None)]
        Chemical['StandardPotential'] = [getattr(self, 'CathodeStandardPotential', None), 
                                          getattr(self, 'AnodeStandardPotential', None)]

        return Chemical

    def _compute_calculated(self):
        Calculated ={}
        
        # Extracting input parameters to local variables.
        T = self.Temperature                                                   # Operating temperature.
        W_pos = self.CathodePumpRate                                           # Volumetric flow rate at positive electrode (m^3/s);
        W_neg = self.AnodePumpRate                                             # Volumetric flow rate at negative electrode (m^3/s);
        I = self.Current                                                       # Constant current for charge and discharge (A);
        Lh = self.ElectrodeHeight                                              # Electrode height (m)
        Lt = self.ElectrodeThickness                                           # Electrode thickness (m);
        Lw = self.ElectrodeWidth                                               # Electrode width (m);
        Lm = self.MembraneThickness                                            # Membrane thickness (m);
        Lc = self.CurrentCollectorThickness                                    # Current collector thickness (m);
        eps_pos = self.CathodeElectrodePorosity                                # Cathode electrode porosity.
        eps_neg = self.AnodeElectrodePorosity                                  # Anode electrode porosity.
        VTP = self.CathodeTankVolume                                           # Electrolyte tank volume (Half cell) (m^3);
        VTN = self.AnodeTankVolume                                             # Electrolyte tank volume (Half cell) (m^3);
        C40 = self.Chemical['CathodeReductant']['InitialConcentrationInCathode'][0]# Initial V(IV) concentration (mol/m^3);
        C50 = self.Chemical['CathodeOxidant']['InitialConcentrationInCathode'][0] # Initial V(V) concentration (mol/m^3);
        C30 = self.Chemical['AnodeOxidant']['InitialConcentrationInAnode'][0]   # Initial V(III) concentration (mol/m^3);
        C20 = self.Chemical['AnodeReductant']['InitialConcentrationInAnode'][0] # Initial V(II) concentration (mol/m^3);
        gamma = self.Constants['EntranceLength']                               # Entrance length constant (1).
        F = self.Constants['Faraday']                                          # Faraday constant (s A/mol);
        R = self.Constants['GasConstant']                                      # Gas constant (J/K?mol)
        E01 = self.AnodeStandardPotential                                      # Negative electrode Equilibrium potential (V);
        E02 = self.CathodeStandardPotential                                    # Positive electrode Equilibrium potential (V);
        D5 = self.Chemical['CathodeOxidant']['Diffusivity'][0]                 # V(V) Diffusivity   (m^2/s);
        D2 = self.Chemical['AnodeReductant']['Diffusivity'][0]                 # V(II) Diffusivity (m^2/s);
        mu_pos = self.CathodeElectrolyteViscosity                              # Electrolyte viscosity in positive (Pa.s);
        mu_neg = self.AnodeElectrolyteViscosity                                # Electrolyte viscosity in negative (Pa.s);
        rho_pos = self.CathodeElectrolyteDensity                               # Electrolyte density in positive (kg/m^3).
        rho_neg = self.AnodeElectrolyteDensity                                 # Electrolyte density in negative (kg/m^3).
        beta_pos = self.CathodeMassTransportCoefficient                        # Mass transport coefficient in positive (1).
        beta_neg = self.AnodeMassTransportCoefficient                          # Mass transport coefficient in negative (1).
        #Sigma_e_pos = self.CathodeElectrodeConductivity                       # Cathode electrode electronic conductivity (S/m);
        #Sigma_e_neg = self.AnodeElectrodeConductivity                         # Cathode electrode electronic conductivity (S/m);
        Sigma_ed_pos = self.CathodeElectrodeConductivity;                      # Cathode electrode electronic conductivity (S/m);
        Sigma_ed_neg = self.AnodeElectrodeConductivity;                        # Cathode electrode electronic conductivity (S/m);
        Sigma_el_pos = self.CathodeElectrolyteConductivity;                    # Cathode electrolyte electronic conductivity (S/m);
        Sigma_el_neg = self.AnodeElectrolyteConductivity;                      # Cathode electrolyte electronic conductivity (S/m);
        beta = self.ElectrolyteEffect                                          # A parameter used to estimate the effect of electrolyte on ions.
        Sigma_c = self.CurrentCollectorConductivity                            # Collector conductivity (S/m);
        g = self.Constants['EntranceLength']                                   # Entrance length;
        df_pos = self.CathodeElectrodeFiberSize                                # The value of fiber size (m).
        df_neg = self.AnodeElectrodeFiberSize                                  # The value of fiber size (m).
        ks_pos = self.CathodeSpecificAreaCoefficient                           # The value of specific area coefficient (1).
        ks_neg = self.AnodeSpecificAreaCoefficient                             # The value of specific area coefficient (1).
        NP = self.Chemical['ElectronNumber'][0]
        NE = self.Chemical['ElectronNumber'][1]
        
        # Calculating estimated electrode geometric properties.
        Sigma_m = self.MembraneElectronicConductivity
        if isinstance(Sigma_m, (int, float)) and Sigma_m > 0:
            Sigma_m = self.MembraneElectronicConductivity
        else:
            Sigma_m = (0.5193 * 22 - 0.326) * np.exp(1268 * (1/303 - 1/T))     # Membrane conductivity.

        Ae = Lh * Lw                                                           # Electrode area (m^2).
        I0 = I / Ae                                                            # Nominal current density.

        if self.ElectrodeMode.lower() == 'strip':
            A_pos_base = 2 / df_pos * (1 - eps_pos)  
            A_pos = ks_pos * A_pos_base                                        # Specific area calculated from a model in Chen2021
            ds_pos = 2 * eps_pos / A_pos_corrected                             # Estimated pore size.
            N_channel_pos = Lw / (df_pos + ds_pos)                             # Number of channels
            A_neg_base = 2 / df_neg * (1 - eps_neg) 
            A_neg = ks_neg * A_neg_base                                        # Specific area calculated from a model in Chen2021
            ds_neg = 2 * eps_neg / A_neg_corrected                             # Estimated pore size.
            N_channel_neg = Lw / (df_neg + ds_neg)                             # Number of channels
        elif self.ElectrodeMode.lower() == 'grid':
            A_pos_base = 4 / df_pos * (np.sqrt(eps_pos) - eps_pos)  
            A_pos = ks_pos * A_pos_base                                        # Specific area calculated from a model in Chen2021
            ds_pos = 4 * eps_pos / A_pos                                       # Estimated pore space.
            N_channel_pos = Lt * Lw / (df_pos + ds_pos)**2                     # Number of channels
            
            A_neg_base = 4 / df_neg * (np.sqrt(eps_neg) - eps_neg) 
            A_neg = ks_neg * A_neg_base                                        # Specific area calculated from a model in Chen2021
            ds_neg = 4 * eps_neg / A_neg                                       # Estimated pore space.
            N_channel_neg = Lt * Lw / (df_neg + ds_neg)**2                     # Number of channels
            
        u0_neg = W_neg / (Lw * Lt * eps_neg)                                   # Average electrolyte flow speed
        u0_pos = W_pos / (Lw * Lt * eps_pos)                                   # Average electrolyte flow speed
        I_density_pos = I / (Lt * Lw * Lh * A_pos)                             # Average current density (A/m^2)
        I_density_neg = I / (Lt * Lw * Lh * A_neg)                             # Average current density (A/m^2)
        VE = Lt * Lw * Lh                                                      # Volume of half electrode (m^3).
        delta_pos = VE / VTP                                                   # Ratio of electrode volume to tank volume.
        delta_neg = VE / VTN                                                   # Ratio of electrode volume to tank volume.
        tau_pos = Lh / u0_pos                                                  # Time required to move from the inlet to the outlet (s).
        tau_neg = Lh / u0_neg                                                  # Time required to move from the inlet to the outlet (s).

        eps0_pos = (1 + eps_pos * delta_pos) / tau_pos                         # Characteristic time change rate (1/s).
        eps0_neg = (1 + eps_neg * delta_neg) / tau_neg                         # Characteristic time change rate (1/s).

        # Approaches to estimate reaction rate constants.
        if hasattr(self, 'TemperatureRef'):
            self.TemperatureRef = 293                                          # Reference temperature (K).
            self.CathodeReactionRateConstantRef = 3e-9                         # Standard rate constant in positive electrode (m/s).
            self.AnodeReactionRateConstantRef = 3.56e-6                        # Standard rate constant in the negative electrode (m/s).
            Tref = self.TemperatureRef
            k1 = self.AnodeReactionRateConstantRef * np.exp(-F * E01 / R * (1 / Tref - 1 / T))    # Negative.
            k2 = self.CathodeReactionRateConstantRef * np.exp(-F * E02 / R * (1 / Tref - 1 / T))  # Positive.
        else:
            k1 = self.AnodeReactionRateConstant                                # Negative electrode rate constant V(III)+e = V(II)  (m/s);
            k2 = self.CathodeReactionRateConstant                              # Positive electrode rate constant V(5)+e = V(4)  (m/s);

        # Nondimensional numbers for Negative/positive electrode.
        Pe_neg = u0_neg * ds_neg / D2
        Pe_pos = u0_pos * ds_pos / D5
        Re_neg = u0_neg * ds_neg * rho_neg / mu_neg
        Re_pos = u0_pos * ds_pos * rho_pos / mu_pos
        Sc_neg = mu_neg / rho_neg / D2
        Sc_pos = mu_pos / rho_pos / D5
        Sh_pos = []
        Sh_neg = []

        Ap = 2 * I_density_pos * Lh / F / NP / u0_pos / ds_pos
        Bp = 1 / 16 * I_density_pos * ds_pos / F / NP / D5
        Zep = gamma * Pe_pos * ds_pos
        An = 2 * I_density_neg * Lh / F / NE / u0_neg / ds_neg
        Bn = 1 / 16 * I_density_neg * ds_neg / F / NE / D2
        Zen = gamma * Pe_neg * ds_neg

        # Switch for including Donnan effect.
        addDonnan = 0
        if self.DonnanMode.lower() == 'yes':
            addDonnan = 1

        # Switch for using 0D or 2D model to calculate concentrations.
        isign_2D = 1                                                           # 1 for 2D model, and 0 for 0D model.
        if self.ZeroD.lower() == 'yes' or self.ZeroD.lower() == 'true':        # If isign_2D = 0, then our formula is equivalent to the 0D model.
            isign_2D = 0

        # Extra potential losses.
        # Over_potential_ohmic = 2 * I0 * Lc / Sigma_c + I0 * Lm / Sigma_m + \
        #                         I0 * Lt / eps_pos**1.5 / (Sigma_e_pos * beta) + \
        #                         I0 * Lt / eps_neg**1.5 / (Sigma_e_neg * beta)  # Ohmic losses.
        
        if self.OhmicMode.lower() == 'ElectrolyteOnly'.lower():
            Over_potential_ohmic = 2*I0*Lc/Sigma_c + I0*Lm/Sigma_m + \
                I0*Lt/eps_pos**1.5/(Sigma_el_pos*beta) + \
                    I0*Lt/eps_neg**1.5/(Sigma_el_neg*beta)
        elif self.OhmicMode.lower() == 'ElectrolyteElectrodePorosity'.lower():
            Over_potential_ohmic = 2*I0*Lc/Sigma_c + I0*Lm/Sigma_m + \
                I0*Lt/(eps_pos**1.5*Sigma_el_pos*beta + (1-eps_pos)**1.5*Sigma_ed_pos*beta) + \
                    I0*Lt/(eps_neg**1.5*Sigma_el_neg*beta + (1-eps_neg)**1.5*Sigma_ed_neg*beta)
        elif self.OhmicMode.lower() == 'ElectrolyteElectrodeEffective'.lower():
            Over_potential_ohmic = 2*I0*Lc/Sigma_c + I0*Lm/Sigma_m + \
                I0*Lt/(eps_pos**1.5*Sigma_el_pos*beta + Sigma_ed_pos*beta) + \
                    I0*Lt/(eps_neg**1.5*Sigma_el_neg*beta + Sigma_ed_neg*beta)
        elif self.OhmicMode.lower() == 'ElectrodeOnly'.lower():
            Over_potential_ohmic = 2*I0*Lc/Sigma_c + I0*Lm/Sigma_m + \
                I0*Lt/eps_pos**1.5/(Sigma_ed_pos*beta) + \
                    I0*Lt/eps_neg**1.5/(Sigma_ed_neg*beta)
        elif self.OhmicMode.lower() == 'ElectrodeOnly2'.lower():
            Over_potential_ohmic = 2*I0*Lc/Sigma_c + I0*Lm/Sigma_m +\
                I0*Lt/(1-eps_pos)**1.5/(Sigma_ed_pos*beta) + \
                    I0*Lt/(1-eps_neg)**1.5/(Sigma_ed_neg*beta);  
        else:
            raise ValueError("Invalid ohmic model. Supported values are 'ElectrolyteOnly',\
                              'ElectrolyteElectrodePorosity, 'ElectrolyteElectrodeEffective',\
                                  'ElectrodeOnly'  and 'ElectrodeOnly2'.")
            
        # Different models for mass transfer coefficients.
        if self.MassTransferMode.lower() == 'none':
            km_neg = float('inf')
            km_pos = float('inf')
        elif self.MassTransferMode.lower() == 'schmal1986':
            km_neg = 1.6e-4 * u0_neg**0.4
            km_pos = 1.6e-4 * u0_pos**0.4
        elif self.MassTransferMode.lower() == 'youd2017':
            km_neg = 8.85e-4 * u0_neg**0.9
            km_pos = 8.85e-4 * u0_pos**0.9
        elif self.MassTransferMode.lower() == 'chao2022v':
            km_neg = 8.9e-5 * u0_neg**0.5232
            km_pos = 6.7e-4 * u0_pos**0.8987
        elif self.MassTransferMode.lower() == 'chao2022dhps':
            km_neg = 1.9e-3 * u0_neg**1.004
            km_pos = 1.1e-3 * u0_pos**0.9192
        elif self.MassTransferMode.lower() == 'youx2017':
            Sh_pos = 1.68 * Re_pos**0.9
            Sh_neg = 1.68 * Re_neg**0.9
            km_pos = Sh_pos * D5 / df
            km_neg = Sh_neg * D2 / df
        elif self.MassTransferMode.lower() == 'chen2020':
            km_neg = u0_neg / ((5 / 16 - 2 * g) * Pe_neg + 2 * Lh / ds_neg - 4 / (3 * Pe_neg))
            km_pos = u0_pos / ((5 / 16 - 2 * g) * Pe_pos + 2 * Lh / ds_pos - 4 / (3 * Pe_pos))
        elif self.MassTransferMode.lower() == 'chen2020diffusiononly':
            km_neg = 16 / 5 * D2 / ds
            km_pos = 16 / 5 * D5 / ds
        elif self.MassTransferMode.lower() == 'bartonflowthrough':
            Sh_pos = 0.004 * Pe_pos**0.75 * Sc_pos**-0.24
            Sh_neg = 0.004 * Pe_neg**0.75 * Sc_neg**-0.24
            km_pos = Sh_pos * D5 / df
            km_neg = Sh_neg * D2 / df
        elif self.MassTransferMode.lower() == 'bartonflowby':
            Sh_pos = 0.018 * Pe_pos**0.68 * Sc_pos**-0.18
            Sh_neg = 0.018 * Pe_neg**0.68 * Sc_neg**-0.18
            km_pos = Sh_pos * D5 / df
            km_neg = Sh_neg * D2 / df
        else:
            raise ValueError("Invalid MassTransferMode. Supported values are 'none', 'schmal1986', 'youd2017','chao2022v', 'chao2022dhps', 'youx2017', 'bartonflowby', 'bartonflowthrough', 'chen2020diffusiononly' and 'chen2020'.")
            
        # Note: The following parameters I_{xx} differ from that in cellVoltage.m.
        if self.Mode.lower() == 'multiple':
            Ick_neg = I_density_neg / F / NE / k1
            Icd_neg = I_density_neg / F / NE / (D2 / ds_neg)
            Icu_neg = I_density_neg / F / NE / u0_neg
            Ick_pos = I_density_pos / F / NP / k2
            Icd_pos = I_density_pos / F / NP / (D5 / ds_pos)
            Icu_pos = I_density_pos / F / NP / u0_pos
            dc_neg = abs(-I_density_neg / km_neg / F / NE) * isign_2D          # Estimation of concentration difference between surface and bulk.
            dc_pos = abs(-I_density_pos / km_pos / F / NP) * isign_2D
        else:
            Ick_neg = I_density_neg / F / NE / (C20 + C30) / k1
            Icd_neg = I_density_neg / F / NE / (C20 + C30) / (D2 / ds_neg)
            Icu_neg = I_density_neg / F / NE / (C20 + C30) / u0_neg
            Ick_pos = I_density_pos / F / NP / (C40 + C50) / k2
            Icd_pos = I_density_pos / F / NP / (C40 + C50) / (D5 / ds_pos)
            Icu_pos = I_density_pos / F / NP / (C40 + C50) / u0_pos
            Pe_neg = u0_neg * ds_neg / D2
            Pe_pos = u0_pos * ds_neg / D5
            dc_neg = abs(-I_density_neg / km_neg / F / NE) / (C20 + C30) * isign_2D  # Estimation of concentration difference between surface and bulk.
            dc_pos = abs(-I_density_pos / km_pos / F / NP) / (C40 + C50) * isign_2D
            
        # Calculating for electrode permeablity.
        K_pos = 1/180*df_pos**2*eps_pos**3/(1-eps_pos)**2
        K_neg = 1/180*df_neg**2*eps_neg**3/(1-eps_neg)**2

        Calculated = {
            'ElectrodeVolume': VE,
            'ElectrodeArea': Ae,
            'CurrentDensityNominal': I0,
            'Donnan': addDonnan,
            'TwoD': isign_2D,
            'ElectronicConductivityMembrane': Sigma_m,
            'PoreSize': np.array([ds_pos, ds_neg]),
            'FiberSize':  np.array([df_pos, df_neg]),
            'Porosity':  np.array([eps_pos, eps_neg]),
            'ElectrodePermeability': np.array([K_pos,K_neg]),
            'SpecificAreaBase':  np.array([A_pos_base, A_neg_base]),
            'SpecificArea':  np.array([A_pos, A_neg]),
            'ChannelNumber':  np.array([N_channel_pos, N_channel_neg]),
            'CurrentDensityLocal':  np.array([I_density_pos, I_density_neg]),
            'ElectrolyteVelocity':  np.array([u0_pos, u0_neg]),
            'ElectrolyteTime':  np.array([tau_pos, tau_neg]),
            'ElectrolyteTimeRate':  np.array([eps0_pos, eps0_neg]),
            'ElectrodeTankRatio': np.array([delta_pos, delta_neg]),
            'Ick':  np.array([Ick_pos, Ick_neg]),
            'Icu':  np.array([Icu_pos, Icu_neg]),
            'Icd': np.array([Icd_pos, Icd_neg]),
            'Pe':  np.array([Pe_pos, Pe_neg]),
            'Dc':  np.array([dc_pos, dc_neg]),
            'OhmicLossCharge': Over_potential_ohmic,
            'OhmicLossDischarge': -Over_potential_ohmic,
            'MassTransferRate':  np.array([km_pos, km_neg]),
            'MassTransportCoefficient':  np.array([beta_pos, beta_neg]),
            'RectionRateConstant':  np.array([k2, k1]),                        # Positive (k2) and negative (k1).
            'Re':  np.array([Re_pos, Re_neg]),                                 # Calculated Reynolds number in positive.
            'Sc':  np.array([Sc_pos, Sc_neg]),                                 # Calculated Schmidt number in positive.
            'Sh':  np.array([Sh_pos, Sh_neg]),                                 # Calculated Schmidt number in positive.
            'ElectrodeConductivity':  np.array([Sigma_ed_pos, Sigma_ed_neg]),
            'ElectrolyteConductivity':  np.array([Sigma_el_pos, Sigma_el_neg]),
            'ModelConstantsPositive':  np.array([Ap, Bp]),
            'ModelConstantsNegative':  np.array([An, Bn]),
            'EntranceLengthPositive':  np.array([Zep, Zen])
        }
        
        # Order V4-5-3-2.
        beta = np.array(Calculated['MassTransportCoefficient'])[np.array([0, 0, 1, 1])].T  # Mass transfer correction coeff for V4/5/3/2.
        sa = np.array([-1, 1, -1, 1])
        sb = np.array([Zep, Zep, Zen, Zen]) / Lh
        
        Cka = sa * np.array([Bp, Bp, Bn, Bn]) * 136 / 35
        Ckb = sa * (np.array([Ap, Ap, An, An]) * (1/2 - 1/2 * sb**2) + np.array([Bp, Bp, Bn, Bn]) * (16/5 + 9/5 * sb))
        Ckc = sa * np.array([Bp, Bp, Bn, Bn]) * 5
        Ckd = sa * (np.array([Ap, Ap, An, An]) * (1 - sb) + np.array([Bp, Bp, Bn, Bn]) * 5)
        
        Calculated['Cka'] = obj2arr(Cka / beta, float)
        Calculated['Ckb'] = obj2arr(Ckb / beta, float)
        Calculated['Ckc'] = obj2arr(Ckc / beta, float)
        Calculated['Ckd'] = obj2arr(Ckd / beta, float)
        return Calculated