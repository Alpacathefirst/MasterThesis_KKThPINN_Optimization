import os.path
import sys
sys.path.append(r"C:\Users\caspe\Reaktoro\Reaktoro_with_maingo\build\Reaktoro\Release")

# Autodiff import seems necessary to recognise autodiff input/outputs
from autodiff import *
import reaktoro4py as rkt
import numpy as np
from scipy.stats import qmc
import csv

from constants import *

# disable failed equilibrium warning from reaktoro
rkt.Warnings.disable(906)


class DataGenerator:
    def __init__(self, filename, inputs, all_inputs, outputs, db, max_outputs, allowed_phases):
        self.filepath = f"data_files/{filename}"
        if allowed_phases not in ['vle', 'l', 'v']:
            raise Exception('allowed phases can only be v, l, or vle')
        self.allowed_phases = allowed_phases  # either v, l or vle
        self.check_filename()
        self.input_names = inputs
        self.all_input_names = all_inputs
        self.output_names = outputs
        self.max_outputs = max_outputs
        self.database = db
        self.state = None
        self.vl_system = None
        self.vl_solver = None
        self.l_system = None
        self.l_solver = None
        self.g_system = None
        self.g_solver = None
        self.input_values = None
        self.column_headers = self.get_column_headers()
        print('header column', self.column_headers)  # print it so the program can be stopped if they are wrong
        self.gas_phase_species = None
        self.aqueous_phase_species = None
        self.gas_phase = None
        self.aqueous_phase = None
        self.input_data = []
        self.output_data = []
        self.molar_amount = 1  # used to scale create the system and scale enthalpy

    def check_filename(self):
        # implement here
        if os.path.exists(self.filepath):
            raise Exception('Filename already used, please create a unique one')

    def handle_reaktoro(self, gas_phase, aqueous_phase, gas_model, aqueous_model):
        self.gas_phase_species = gas_phase
        self.aqueous_phase_species = aqueous_phase
        g = rkt.GaseousPhase(gas_phase)
        l = rkt.AqueousPhase(aqueous_phase)
        # set EOS model
        g.set(gas_model)
        # set Activity model
        l.set(aqueous_model)

        self.vl_system = rkt.ChemicalSystem(self.database, g, l)
        self.vl_solver = rkt.EquilibriumSolver(self.vl_system)

        self.l_system = rkt.ChemicalSystem(self.database, l)
        self.l_solver = rkt.EquilibriumSolver(self.l_system)

        self.g_system = rkt.ChemicalSystem(self.database, g)
        self.g_solver = rkt.EquilibriumSolver(self.g_system)

    def generate_inputs(self, input_ranges, n_datapoints):
        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=len(self.input_names))
        sample = sampler.random(n=n_datapoints)

        lower_bounds = []
        upper_bounds = []

        for i in self.input_names:
            lower_bounds.append(input_ranges[i][0])
            upper_bounds.append(input_ranges[i][1])
        self.input_values = qmc.scale(sample, lower_bounds, upper_bounds)

    def handle_state(self, system, solver, input_vector):
        inp = [0] * len(self.all_input_names)
        # set T, P, species and save these values to input
        self.state = rkt.ChemicalState(system)
        molar_fraction_water = 1
        # assign inputs to reaktoro state
        for index, value in enumerate(input_vector):
            input_name = self.input_names[index]
            if input_name == 'T':
                self.state.temperature(value, 'K')
                self.save_input(name=input_name, value=value, inp=inp)
            elif input_name == 'P':
                self.state.pressure(value, 'bar')
                self.save_input(name=input_name, value=value, inp=inp)
            else:
                if input_name == "N2(g)":
                    value = value * input_vector[2]  # This assumes CO2(g) is third input!!!!!!!, z_n2 = value*z_CO2
                self.state.set(input_name, value * self.molar_amount, 'mol')
                self.save_input(name=input_name, value=value*self.molar_amount, inp=inp)
                molar_fraction_water -= value
        # add water to the system
        if molar_fraction_water <= 0:
            raise Exception('molfraction water < 0')
        # if there is a aqeous phase always initialise water as a liquid, if there is a gasphase only assign it to the
        # gas phase
        self.state.set('H2O(aq)', molar_fraction_water * self.molar_amount, 'mol')
        self.save_input(name='H2O(aq)', value=molar_fraction_water * self.molar_amount, inp=inp)

        result = solver.solve(self.state)
        return result.succeeded(), inp

    def save_input(self, name, value, inp):
        if name in self.all_input_names:
            index = self.all_input_names.index(name)
            inp[index] = value

    def save_output(self, inp):
        props = rkt.ChemicalProps(self.state)
        # calculate total moles of vapor and liquid phase
        n_vapor = 0
        for specie in self.gas_phase_species:
            n_vapor += self.state.speciesAmount(specie)
        n_liquid = 0
        for specie in self.aqueous_phase_species:
            n_liquid += self.state.speciesAmount(specie)
        n_total = n_vapor + n_liquid
        out = []
        if self.allowed_phases == 'v':
            inp, out = self.save_sat_vapor(inp, out, n_vapor)
        elif self.allowed_phases == 'l':
            inp, out = self.save_sat_liquid(inp, out, n_liquid)
        else:
            for entry in self.output_names:
                if entry in SPECIES:
                    try:
                        specie_molfraction = props.speciesMoleFraction(entry)
                        specie_amount = self.state.speciesAmount(entry)
                        # specie amount has to be smaller then 1e-15, otherwise a calculation with a disapearing phase will
                        # be discarded because it will have really high molfraction
                        if entry in self.max_outputs:
                            if specie_molfraction > self.max_outputs[entry]: # and specie_amount > 1e-15
                                # print(self.state)
                                # print(f'specie {specie}, molfrac = {specie_molfraction}, max is {self.max_outputs[specie]}')
                                return False
                        out.append(specie_amount)
                    except RuntimeError:
                        out.append(0)
                elif entry == 'enthalpy':
                    props = rkt.ChemicalProps(self.state)
                    enthalpy = props.enthalpy() / n_total  # to obtain enthalpy in J/mol
                    out.append(enthalpy)
        self.input_data.append(inp)
        self.output_data.append(out)
        return True

    def save_sat_vapor(self, inp, out, n_vap):
        props = rkt.ChemicalProps(self.state)
        for index, name in enumerate(self.output_names):
            out.append(0)
            if name in self.gas_phase_species:
                out[index] = props.speciesMoleFraction(name) * self.molar_amount
            elif name in self.aqueous_phase_species:
                out[index] = 0  # set liquid molfractions in the output to zero
            elif name == 'enthalpy':
                enthalpy = props.phaseProps("GaseousPhase").enthalpy()
                out[index] = enthalpy / n_vap  # get the enthalpy in J/mol
        for index, name in enumerate(self.all_input_names):
            if name in self.gas_phase_species:
                inp[index] = props.speciesMoleFraction(name) * self.molar_amount  # the inpuit and output are equal in the case of a vapor phase only
            elif name == 'H2O(aq)':
                inp[index] = props.speciesMoleFraction('H2O(g)') * self.molar_amount
            elif name in self.aqueous_phase_species:
                inp[index] = 0  # set liquid molfractions in the input to zero
        return inp, out

    def save_sat_liquid(self, inp, out, n_liq):
        props = rkt.ChemicalProps(self.state)
        for index, name in enumerate(self.output_names):
            out.append(0)
            if name in self.gas_phase_species:
                out[index] = 0  # set vapor molfractions to zero
            elif name in self.aqueous_phase_species:
                out[index] = props.speciesMoleFraction(name) * self.molar_amount
            elif name == 'enthalpy':
                enthalpy = props.phaseProps('AqueousPhase').enthalpy() / n_liq  # get the enthalpy in J/mol
                out[index] = enthalpy

        for index, name in enumerate(self.all_input_names):
            if name == 'CO2(g)':
                inp[index] = self.add_specie_amounts(['CO2(aq)', 'HCO3-', 'CO3-2'])
            elif name == 'N2(g)':
                inp[index] = props.speciesMoleFraction('N2(aq)')
            elif name == 'NaOH(aq)':
                inp[index] = self.add_specie_amounts(['NaOH(aq)', 'Na+'])
        return inp, out

    def add_specie_amounts(self, specie_list):
        props = rkt.ChemicalProps(self.state)
        total_amount = 0
        for specie in specie_list:
            total_amount += (props.speciesMoleFraction(specie) * self.state.speciesAmount(specie))
        return total_amount

    def set_vapor_fractions_to_zero(self, out):
        for index, name in enumerate(self.output_names):
            if name in self.gas_phase_species:
                out[index] = 0  # set molar amount of the gasphase species to 0
        return out

    def set_liquid_fractions_to_zero(self, out):
        for index, name in enumerate(self.output_names):
            if name in self.aqueous_phase_species:
                out[index] = 0  # set molar amount of the lioiquid species to 0
        return out

    def generate_data(self):
        failed_calculations = 0
        discarded_calculations = 0

        for input_vector in self.input_values:
            # set and calculate the state, only gas phase set to false
            success, nn_input = self.handle_state(self.vl_system, self.vl_solver, input_vector)
            # if the eq-calculation succeeded, add the data point to the input and output data
            if success:
                good_ouputs = self.save_output(nn_input)
                if not good_ouputs:
                    discarded_calculations += 1
            else:
                failed_calculations += 1

        print(f"Failed calculations: {failed_calculations}")
        print(f"Discarded calculations: {discarded_calculations}")

        # Save to CSV
        # Combine input and output
        combined_data = np.hstack((np.array(self.input_data), np.array(self.output_data)))

        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.column_headers)
            writer.writerows(combined_data)

        print(f"Data saved to {self.filepath}")

    def get_column_headers(self):
        column_headers = []
        for inp_name in self.all_input_names:
            if inp_name in ['T', 'P']:
                column_headers.append(inp_name)
            else:
                column_headers.append(inp_name)
        for output_name in self.output_names:
            column_headers.append(output_name)
        return column_headers
