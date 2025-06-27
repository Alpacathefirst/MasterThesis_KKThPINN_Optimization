from data_generation_functions import *

# Create file name
FILENAME = f'VLE_H_dataset'

# Select NN inputs, T[K], P[bar], Species[mol/mol],
# H2O will be calculated as 1-z_i
# N2(g) will be calculated as N2_value*x_CO2
NN_INPUTS = ['T', 'P', 'CO2(g)', 'N2(g)', 'NaOH(aq)']
NN_ALL_INPUTS = ['T', 'P', 'CO2(g)', 'N2(g)', 'H2O(aq)', 'NaOH(aq)']
# The specie inputs are converted to element inputs if this is set True
ELEMENTS = False
# input ranges
INPUT_RANGES = {'T': [293, 473], 'P': [1, 170], 'CO2(g)': [0, 0.97], 'N2(g)': [0, 0.001], 'NaOH(aq)': [0, 0.02]}
# Select NN outputs, Species[mol/mol]
NN_OUTPUTS = ['CO2(g)', 'H2O(g)', 'N2(g)', 'CO2(aq)', 'H2O(aq)', 'N2(aq)', 'HCO3-', 'CO3-2', 'OH-', 'H+', 'Na+', 'NaOH(aq)', 'enthalpy']
OUTPUT_MAX = {'H2O(g)': 0.6, 'N2(g)': 0.002, 'CO2(aq)': 0.025, 'N2(aq)': 5e-7, 'HCO3-': 0.02, 'CO3-2': 0.0002,
              'OH-': 1e-5, 'Na+': 0.02, 'NaOH(aq)': 1e-5}
# Number of data points
N_DATAPOINTS = 80000

# Choose Reaktoro Database
DATABASE = rkt.SupcrtDatabase.withName("supcrtbl-organics")
# Define Gas and Aqueous Phase
GAS_PHASE = ['CO2(g)', 'H2O(g)', 'N2(g)']
AQUEOUS_PHASE = ['H2O(aq)', 'CO2(aq)', 'N2(aq)', 'HCO3-', 'CO3-2', 'OH-', 'H+', 'NaOH(aq)', 'Na+']
# Set the EOS, the custom bip is needed to set the binary interaction parameter (BIP) for Redlich Kwong
params_suprcrt = rkt.CubicEOS.BipModelParamsCustomRedlichKwong()
params_suprcrt.kH2O_CO2 = 0.1249


def custom_bip_suprcrt(specieslist):
    formulas = [str(s.formula()) for s in specieslist]
    return rkt.CubicEOS.BipModelCustomRedlichKwong(formulas, params_suprcrt)


GAS_MODEL = rkt.ActivityModelRedlichKwong(custom_bip_suprcrt)
# # set EOS model, rkt.CubicBipModelPhreeqc contains the binary interaction parameters for the PR EOS
# GAS_MODEL = rkt.ActivityModelPengRobinson(rkt.CubicBipModelPhreeqc())
# set Activity model
AQUEOUS_MODEL = rkt.ActivityModelPitzer()

# create a
data_generator = DataGenerator(filename=FILENAME, inputs=NN_INPUTS, all_inputs=NN_ALL_INPUTS, outputs=NN_OUTPUTS,
                               db=DATABASE, max_outputs=OUTPUT_MAX, allowed_phases='vle')
data_generator.handle_reaktoro(gas_phase=GAS_PHASE, gas_model=GAS_MODEL,
                               aqueous_phase=AQUEOUS_PHASE, aqueous_model=AQUEOUS_MODEL)
data_generator.generate_inputs(input_ranges=INPUT_RANGES, n_datapoints=N_DATAPOINTS)
data_generator.generate_data()
