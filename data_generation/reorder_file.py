import pandas as pd

# Load the dataset
file_path = "data_files/VLE_H_dataset"
df = pd.read_csv(file_path)

# # Define a new arbitrary column order
# new_order = [
#     "T", "P", "CO2(g)", "N2(g)", "H2O(aq)", "NaOH(aq)", "CO2(g).1", "H2O(g)", "N2(g).1",
#     "HCO3-", "Na+", "CO2(aq)", "H2O(aq).1", "N2(aq)", "CO3-2", "OH-", "H+", "NaOH(aq).1", "enthalpy"
# ]

new_order = [
    "T", "P", "CO2(g)", "N2(g)", "H2O(aq)", "NaOH(aq)", "CO2(g).1", "N2(g).1", "H2O(aq).1",
    "HCO3-", "Na+", "H2O(g)", "CO2(aq)", "N2(aq)", "CO3-2", "OH-", "H+", "NaOH(aq).1", "enthalpy"
]

# Reorder the DataFrame
df_reordered = df[new_order]

# Save the reordered DataFrame
output_path = "data_files/VLE_H_dataset_reordered_2"
df_reordered.to_csv(output_path, index=False)
