# Re-import libraries after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt
import math

# Load the log-scaled data
# df = pd.read_csv("data_files/VLE_H_dataset")
df = pd.read_csv(r"C:\Users\caspe\PycharmProjects\NN_optimization\data_preprocessing\scaled_data\VLE_H_dataset_scaled")

# Define column names
NN_INPUTS = ['T', 'P', 'CO2(g)', 'N2(g)', 'NaOH(aq)']
NN_OUTPUTS = ['H2O(g)', 'N2(g).1', 'CO2(aq)', 'N2(aq)', 'HCO3-', 'CO3-2', 'OH-', 'Na+', 'NaOH(aq).1', 'enthalpy', 'vapor fraction']

# Print min/max values
print("Input Ranges:")
for col in NN_INPUTS:
    print(f"{col}: min = {df[col].min()}, max = {df[col].max()}")

print("\nOutput Ranges:")
for col in NN_OUTPUTS:
    print(f"{col}: min = {df[col].min()}, max = {df[col].max()}")

input_cols = df.columns[:5]
output_cols = df.columns[5:]

# Plot histograms (frequency over value range) for each input column
n_cols = 2
n_rows = math.ceil(len(input_cols) / n_cols)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
axs = axs.flatten()

for i, col in enumerate(input_cols):
    axs[i].hist(df[col], bins=100)
    axs[i].set_title(f"{col}")
    # axs[i].set_xlabel("Scaled Value")
    axs[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# Plot histograms (frequency over value range) for each output column
n_cols = 2
n_rows = math.ceil(len(output_cols) / n_cols)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
axs = axs.flatten()

for i, col in enumerate(output_cols):
    axs[i].hist(df[col], bins=100)
    axs[i].set_title(f"{col}")
    # axs[i].set_xlabel("Scaled Value")
    axs[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


