import onnxruntime as ort
import numpy as np
import joblib
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load ONNX model
session = ort.InferenceSession(r'C:\Users\caspe\PycharmProjects\KKThPINN_Optimization\models\flash_wo_electrolytes\KKThPINN\0.2\test_0.2_0.onnx')

df = pd.read_csv(r'C:\Users\caspe\PycharmProjects\KKThPINN_Optimization\data_generation\data_files\VLE_H_dataset_transformed')

scaler = joblib.load(r'C:\Users\caspe\PycharmProjects\KKThPINN_Optimization\models\flash\scalers\VLE_H_dataset')  # TODO: Save and load this more nicely

# Define columns
NN_inputs = ['T', 'P', 'C', 'H', 'O', 'Na', 'N']
NN_outputs = ['CO2_g_out', 'H2O_aq_out', 'O_elec', 'Na_elec', 'N2_g_out',
                 'CO2_aq_out', 'C_elec', 'H2O_g_out', 'H_elec', 'N2_aq_out', 'enthalpy_out']

# Prepare stats
mae_list = []
abs_errors = []
rel_errors = []

for _, row in df.iterrows():
    X = row[NN_inputs].to_numpy()
    Y_true = row[NN_outputs].to_numpy()

    X_scaled = X / scaler.scale_[:7]  # shape must match training input dim

    X_scaled = X_scaled.reshape(1, -1)
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    Y_scaled = session.run([output_name], {input_name: X_scaled})[0]
    XY = scaler.inverse_transform(np.concatenate([X_scaled, Y_scaled], axis=1))

    Y = XY[:, 7:]

    # Error metrics
    mae = mean_absolute_error(Y_true, Y[0])
    mae_list.append(mae)

    abs_error = np.abs(Y_true - Y[0])
    rel_error = abs_error / np.where(Y_true != 0, np.abs(Y_true), 1e-16) * 100

    abs_errors.append(abs_error)
    rel_errors.append(rel_error)


# Aggregate stats
abs_errors = np.array(abs_errors)
rel_errors = np.array(rel_errors)
mae_per_output = np.mean(abs_errors, axis=0)
mre_per_output = np.mean(rel_errors, axis=0)

# Print
print("\nMean Absolute Error (MAE) per output:")
for name, mae in zip(NN_outputs, mae_per_output):
    print(f"{name}: {mae:.4e}")

print("\nMean Relative Error (MRE) per output:")
for name, mre in zip(NN_outputs, mre_per_output):
    print(f"{name}: {mre:.1f}%")

print(f"\nOverall MAE: {np.mean(mae_per_output):.4e}")
print(f"Overall MRE: {np.mean(mre_per_output):.1f}%")
