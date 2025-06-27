import onnxruntime as ort
import numpy as np
import joblib
import torch

# Load ONNX model
session = ort.InferenceSession(r'C:\Users\caspe\PycharmProjects\KKThPINN_Optimization\models\flash\KKThPINN\0.2\MODELID_0.2_0.onnx')

# Prepare input
x_input = np.array([[437.0704970411541,79.72605682346513,0.4916948495159864,0.00021788172053848452,0.5054228453860665,0.0026644233774086286]], dtype=np.float64)
X = x_input

scaler = joblib.load(r'C:\Users\caspe\PycharmProjects\KKThPINN_Optimization\models\flash\scalers\VLE_H_dataset')
x_input = x_input / scaler.scale_[:6]  # shape must match training input dim


# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
y_output = session.run([output_name], {input_name: x_input})[0]
xy_output = scaler.inverse_transform(np.concatenate([x_input, y_output], axis=1))

y_output = xy_output[:, 6:]
# print("Predicted output:", y_output)
Y = y_output
error = y_output - np.array([0.484636,0.0595831,0.000217714,0.00439589,0.445841,1.67696e-07,0.00266127,1.53433e-06,8.61042e-08,7.31339e-09,0.00266441,1.03579e-08,-329937])
# print(error)
print(['CO2(g)', 'H2O(g)', 'N2(g)', 'CO2(aq)', 'H2O(aq)', 'N2(aq)', 'HCO3-', 'CO3-2', 'OH-', 'H+', 'Na+', 'NaOH(aq)', 'enthalpy'])
print('rel error (%)', np.round(error / y_output * 100).astype(int))


#
# print(X)
# print(Y)

# Define A (5×6) and B (5×13) as per your formulation
A = torch.tensor([
    [0, 0, 1, 0, 0, 0],    # Carbon balance
    [0, 0, 0, 0, 2, 1],    # Hydrogen balance
    [0, 0, 2, 0, 1, 1],    # Oxygen balance
    [0, 0, 0, 2, 0, 0],    # Nitrogen balance
    [0, 0, 0, 0, 0, 0],    # Charge (dropped Na row)
], dtype=torch.float64)

B = torch.tensor([
    [-1,  0,  0, -1,  0,  0, -1, -1,  0,  0,  0,  0,  0],   # Carbon
    [ 0, -2,  0,  0, -2,  0, -1,  0, -1, -1,  0, -1,  0],   # Hydrogen
    [-2, -1,  0, -2, -1,  0, -3, -3, -1,  0,  0, -1,  0],   # Oxygen
    [ 0,  0, -2,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0],   # Nitrogen
    [ 0,  0,  0,  0,  0,  0, -1, -2, -1,  1,  1,  0,  0],   # Charge
], dtype=torch.float64)

# Compute terms
AX = A @ X.T        # shape: (5, n_samples)
BY = B @ Y.T        # shape: (5, n_samples)
AX_plus_BY = AX + BY

# print(X)
# print(Y)
# print(AX)
# print(BY)
print('AX+BY', AX_plus_BY)

