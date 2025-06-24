import numpy as np
import matplotlib.pyplot as plt

# Reload data after code execution reset
train_losses = np.load("data/learning_curves/plant/NN/0.2/MODELID_train_losses_run0.npy")
val_losses = np.load("data/learning_curves/plant/NN/0.2/MODELID_val_losses_run0.npy")
train_violations = np.load("data/learning_curves/plant/NN/0.2/MODELID_train_violations_run0.npy")
val_violations = np.load("data/learning_curves/plant/NN/0.2/MODELID_val_violations_run0.npy")

# Plot losses
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot violations
plt.figure()
plt.plot(train_violations, label='Train Violation')
plt.plot(val_violations, label='Validation Violation')
plt.title('Constraint Violation over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Violation')
plt.legend()
plt.grid(True)
plt.show()