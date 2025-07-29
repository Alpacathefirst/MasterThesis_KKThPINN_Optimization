import numpy as np
import matplotlib.pyplot as plt

plt.figure()

for model in ['NN', 'PINN', 'KKThPINN', 'ECNN']:
    # Reload data after code execution reset
    train_losses = np.load(f"data\\learning_curves\\flash/{model}\\0.2\\FINAL_EXPERIMENT_train_losses_run0.npy")
    val_losses = np.load(f"data\\learning_curves\\flash\\{model}\\0.2\\FINAL_EXPERIMENT_val_losses_run0.npy")
    train_violations = np.load(f"data\\learning_curves\\flash\\{model}\\0.2\\FINAL_EXPERIMENT_train_violations_run0.npy")
    val_violations = np.load(f"data\\learning_curves\\flash\\{model}\\0.2\\FINAL_EXPERIMENT_val_violations_run0.npy")

# train_losses = np.load("data/learning_curves/flash/ECNN/0.2/MODELID_train_losses_run0.npy")
# val_losses = np.load("data/learning_curves/flash/ECNN/0.2/MODELID_val_losses_run0.npy")
# train_violations = np.load("data/learning_curves/flash/ECNN/0.2/MODELID_train_violations_run0.npy")
# val_violations = np.load("data/learning_curves/flash/ECNN/0.2/MODELID_val_violations_run0.npy")

    # # Plot losses
    # plt.plot(train_losses, label=f'Train Loss {model}')
    # # plt.plot(val_losses, label=f'Validation Loss {model}')
    # plt.title('Loss over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # Plot violations
    plt.plot(train_violations, label=f'Train Violation {model}')
    # plt.plot(val_violations, label='Validation Violation')
    plt.title('Constraint Violation over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Violation')
    plt.legend()
    #

plt.grid(True)
plt.show()