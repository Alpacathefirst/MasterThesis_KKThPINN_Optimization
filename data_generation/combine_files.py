import pandas as pd

# Load the datasets
vle_df = pd.read_csv("data_files/VLE_H_dataset")
v_df = pd.read_csv("data_files/V_H_dataset")
l_df = pd.read_csv("data_files/L_H_dataset")

# Combine the datasets
combined_df = pd.concat([vle_df, v_df, l_df], ignore_index=True)

combined_df.to_csv("data_files/combined_dataset", index=False)
