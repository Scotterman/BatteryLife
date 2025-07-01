import pandas as pd
import matplotlib.pyplot as plt

# Load saved CSV
df = pd.read_csv('checkpoints/CPTransformer_sl1_lr5e-05_dm128_nh4_el12_dl0_df256_lradjconstant_datasetTongji_lossMSE_wd0.0_wlFalse_bs16_s2021-CPTransformer/predictions_vs_truth.csv')

# Plot predictions vs. ground truth
plt.figure(figsize=(10, 6))
plt.plot(df['cycle'], df['ground_truth'], label='Ground Truth', linewidth=2)
plt.plot(df['cycle'], df['prediction'], label='Prediction', linestyle='--')
plt.xlabel('Cycle')
plt.ylabel('State of Health (SoH) or Capacity')
plt.title('Model Predictions vs Ground Truth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_plot.png', dpi=300)
plt.show()
