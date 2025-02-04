import pandas as pd
import matplotlib.pyplot as plt

csv_filename = 'test_eval.csv'
df = pd.read_csv(csv_filename)

epoch_averages = df.groupby('Epoch')[['MSE', 'PSNR']].mean()

best_epoch_psnr = epoch_averages['PSNR'].idxmax()
best_psnr_value = epoch_averages['PSNR'].max()
best_epoch_mse = epoch_averages['MSE'].idxmin()
best_mse_value = epoch_averages['MSE'].min()

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(epoch_averages.index, epoch_averages['MSE'], label=f'MSE (Best: {best_mse_value:.6f})', color='g', marker='o', markersize=6)
plt.axvline(best_epoch_mse, color='r', linestyle='--', label=f'Best Epoch (MSE): {best_epoch_mse}')
plt.title('Average MSE over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend(loc='best')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5)

plt.subplot(1, 2, 2)
plt.plot(epoch_averages.index, epoch_averages['PSNR'], label=f'PSNR (Best: {best_psnr_value:.2f} dB)', color='b', marker='o', markersize=6)
plt.axvline(best_epoch_psnr, color='r', linestyle='--', label=f'Best Epoch (PSNR): {best_epoch_psnr}')
plt.title('Average PSNR over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.show()

