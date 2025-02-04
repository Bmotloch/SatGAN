import pandas as pd
import matplotlib.pyplot as plt

def plot_losses_with_custom_x(csv_file, window_size=50):
    try:
        data = pd.read_csv(csv_file)
        data['Custom_X'] = (data['Epoch'] - 1) * 562 + (data['Step'] - 1)

        data['Loss_G_MA'] = data['Loss_G'].rolling(window=window_size, min_periods=1).mean()
        data['Loss_Fake_MA'] = data['Loss_Fake'].rolling(window=window_size, min_periods=1).mean()
        data['Loss_Real_MA'] = data['Loss_Real'].rolling(window=window_size, min_periods=1).mean()
        data['Loss_D_MA'] = data['Loss_D'].rolling(window=window_size, min_periods=1).mean()

        plt.figure(figsize=(12, 8))

        plt.plot(data['Custom_X'], data['Loss_G'], label='Loss_G (Original)', color='orange', alpha=0.5)
        plt.plot(data['Custom_X'], data['Loss_Fake'], label='Loss_Fake (Original)', color='red', alpha=0.5)
        plt.plot(data['Custom_X'], data['Loss_Real'], label='Loss_Real (Original)', color='green', alpha=0.5)
        plt.plot(data['Custom_X'], data['Loss_D'], label='Loss_D (Original)', color='blue', alpha=0.5)

        plt.plot(data['Custom_X'], data['Loss_G_MA'], label='Loss_G (Moving Avg)', color='orange', linewidth=2)
        plt.plot(data['Custom_X'], data['Loss_Fake_MA'], label='Loss_Fake (Moving Avg)', color='red', linewidth=2)
        plt.plot(data['Custom_X'], data['Loss_Real_MA'], label='Loss_Real (Moving Avg)', color='green', linewidth=2)
        plt.plot(data['Custom_X'], data['Loss_D_MA'], label='Loss_D (Moving Avg)', color='blue', linewidth=2)

        plt.title(f"Losses During Training ({len(data['Epoch'].unique())} epochs)", fontsize=14)
        plt.xlabel('Steps (batch size 16)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file}")
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

plot_losses_with_custom_x('training_log_big.csv', window_size=50)