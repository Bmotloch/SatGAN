import pandas as pd
import matplotlib.pyplot as plt


def plot_losses_with_custom_x(csv_file):
    try:
        data = pd.read_csv(csv_file)
        data['Custom_X'] = (data['Epoch'] - 1) * 562 + (data['Step'] - 1)

        plt.figure(figsize=(12, 8))
        plt.plot(data['Custom_X'], data['Loss_G'], label='Loss_G', color='orange', alpha=0.7)
        plt.plot(data['Custom_X'], data['Loss_Fake'], label='Loss_Fake', color='red', alpha=0.7)
        plt.plot(data['Custom_X'], data['Loss_Real'], label='Loss_Real', color='green', alpha=0.7)
        plt.plot(data['Custom_X'], data['Loss_D'], label='Loss_D', color='blue', alpha=0.7)

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


plot_losses_with_custom_x('training_log_big.csv')
