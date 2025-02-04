import pandas as pd
from matplotlib import pyplot as plt

eval_csv_filename = 'timing_test.csv'
df = pd.read_csv(eval_csv_filename)

generation_avg = df['Generation Time (s)'].mean()
stitching_avg = df['Stitching Time (s)'].mean()

df['Index'] = range(1, len(df) + 1)

plt.figure(figsize=(10, 6))

plt.plot(df['Index'], df['Stitching Time (s)'], label='Stitching Time', linestyle='-', color='r', alpha=0.7)
plt.plot(df['Index'], df['Generation Time (s)'], label='Generation Time', linestyle='-', color='b',alpha=0.7)

plt.axhline(y=generation_avg, color='b', linestyle='--', label=f'Avg Generation Time: {generation_avg:.4f}s')
plt.axhline(y=stitching_avg, color='r', linestyle='--', label=f'Avg Stitching Time: {stitching_avg:.4f}s')

plt.xlabel('Test Index')
plt.ylabel('Time (seconds)')
plt.title('Generation and Stitching Times per Method')

plt.legend()

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.show()
