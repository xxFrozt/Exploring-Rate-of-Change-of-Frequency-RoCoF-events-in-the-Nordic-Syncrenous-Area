# This code makes a heatmap which shows in which month are missing more data. 
# The code is using a .csv file which have been manually inputted with the missing data in percent for each month of each year.
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 20,
    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True
from matplotlib.colors import LogNorm
#Heatmap of missing data
import matplotlib.pyplot as plt
import os

file = r'your_folder'
file_name = "missing_data_overview.csv"  # Include the file extension

df = pd.read_csv(r'your_folder.csv', index_col=0)

# Transpose the DataFrame
df = df.T

# Create a logarithmic color scale
log_norm = LogNorm(vmin=df.min().min(), vmax=df.max().max())

plt.figure(figsize=(10, 8))
ax = sns.heatmap(df, annot=False, norm=log_norm, cmap='YlGnBu_r')

ax.set(xlabel='Year', ylabel='Months')
cbar = ax.collections[0].colorbar
cbar.set_label('Percentage of Missing Data')
plt.title('Missing Data in FinGrid Dataset')

# Save the plot to the same directory as the data file
file_path = r'C:\Users\Tore Tang\Data FinGrid clean\missing_data_overview.csv'
directory = os.path.dirname(file_path)
file_name = os.path.basename(file_path)
file_name_without_extension = os.path.splitext(file_name)[0]
plt.tight_layout()
plt.savefig(os.path.join(directory, f'{file_name_without_extension}_heatmap.pdf'))

plt.show()
