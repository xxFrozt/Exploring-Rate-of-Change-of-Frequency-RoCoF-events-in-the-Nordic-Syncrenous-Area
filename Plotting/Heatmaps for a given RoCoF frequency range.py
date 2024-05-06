#Heatmap

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'                    #making figure prettier in latex.
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 20,
    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import itertools
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


years= [str(year) for year in range(2015,2024)]

# Reading all files from a given folder
base_folder_path = r'your_folder'#+ year + '/'  # Replace with the actual folder path
file_prefix = 'rocofevents_'


def create_heatmap(df,min_rocof,max_rocof,folder_path,min_rocof_string,subplot,ylabel,cbar_label=True): #,min_rocof,max_rocof
        
    # Get the min and max year from the original data
    df['Year'] = df['Time'].dt.year
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    
    # Filter the data
    df = df.loc[df['event_start']].copy()

    #Filtering out time to nadir events over 25 seconds and under 2 seconds. Doing this for all my functions.
    # Calculate time difference between event start and nadir in seconds.
    df['time_to_nadir'] = (df['nadir_time'] - df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    df['time_to_nadir'] = df['time_to_nadir'].abs()

    df = df.loc[df['time_to_nadir'].between(2, 25)]

    #print(df)
    df['max_rocof'] = df['max_rocof'].abs()
    df = df.loc[df['max_rocof'].between(min_rocof, max_rocof)]

    df['Month'] = df['Time'].dt.month

    # Group by year and month and count the number of events
    event_counts = df.groupby(['Year', 'Month']).size().reset_index(name='Count')

    # Create a blank heatmap
    blank_heatmap_data = pd.DataFrame(np.zeros((12, max_year - min_year + 1)), index=np.arange(1, 13), columns=np.arange(min_year, max_year + 1))

    # Fill in the data
    for index, row in event_counts.iterrows():
        blank_heatmap_data.loc[row['Month'], row['Year']] = row['Count']

    # Replace the numeric month indices with their corresponding names
    blank_heatmap_data.index = blank_heatmap_data.index.map({1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})    
    
    # Plot the heatmap
    sns.heatmap(blank_heatmap_data, fmt="d", cmap='YlGnBu',annot_kws={"size": 10},cbar_kws={'label': 'Number of Events' if cbar_label else ''}, ax=subplot,)
    
    subplot.set_title(fr'Max RoCoF between {min_rocof_string} - {max_rocof} $\varphi$ [mHz{{$\cdot$}}s$^{{-1}}$]')
   # r'$\varphi$ [mHz{$\cdot$}s$^{-1}$]'
    # Change the size of the y and x-axis text
    subplot.tick_params(axis='y', labelsize=22)
    subplot.tick_params(axis='x', labelsize=19)
    subplot.set_xlabel('Year',labelpad=3, fontsize=24)
    if ylabel:
        subplot.set_ylabel('Month',labelpad=3, fontsize=26)
    else:
        subplot.set_ylabel('')



all_dfs = []
for i, year in enumerate(years):
    # Update folder path and file suffix for the current year
    folder_path = base_folder_path + year + '/'
    file_suffix = year + '.csv'

    # List all files in the folder that match the file prefix and suffix
    file_list = [file for file in os.listdir(folder_path) if file.startswith(file_prefix) and file.endswith(file_suffix)]
    print(f'File list: {file_list}')

    # Read each file and combine them into one dataframe
    dfs = []
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    # Combine all dataframes in dfs into one dataframe
    combined_df = pd.concat(dfs)

    # Create a new column to indicate if the RoCoF is positive or negative
    combined_df['rocof_sign'] = np.where(combined_df['rocof'] >= 0, 'positive', 'negative')

    # Save the combined dataframe as a new csv file
    #output_file = os.path.join(folder_path, f'cumulative_rocofevents_{year}.csv')
    #combined_df.to_csv(output_file, index=False)

    # Convert the Time and nadir_time columns to datetime objects
    combined_df['Time'] = pd.to_datetime(combined_df['Time'])
    combined_df['nadir_time'] = pd.to_datetime(combined_df['nadir_time'])

    # Append the combined_df of each year to all_dfs
    all_dfs.append(combined_df)

# Concatenate all dataframes in all_dfs to create a dataframe that includes data from all years
all_years_df = pd.concat(all_dfs)
print(all_years_df)

# Define the intervals
intervals = [(40, 100, '40'), (100.00001, 160, '100'), (160.00001, 220, '160'), (220.00001, 280, '220')]

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(18.75, 15),gridspec_kw={'width_ratios': [3, 3], 'height_ratios': [3, 3]})

# Create a heatmap for each interval
for i, (interval, subplot) in enumerate(zip(intervals, axs.flatten())):
    min_rocof, max_rocof, min_rocof_string = interval
    ylabel = i % 2 == 0
    cbar_label = i % 2 == 1
    create_heatmap(all_years_df, min_rocof, max_rocof, base_folder_path, min_rocof_string, subplot,ylabel,cbar_label)
fig.subplots_adjust(wspace=.25)    
# Reduce whitespace around the figure
plt.tight_layout()
plt.savefig(f'{os.path.dirname(base_folder_path)}/heatmaps for different Max RoCoF values.pdf') 
plt.show()
