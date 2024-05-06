### 2015-2023 plotting histograms for 2015-2023. Subplots for each year.

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'                      #Making figure prettier for latex usage.
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 20,
    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 32})
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import itertools
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator



years= [str(year) for year in range(2015,2023)]

# Reading all files from a given folder
base_folder_path = r'your_folder'#+ year + '/'  # Replace with the actual folder path
file_prefix = 'cumulative_'
#file_suffix = year + '.csv'

# List all files in the folder that match the file prefix and suffix
file_list = [file for file in os.listdir(base_folder_path) if file.startswith(file_prefix)] #and file.endswith(file_suffix)
print(f'File list: {file_list}')

def create_histograms(df,folder_path,ax):
    # Filter rows where event_start is True. Now I only have the rows where the event starts. Aka, all my events.
    event_df = df.loc[df['event_start']].copy()
    event_df['Time'] = pd.to_datetime(event_df['Time'])
    event_df['nadir_time'] = pd.to_datetime(event_df['nadir_time'])

    #Filtering out time to nadir events over 25 seconds and under 2 seconds. Doing this for all my functions.
    # Calculate time difference between event start and nadir in seconds.
    event_df['time_to_nadir'] = (event_df['nadir_time'] - event_df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    event_df['time_to_nadir'] = event_df['time_to_nadir'].abs()

    event_df = event_df.loc[event_df['time_to_nadir'].between(2, 25)]
    # Extract hour from datetime
    event_df['hour'] = event_df['Time'].dt.hour

    # Calculate total number of events
    total_events = len(event_df)

    # Create histogram for hour
    ax.hist(event_df['hour'], bins=range(0, 25), edgecolor='black')
    ax.set_xlabel(r'Hour')
    ax.set_xticks([0, 12, 24])



def create_histograms_2015_2023(df,folder_path,ax_big):
    # Filter rows where event_start is True. Now I only have the rows where the event starts. Aka, all my events.
    event_df = df.loc[df['event_start']].copy()


    #Filtering out time to nadir events over 25 seconds and under 2 seconds. Doing this for all my functions.
    
    # Calculate time difference between event start and nadir in seconds.
    event_df['time_to_nadir'] = (event_df['nadir_time'] - event_df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    event_df['time_to_nadir'] = event_df['time_to_nadir'].abs()

    event_df = event_df.loc[event_df['time_to_nadir'].between(2, 25)]

    # Extract hour from datetime
    event_df['hour'] = event_df['Time'].dt.hour

    # Save the DataFrame to a CSV file.
    event_df.to_csv(f'{os.path.dirname(folder_path)}/histogram number of events vs hours cumulative 2015-2023.pdf', index=False)

    # Calculate total number of events
    total_events = len(event_df)

    # Create histogram for hour
    ax_big.hist(event_df['hour'], bins=range(0, 25), edgecolor='black')
    ax_big.set_title(r'Events occuring by hour 2015 - 2023',size= 25)
    ax_big.set_xlabel(r'Hour',size=22)
    ax_big.set_ylabel(r'Number of events',size=22)
    ax_big.set_xticks([0, 12, 24]) # Set x-axis ticks to show all hour stamps
    #plt.savefig(f'{os.path.dirname(folder_path)}/histogram number of events vs hours cumulative 2015-2023.pdf')
    #plt.show()



def create_rocof_histogram(df):
    # Filter rows where event_start is True.
    event_df = df.loc[df['event_start']].copy()

    # Calculate time difference between event start and nadir in seconds.
    event_df['time_to_nadir'] = (event_df['nadir_time'] - event_df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    event_df['time_to_nadir'] = event_df['time_to_nadir'].abs()

    # Filter out time to nadir events over 25 seconds and under 2 seconds.
    event_df = event_df.loc[event_df['time_to_nadir'].between(2, 25)]

    # Create a new column 'rocof_range' that categorizes the max_rocof value into different ranges.
    bins = [40, 100, 160, 220, 280]
    labels = ['40-100', '100-160', '160-220', '220-280']
    event_df['rocof_range'] = pd.cut(event_df['max_rocof'], bins=bins, labels=labels)

    # Group by year and rocof_range, and count the number of events in each group.
    event_counts = event_df.groupby([event_df['Time'].dt.year, 'rocof_range']).size().unstack(fill_value=0)

    # Create a bar chart with years on the x-axis and number of events on the y-axis, with different colors for different rocof_range groups.
    event_counts.plot(kind='bar', stacked=True, color=['tab:blue', 'green', 'yellow', 'darkred'],edgecolor='black',linewidth=1.2,width=0.8)
    plt.xticks(range(0, 9),rotation=0)  # Set x-axis ticks to show all hour stamps
    plt.yticks(range(0, 70, 15))
    plt.tick_params(axis='x', labelsize=17, pad=5, length=5)
    plt.xlabel(r'Year')
    plt.ylabel(r'Number of events')
    plt.title(r'Events sorted by Max RoCoF')
    plt.legend( fontsize='9') #title='RoCoF range', title_fontsize='8',
    plt.tight_layout()

xmax = 24
xmin = 0
ymax = 20
ymin = 0

# Set the x and y-axis limits
x_limits = [xmin, xmax]  # replace xmin and xmax with the desired values
y_limits = [ymin, ymax]  # replace ymin and ymax with the desired values

# Create a figure and a grid of subplots
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(3, 6)

# Create a big subplot for the main plot.
ax_big = fig.add_subplot(gs[:3, :3])
ax = [fig.add_subplot(gs[i//3, 3 + i%3]) for i in range(9)]

# Number of columns in the grid
num_cols = 3

# Set the x and y-axis limits for all subplots
for i,axis in enumerate(ax):
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    # Add the year above each subplot
    axis.set_title(f'{years[i]}', y=1,size= 19)
    
# Initialize an empty list to store all dataframes
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

    # Convert the Time and nadir_time columns to datetime objects
    combined_df['Time'] = pd.to_datetime(combined_df['Time'])
    combined_df['nadir_time'] = pd.to_datetime(combined_df['nadir_time'])

    # Append the combined_df of each year to all_dfs
    all_dfs.append(combined_df)

    # Call your function to create the subplot for the current year.
    create_histograms(combined_df,folder_path,ax[i])
    ax[i].set_xticks([0, 12, 24])
    ax[i].tick_params(axis='y', labelsize=20, pad=5, length=5)  # This line changes the y-axis label size
    ax[i].tick_params(axis='x', labelsize=20, pad=3, length=5)
    ymin, ymax = ax[i].get_ylim()
    ax[i].set_ylim(ymin, ymax * 0.95) 
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins=1))  # Add this line after setting the tick parameters
    ax[i].xaxis.set_major_locator(MaxNLocator(nbins=3))
    if i < 6:
        ax[i].set_xticklabels([])
        ax[i].set_xlabel('')
    ax[i].set_title(f'{year}', y=0.62, fontsize=20)
# Concatenate all dataframes in all_dfs to create a dataframe that includes data from all years
all_years_df = pd.concat(all_dfs)
#ax_big.legend(handlelength=1.1, handletextpad=.5, loc=2, fontsize=18)
fig.subplots_adjust(left=.075, bottom=.15, right=.99, top=.70,
                    hspace=0, wspace=0)
create_histograms_2015_2023(all_years_df,folder_path, ax_big) # Call your function to create the main plot. Using the combined dataframe of all years. Do not put this in the loop above.
ax_big.xaxis.set_major_locator(MaxNLocator(nbins=3))
ax_big.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax_big.tick_params(axis='both', labelsize=22, pad=3, length=5)
ax_big.set_xlim(x_limits)
# Reduce whitespace around the figure
plt.tight_layout()
plt.savefig(f'{os.path.dirname(base_folder_path)}/histogram number of events.pdf') 

fig_cumulative = plt.figure(figsize=(21, 7))
create_rocof_histogram(all_years_df)
#plt.tight_layout()
plt.savefig(f'{os.path.dirname(base_folder_path)}/histogram max rocof overview.pdf') 
