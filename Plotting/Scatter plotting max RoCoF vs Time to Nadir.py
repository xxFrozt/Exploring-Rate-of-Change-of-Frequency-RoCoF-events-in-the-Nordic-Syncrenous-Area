#Creating histograms and fiolin plots for the data(yay!)

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 20,
    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
#Creating histograms and fiolin plots for the data(yay!)


#year = '2023'

years= [str(year) for year in range(2015,2024)]

# Reading all files from a given folder
base_folder_path = r'your_folder'#+ year + '/'  # Replace with the actual folder path
file_prefix = 'rocofevents_'                    
#file_suffix = year + '.csv'

def create_scatter_max_RoCoF_vs_time_to_nadir(df, folder_path,ax):
    # Create scatter plot for max RoCoF vs time to nadir

    ax.tick_params(axis='both', which='major', labelsize=10)

    event_df = df.loc[df['event_start']].copy()

    # Get absolute value of max_rocof
    event_df['max_rocof'] = event_df['max_rocof'].abs()

        #Filtering out time to nadir events over 25 seconds and under 2 seconds. Doing this for all my functions.
    # Calculate time difference between event start and nadir in seconds.
    event_df['time_to_nadir'] = (event_df['nadir_time'] - event_df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    event_df['time_to_nadir'] = event_df['time_to_nadir'].abs()

    event_df = event_df.loc[event_df['time_to_nadir'].between(2, 25)]



    for sign, color in [('positive', 'blue'), ('negative', 'red')]:
        subset = event_df.loc[event_df['rocof_sign'] == sign]
        ax.scatter(subset['max_rocof'], subset['time_to_nadir'], label=f'{sign.capitalize()} event', color=color,s=40,edgecolors='black')
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['max_rocof'], subset['time_to_nadir'])
        ax.plot(subset['max_rocof'], intercept + slope * subset['max_rocof'], color=color)

    #ax.set_title(f'Max RoCoF vs Time to Nadir - ')#{year}
    ax.set_xlabel(r'$\varphi$ [mHz{$\cdot$}s$^{-1}$]')
    #ax.set_ylabel('Time to Nadir [s]')
    #ax.legend()
    #plt.savefig(f'{os.path.dirname(base_folder_path)}/max_rocof_vs_time_to_nadir{year}.pdf')
    #plt.show()

def create_scatter_max_RoCoF_vs_time_to_nadir_2015_2023(df, ax):

    ax.set_xlim([37, 300])
    ax.set_ylim([0, 26])

    # Create scatter plot for max RoCoF vs time to nadir
    event_df = df.loc[df['event_start']].copy()

    # Get absolute value of max_rocof
    event_df['max_rocof'] = event_df['max_rocof'].abs()

    # Calculate time difference between event start and nadir in seconds.
    event_df['time_to_nadir'] = (event_df['nadir_time'] - event_df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    event_df['time_to_nadir'] = event_df['time_to_nadir'].abs()

    event_df = event_df.loc[event_df['time_to_nadir'].between(2, 25)]

    for sign, color in [('positive', 'blue'), ('negative', 'red')]:
        subset = event_df.loc[event_df['rocof_sign'] == sign]
        ax_big.scatter(subset['max_rocof'], subset['time_to_nadir'], label=f'{sign.capitalize()} event', color=color,s=47,edgecolors='black')
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['max_rocof'], subset['time_to_nadir'])
        ax.plot(subset['max_rocof'], intercept + slope * subset['max_rocof'], color=color)
        # Fill the area between 6 and 10 seconds with a greyish background
    ax.fill_between([xmin, xmax], 6, 10, color='k', alpha=0.3)
    
    ax_big.set_title(r'Max RoCoF vs Time to Nadir 2015 - 2023',size=20)
    ax_big.set_xlabel(r'RoCoF $\varphi$ [mHz{$\cdot$}s$^{-1}$]',size = 20)
    ax_big.set_ylabel(r'Time $t$ [s] to Nadir',size = 20)
    ax_big.legend(fontsize=15)

xmax = 300
xmin = 29
ymax = 28
ymin = 0
# Set the x and y-axis limits
x_limits = [xmin, xmax]  # replace xmin and xmax with the desired values
y_limits = [ymin, ymax]  # replace ymin and ymax with the desired values

# Create a figure and a grid of subplots
fig = plt.figure(figsize=(12, 5))
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
    #axis.set_yticklabels(axis.get_yticks(), fontsize='large')  # This line changes the y-axis tick label size
    #axis.tick_params(axis='y', pad=3, length=5,labelsize='large')  # This line changes the y-axis label size
# Hide the x-axis for the top 6 subplots


    # Add the year above each subplot
   # axis.set_title(f'{years[i]}', y=0.96,size=17)
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

    combined_df = pd.concat(dfs)

    # Create a new column to indicate if the RoCoF is positive or negative
    combined_df['rocof_sign'] = np.where(combined_df['rocof'] >= 0, 'positive', 'negative')

    # Save the combined dataframe as a new csv file
    #output_file = os.path.join(folder_path, f'cumulative_rocofevents_{year}.csv')
    #combined_df.to_csv(output_file, index=False)

# Histogram for number of occurance of events by hour.

    combined_df['Time'] = pd.to_datetime(combined_df['Time'])
    combined_df['nadir_time'] = pd.to_datetime(combined_df['nadir_time'])

    # Append the combined_df of each year to all_dfs
    all_dfs.append(combined_df)
    #create_histograms(combined_df)
    #create_scatter_max_RoCoF_vs_Nadir(combined_df,folder_path,ax[i])
    create_scatter_max_RoCoF_vs_time_to_nadir(combined_df,folder_path,ax[i])
    ax[i].tick_params(axis='y', labelsize=17, pad=3, length=5)  # This line changes the y-axis label size
    ax[i].tick_params(axis='x', labelsize=17, pad=3)
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins=1))  # Add this line after setting the tick parameters
    ax[i].xaxis.set_major_locator(MaxNLocator(nbins=2))
    if i < 6:
        ax[i].set_xticklabels([])
        #ax[i].set_yticklabels([])
        ax[i].set_xlabel('')
    ax[i].set_title(f'{year}', y=0.62,x=0.72, fontsize=20)
# Concatenate all dataframes in all_dfs to create a dataframe that includes data from all years
all_years_df = pd.concat(all_dfs)

fig.subplots_adjust(left=.075, bottom=.15, right=.99, top=.90,
                    hspace=.1, wspace=.5)
    # Call your function to create the main plot
create_scatter_max_RoCoF_vs_time_to_nadir_2015_2023(all_years_df, ax_big)
#ax_big.legend(handlelength=1.1, handletextpad=.5, loc=1, fontsize=18)
# Reduce whitespace around the figure
plt.tight_layout()
plt.savefig(f'{os.path.dirname(base_folder_path)}/max_rocof_vs_time_to_nadir{year}.pdf') 
plt.show()
