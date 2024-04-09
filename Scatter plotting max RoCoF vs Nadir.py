#This code creates a scatterplot of max RoCoF vs Nadir.
# It is designed to make a big plot of all year 2015-2023 and then 9 smaller plots for each year.
# The code is designed to be used with the csv files created by the code in RoCoF events.py
# Please change the file path to the folder where the csv files are stored if you are to use this on your data :)).
# There are also adjustments to the font size and the size of the plot to make it look better in a pdf file.

import pandas as pd
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'                   # This line is used to make the font in the plot look better when saved as a pdf file for use in Latex.
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 15,
    'axes.labelsize': 20,'axes.titlesize': 20, 'figure.titlesize' : 48})
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
#

years= [str(year) for year in range(2015,2024)]

# Reading all files from a given folder
base_folder_path = r'C:/Users/Tore Tang/Data Fingrid RoCoF events/csvfiles/'#+ year + '/'  # Replace with the actual folder path
file_prefix = 'rocofevents_'

def create_scatter_max_RoCoF_vs_Nadir(df, folder_path,ax):
    
    # Set the x and y-axis parameters
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Filter rows where event_start is True
    event_df = df.loc[df['event_start']].copy()

    # Get absolute value of max_rocof
    event_df['max_rocof'] = event_df['max_rocof'].abs()

    # Filtering out time to nadir events over 25 seconds and under 2 seconds. Doing this for all my functions.
    # Calculate time difference between event start and nadir in seconds.
    event_df['time_to_nadir'] = (event_df['nadir_time'] - event_df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    event_df['time_to_nadir'] = event_df['time_to_nadir'].abs()

    event_df = event_df.loc[event_df['time_to_nadir'].between(2, 25)]

    event_df.rename(columns={'nadir': 'nadir_point'}, inplace=True)

    event_df['nadir'] = event_df['Frequency'] - event_df['nadir_point']

    event_df['nadir'] = event_df['nadir'].abs()

   # Create scatter plot for max RoCoF vs Nadir
    for sign, color in [('positive', 'blue'), ('negative', 'red')]:
        subset = event_df.loc[event_df['rocof_sign'] == sign]
        ax.scatter(subset['max_rocof'], subset['nadir'], label=f'{sign.capitalize()} event',color=color,s=40,edgecolors='black')
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['max_rocof'], subset['nadir'])
        ax.plot(subset['max_rocof'], intercept + slope * subset['max_rocof'], color=color)
    ax.set_xlabel(r'$\varphi$ [mHz{$\cdot$}s$^{-1}$]')



def create_scatter_max_RoCoF_vs_Nadir_2015_2023(df,ax):

    ax.set_xlim([37, 300]) # Set the x-axis limits
    ax.set_ylim([0, 750])  # Set the y-axis limits

    # Filter rows where event_start is True
    event_df = df.loc[df['event_start']].copy()

    #Filtering out time to nadir events over 25 seconds and under 2 seconds. Doing this for all my functions.
    # Calculate time difference between event start and nadir in seconds.
    event_df['time_to_nadir'] = (event_df['nadir_time'] - event_df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    event_df['time_to_nadir'] = event_df['time_to_nadir'].abs()

    event_df = event_df.loc[event_df['time_to_nadir'].between(2, 25)]

    # Get absolute value of max_rocof
    event_df['max_rocof'] = event_df['max_rocof'].abs()

    event_df.rename(columns={'nadir': 'nadir_point'}, inplace=True)

    event_df['nadir'] = event_df['Frequency'] - event_df['nadir_point']

    event_df['nadir'] = event_df['nadir'].abs()

    event_df.to_csv(f'{os.path.dirname(base_folder_path)}/scatterplot max RoCoF vs nadir cumulative 2015-2023.csv', index=False)

    # Create scatter plot for max RoCoF vs Nadir
    
    marker_dict = {'positive': 'x', 'negative': 'v'}
    for sign, color in [('positive', 'blue'), ('negative', 'red')]:
        subset = event_df.loc[event_df['rocof_sign'] == sign]
        ax.scatter(subset['max_rocof'], subset['nadir'], label=f'{sign.capitalize()} event',color=color,s=47,edgecolors='black')
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['max_rocof'], subset['nadir'])
        ax_big.plot(subset['max_rocof'], intercept + slope * subset['max_rocof'], color=color)
    ax_big.set_title(r'Max RoCoF vs Nadir 2015 - 2023',size= 25) 
    ax_big.set_xlabel(r'RoCoF $\varphi$ [mHz{$\cdot$}s$^{-1}$]',size = 22)
    ax_big.set_ylabel(r'Nadir $\beta$ [mHz]',size = 22)
    ax_big.legend(fontsize='large')
    #ax_big.tick_params(axis='x', labelsize=19, pad=3, length=5)
    #ax_big.tick_params(axis='y', labelsize=19, pad=3, length=5)
    ax_big.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_big.xaxis.set_major_locator(MaxNLocator(nbins=5))
xmax = 300
xmin = 32
ymax = 750
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
    #axis.tick_params(axis='y',pad=3,length=5)  # This line changes the y-axis label size
# Hide the x-axis for the top 6 subplots
    if i < 6:
        axis.xaxis.set_visible(False)

    # Add the year above each subplot
    #axis.set_title(f'{years[i]}', y=0.96,size= 17)
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

    # Create a new column to indicate if the RoCoF is positive or negative
    combined_df['rocof_sign'] = np.where(combined_df['rocof'] >= 0, 'positive', 'negative')

    # Save the combined dataframe as a new csv file
    output_file = os.path.join(folder_path, f'cumulative_rocofevents_{year}.csv')
    combined_df.to_csv(output_file, index=False)


    # Convert the Time and nadir_time columns to datetime objects
    combined_df['Time'] = pd.to_datetime(combined_df['Time'])
    combined_df['nadir_time'] = pd.to_datetime(combined_df['nadir_time'])

    # Append the combined_df of each year to all_dfs
    all_dfs.append(combined_df)

    # Call your function to create the subplot for the current year.
    create_scatter_max_RoCoF_vs_Nadir(combined_df,folder_path,ax[i])
    ax[i].tick_params(axis='y', labelsize=21, pad=3, length=5)  # This line changes the y-axis label size
    ax[i].tick_params(axis='x', labelsize=15, pad=3, length=5)
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins=1))  # Add this line after setting the tick parameters
# Concatenate all dataframes in all_dfs to create a dataframe that includes data from all years
all_years_df = pd.concat(all_dfs)
#ax_big.legend(handlelength=1.1, handletextpad=.5, loc=2, fontsize=18)
fig.subplots_adjust(left=.075, bottom=.15, right=.99, top=.90,
                    hspace=.1, wspace=.5)
create_scatter_max_RoCoF_vs_Nadir_2015_2023(all_years_df, ax_big) # Call your function to create the main plot. Using the combined dataframe of all years. Do not put this in the loop above.
#ax_big.set_title(r'Max RoCoF vs Nadir 2015 - 2023')

plt.savefig(f'{os.path.dirname(base_folder_path)}/max_rocof_vs_nadir{year}.pdf') 
plt.show()