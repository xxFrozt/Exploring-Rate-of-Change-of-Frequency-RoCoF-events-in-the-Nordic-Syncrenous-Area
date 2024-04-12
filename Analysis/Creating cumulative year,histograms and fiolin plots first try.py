#Creating histograms and fiolin plots for the data(yay!)

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 14,
    'axes.labelsize': 10,'axes.titlesize': 14, 'figure.titlesize' : 23})
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import matplotlib.gridspec as gridspec
#Creating histograms and fiolin plots for the data(yay!)


#year = '2023'

years= [str(year) for year in range(2015,2024)]

# Reading all files from a given folder
base_folder_path = r'C:/Users/Tore Tang/Data Fingrid RoCoF events/csvfiles/'#+ year + '/'  # Replace with the actual folder path
file_prefix = 'rocofevents_'
#file_suffix = year + '.csv'


def create_histograms(df):
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

    # Calculate total number of events
    total_events = len(event_df)

    # Create histogram for hour
    plt.figure(figsize=(10, 5))
    plt.hist(event_df['hour'], bins=range(0, 25), edgecolor='black')
    plt.title(f'Events by hour in {year} (Number of events: {total_events})')
    plt.xlabel('Hour')
    plt.ylabel('Number of events')
    plt.xticks(range(0, 24))  # Set x-axis ticks to show all hour stamps
    plt.savefig(f'{os.path.dirname(folder_path)}/histogram_events_vs_hour{year}.pdf')
#    plt.show()


# max RoCoF value vs Nadir





"""""""""
def create_scatter_max_RoCoF_vs_Nadir(df,folder_path):
    # Filter rows where event_start is True
    event_df = df.loc[df['event_start']].copy()

    # Get absolute value of max_rocof
    event_df['max_rocof'] = event_df['max_rocof'].abs()

    event_df['nadir'] = event_df['nadir'].abs()

    # Create scatter plot for max RoCoF vs Nadir
    plt.figure(figsize=(10, 5))
    for sign in ['positive', 'negative']:
        plt.scatter(event_df.loc[event_df['max_rocof_sign'] == sign, 'max_rocof'], 
                    event_df.loc[event_df['max_rocof_sign'] == sign, 'nadir'], 
                    label=sign)
    #plt.scatter(event_df['max_rocof'], event_df['nadir'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(event_df['max_rocof'], event_df['nadir'])
    correlation_coefficient = event_df['max_rocof'].corr(event_df['nadir']) # added in to test
    plt.plot(event_df['max_rocof'], intercept + slope * event_df['max_rocof'], 'r', label=f'R-squared: {r_value**2:.2f}\nCorrelation Coefficient: {correlation_coefficient:.2f}')
    plt.title(f'Max RoCoF vs Nadir - {year}') # added in to test')
    plt.xlabel('Max RoCoF')
    plt.ylabel('Nadir')
    #plt.text(0.95, 0.01, f'Correlation Coefficient: {correlation_coefficient:.2f}',
    #     verticalalignment='bottom', horizontalalignment='right',
    #     transform=plt.gca().transAxes,
    #     color='green', fontsize=15)
    plt.legend()
    plt.savefig(f'{os.path.dirname(folder_path)}/max_rocof_nadir_{year}.pdf')
    plt.show()
    plt.close()   

def create_scatter_max_RoCoF_vs_time_to_nadir(df,folder_path):
    # Create scatter plot for max RoCoF vs time to nadir
    event_df = df.loc[df['event_start']].copy()

    # Get absolute value of max_rocof
    event_df['max_rocof'] = event_df['max_rocof'].abs()

    # Calculate time difference between event start and nadir in seconds.
    event_df['time_to_nadir'] = (event_df['nadir_time'] - event_df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    event_df['time_to_nadir'] = event_df['time_to_nadir'].abs()


    plt.figure(figsize=(10, 5))
    for sign in ['positive', 'negative']:
        plt.scatter(event_df.loc[event_df['max_rocof_sign'] == sign, 'max_rocof'], 
                    event_df.loc[event_df['max_rocof_sign'] == sign, 'time_to_nadir'], 
                    label=sign)    
    
    #plt.scatter(event_df['max_rocof'], event_df['time_to_nadir'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(event_df['max_rocof'], event_df['time_to_nadir'])
    correlation_coefficient = event_df['max_rocof'].corr(event_df['time_to_nadir']) # added in to test
    plt.plot(event_df['max_rocof'], intercept + slope * event_df['max_rocof'], 'r', label=f'R-squared: {r_value**2:.2f}\nCorrelation Coefficient: {correlation_coefficient:.2f}')

    plt.title(f'Max RoCoF vs Time to Nadir - {year}')
    plt.xlabel('Max RoCoF')
    plt.ylabel('Time to Nadir (seconds)')
    #plt.text(0.95, 0.01, f'Correlation Coefficient: {correlation_coefficient:.2f}',
    #     verticalalignment='bottom', horizontalalignment='right',
    #     transform=plt.gca().transAxes,
    #     color='green', fontsize=15)
    plt.legend()
    plt.savefig(f'{os.path.dirname(folder_path)}/max_rocof_vs_time_to_nadir{year}.pdf')
    plt.show()
"""""""""

#trying new stuff:

from scipy import stats

def create_scatter_max_RoCoF_vs_Nadir(df, folder_path,ax):
    # Filter rows where event_start is True
    event_df = df.loc[df['event_start']].copy()

    # Get absolute value of max_rocof
    event_df['max_rocof'] = event_df['max_rocof'].abs()

    #Filtering out time to nadir events over 25 seconds and under 2 seconds. Doing this for all my functions.
    # Calculate time difference between event start and nadir in seconds.
    event_df['time_to_nadir'] = (event_df['nadir_time'] - event_df['Time']).dt.total_seconds()

    # Get absolute value of time_to_nadir
    event_df['time_to_nadir'] = event_df['time_to_nadir'].abs()

    event_df = event_df.loc[event_df['time_to_nadir'].between(2, 25)]



    event_df.rename(columns={'nadir': 'nadir_point'}, inplace=True)

    event_df['nadir'] = event_df['Frequency'] - event_df['nadir_point']

    event_df['nadir'] = event_df['nadir'].abs()

    # Create scatter plot for max RoCoF vs Nadir
    plt.figure(figsize=(10, 5))
    for sign, color in [('positive', 'blue'), ('negative', 'red')]:
        subset = event_df.loc[event_df['rocof_sign'] == sign]
        ax.scatter(subset['max_rocof'], subset['nadir'], label=f'{sign.capitalize()} event',color=color)
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['max_rocof'], subset['nadir'])
        plt.plot(subset['max_rocof'], intercept + slope * subset['max_rocof'], color=color)

    plt.title(f'Max RoCoF vs Nadir - {year}')
    plt.xlabel('Max RoCoF [mHz/s]')
    plt.ylabel('Nadir [mHz]')
    plt.legend()
    plt.savefig(f'{os.path.dirname(folder_path)}/max_rocof_nadir_{year}.pdf')
    plt.show()
    plt.close()

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
        ax.scatter(subset['max_rocof'], subset['time_to_nadir'], label=f'{sign.capitalize()} event', color=color)
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['max_rocof'], subset['time_to_nadir'])
        ax.plot(subset['max_rocof'], intercept + slope * subset['max_rocof'], color=color)

    #ax.set_title(f'Max RoCoF vs Time to Nadir - ')#{year}
    ax.set_xlabel(r'$\varphi$ [mHz{$\cdot$}s$^{-1}$]')
    #ax.set_ylabel('Time to Nadir [s]')
    #ax.legend()
    #plt.savefig(f'{os.path.dirname(base_folder_path)}/max_rocof_vs_time_to_nadir{year}.pdf')
    #plt.show()

def create_scatter_max_RoCoF_vs_time_to_nadir_2015_2023(df, ax):

    ax.set_xlim([36, 300])
    ax.set_ylim([1.5, 26])

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
        ax.scatter(subset['max_rocof'], subset['time_to_nadir'], label=f'{sign.capitalize()} event', color=color)
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['max_rocof'], subset['time_to_nadir'])
        ax.plot(subset['max_rocof'], intercept + slope * subset['max_rocof'], color=color)

    ax.set_title(r'Max RoCoF vs Time to Nadir 2015 - 2023')
    ax.set_xlabel(r'$\varphi$ [mHz{$\cdot$}s$^{-1}$]')
    ax.set_ylabel(r'Time $t$ [s] to Nadir')
    #ax.legend()




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
# Hide the x-axis for the top 6 subplots
    if i < 6:
        axis.xaxis.set_visible(False)

    # Add the year above each subplot
    axis.set_title(f'{years[i]}', y=0.97)
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
    output_file = os.path.join(folder_path, f'cumulative_rocofevents_{year}.csv')
    combined_df.to_csv(output_file, index=False)

# Histogram for number of occurance of events by hour.

    combined_df['Time'] = pd.to_datetime(combined_df['Time'])
    combined_df['nadir_time'] = pd.to_datetime(combined_df['nadir_time'])

    # Append the combined_df of each year to all_dfs
    all_dfs.append(combined_df)
    #create_histograms(combined_df)
    #create_scatter_max_RoCoF_vs_Nadir(combined_df,folder_path,ax[i])
    create_scatter_max_RoCoF_vs_time_to_nadir(combined_df,folder_path,ax[i])

# Concatenate all dataframes in all_dfs to create a dataframe that includes data from all years
all_years_df = pd.concat(all_dfs)

    # Call your function to create the main plot
create_scatter_max_RoCoF_vs_time_to_nadir_2015_2023(all_years_df, ax_big)
plt.savefig(f'{os.path.dirname(base_folder_path)}/max_rocof_vs_time_to_nadir{year}.pdf') 
#