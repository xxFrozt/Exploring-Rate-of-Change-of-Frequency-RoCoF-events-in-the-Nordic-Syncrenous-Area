### 2015-2023 plotting histograms for 2015-2023. Subplots for each year.



import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
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
#Creating histograms and fiolin plots for the data(yay!)


years= [str(year) for year in range(2015,2024)]
# Reading all files from a given folder
base_folder_path = r'C:/Users/Tore Tang/Data Fingrid RoCoF events/csvfiles/'#+ year + '/'  # Replace with the actual folder path
file_prefix = 'cumulative_'
#file_suffix = year + '.csv'

# List all files in the folder that match the file prefix and suffix
file_list = [file for file in os.listdir(base_folder_path) if file.startswith(file_prefix)] #and file.endswith(file_suffix)
print(f'File list: {file_list}')

# Read each file and combine them into one dataframe
#dfs = []
#for file in file_list:
#    file_path = os.path.join(base_folder_path, file)
#    df = pd.read_csv(file_path)
#    dfs.append(df)

#combined_df = pd.concat(dfs)

# Save the combined dataframe as a new csv file
#output_file = os.path.join(base_folder_path, f'{year}.csv')
#combined_df.to_csv(output_file, index=False)

# Histogram for number of occurance of events by hour.

#combined_df['Time'] = pd.to_datetime(combined_df['Time'])
#combined_df['nadir_time'] = pd.to_datetime(combined_df['nadir_time'])

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
    #plt.title(f'Events by hour in {year} (Number of events: {total_events})')
    ax.set_xlabel(r'Hour')
    #ax.ylabel('Number of events')
    #ax.xticks(range(0, 24))  # Set x-axis ticks to show all hour stamps
    #plt.savefig(f'{os.path.dirname(folder_path)}/histogram_events_vs_hour{year}.pdf')
#    plt.show()


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
    ax_big.set_xticks(range(0, 24))  # Set x-axis ticks to show all hour stamps
    #plt.savefig(f'{os.path.dirname(folder_path)}/histogram number of events vs hours cumulative 2015-2023.pdf')
    #plt.show()


# max RoCoF value vs Nadir

# max RoCoF vs time to nadir


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
    #combined_df['rocof_sign'] = np.where(combined_df['rocof'] >= 0, 'positive', 'negative')

    # Save the combined dataframe as a new csv file
    output_file = os.path.join(folder_path, f'cumulative_rocofevents_{year}.csv')
    combined_df.to_csv(output_file, index=False)


    # Convert the Time and nadir_time columns to datetime objects
    combined_df['Time'] = pd.to_datetime(combined_df['Time'])
    combined_df['nadir_time'] = pd.to_datetime(combined_df['nadir_time'])

    # Append the combined_df of each year to all_dfs
    all_dfs.append(combined_df)

    # Call your function to create the subplot for the current year.
    create_histograms(combined_df,folder_path,ax[i])
    ax[i].tick_params(axis='y', labelsize=18, pad=5, length=5)  # This line changes the y-axis label size
    ax[i].tick_params(axis='x', labelsize=16, pad=3, length=5)
    ymin, ymax = ax[i].get_ylim()
    ax[i].set_ylim(ymin, ymax * 0.95) 
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins=1))  # Add this line after setting the tick parameters
    ax[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
# Concatenate all dataframes in all_dfs to create a dataframe that includes data from all years
all_years_df = pd.concat(all_dfs)
#ax_big.legend(handlelength=1.1, handletextpad=.5, loc=2, fontsize=18)
fig.subplots_adjust(left=.075, bottom=.15, right=.99, top=.90,
                    hspace=.1, wspace=.5)
create_histograms_2015_2023(all_years_df,folder_path, ax_big) # Call your function to create the main plot. Using the combined dataframe of all years. Do not put this in the loop above.
ax_big.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax_big.set_xlim(x_limits)
# Reduce whitespace around the figure
plt.tight_layout()
plt.savefig(f'{os.path.dirname(base_folder_path)}/histogram number of events.pdf') 
plt.show()