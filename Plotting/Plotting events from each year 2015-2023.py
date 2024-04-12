### Plotting individual events



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
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import date2num
import datetime

#Creating histograms and fiolin plots for the data(yay!)


years= [str(year) for year in range(2015,2024)]
# Reading all files from a given folder
base_folder_path = r'C:/Users/Tore Tang/Data Fingrid RoCoF events/csvfiles/'#+ year + '/'  # Replace with the actual folder path
file_prefix = 'cumulative_'
#file_suffix = year + '.csv'

# List all files in the folder that match the file prefix and suffix
file_list = [file for file in os.listdir(base_folder_path) if file.startswith(file_prefix)] #and file.endswith(file_suffix)
print(f'File list: {file_list}')

def plot_event(df_event,ax1,is_leftmost,is_rightmost,is_bottom,event_time):
    color = 'tab:blue'
    if is_bottom:
        ax1.set_xlabel('Time')
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax1.tick_params(axis='x', labelsize=16,length=5)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format the x-axis labels as 'HH:MM:SS'
        plt.setp(ax1.get_xticklabels(), rotation=0)

        # Set a fixed range for the x-axis
        start_time = event_time - pd.Timedelta(seconds=5)
        end_time = event_time + pd.Timedelta(seconds=30)  # Add 24 hours to the start time
        start_time_num = date2num(start_time.to_pydatetime())
        end_time_num = date2num(end_time.to_pydatetime())
        ax1.set_xlim(start_time_num, end_time_num)

    ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax1.tick_params(axis='x', labelsize=16,length=5)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format the x-axis labels as 'HH:MM:SS'


    start_time = event_time - pd.Timedelta(seconds=5)
    end_time = event_time + pd.Timedelta(seconds=30)  # Add 24 hours to the start time
    start_time_num = date2num(start_time.to_pydatetime())
    end_time_num = date2num(end_time.to_pydatetime())
    ax1.set_xlim(start_time_num, end_time_num)

    plt.setp(ax1.get_xticklabels(), rotation=0)
    
    if is_leftmost:
        ax1.set_ylabel('Frequency [mHz]', color=color)
    ax1.plot(df_event['Time'], df_event['Raw_Frequency'], color='yellowgreen', label='Raw Frequency')  # Add this line
    ax1.plot(df_event['Time'], df_event['Frequency'], color=color, label='Frequency')
    ax1.tick_params(axis='y', labelcolor=color, length=5)
    ax1.tick_params(axis='x', labelsize=16,length=5)
    #ax1.legend(False)  # Adding a legend

    ax2 = ax1.twinx()

    color = 'tab:red'
    if is_rightmost:
        ax2.set_ylabel('RoCoF [mHz/s]', color=color)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(1))
    
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.plot(df_event['Time'], df_event['rocof'], color=color, label='RoCoF')
    ax2.tick_params(axis='y', labelcolor=color)
    #ax2.legend()  # Add this line

    # Calculate the maximum RoCoF value and the date and time of the event
    max_rocof = df_event['rocof'].abs().max()
    max_rocof_time = df_event.loc[df_event['rocof'].idxmax(), 'Time']

    # Format the numbers to include only two decimal points
    max_rocof_str = "{:.2f}".format(max_rocof)
    max_rocof_time_str = max_rocof_time.strftime("%d-%m-%Y %H:%M:%S")
    day = max_rocof_time.day
    month = max_rocof_time.month
    # Add a title to the figure
    ax1.set_title(f'{day}-{month}-{year}')

    fig.tight_layout()
    #plt.show()

# Create a figure and a grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 8)) 


events_to_plot = {
    '2015': '2015-01-16 09:51:17',
    '2016': '2016-07-12 14:42:29',
    '2017': '2017-06-13 04:55:29',
    '2018': '2018-06-08 10:43:11',
    '2019': '2019-06-04 13:51:20',    #04-06-2019  13:51:20 trying this. OLD: '2019-10-04 09:53:13'
    '2020': '2020-09-10 10:11:21',
    '2021': '2021-07-26 19:02:36',
    '2022': '2022-08-21 21:13:01',
    '2023': '2023-06-25 21:08:33'
}  #'2015': '16-01-2015 09:51:17'
    #axis.tick_params(axis='y',pad=3,length=5)  # This line changes the y-axis label size
# Hide the x-axis for the top 6 subplots

    # Add the year above each subplot
# Initialize an empty list to store all dataframes
all_dfs = []
lines_labels = [([], [])]
for i, (year,event_time) in enumerate(events_to_plot.items()):
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
    combined_df['Time'] = pd.to_datetime(combined_df['Time'], format="%Y-%m-%d %H:%M:%S.%f")
    event_time = pd.to_datetime(events_to_plot[year])

    # Create a time range from 5 seconds before to 30 seconds after the event
    start_time = event_time - pd.Timedelta(seconds=5)
    end_time = event_time + pd.Timedelta(seconds=30)

    df_event = combined_df[(combined_df['Time'] >= start_time) & (combined_df['Time'] <= end_time)]
    ax1 = axs[i // 3, i % 3]
    is_leftmost = (i % 3 == 0)  # Check if the subplot is in the leftmost column
    is_rightmost = (i % 3 == 2)
    is_bottom = (i // 3 == 2)
    if not df_event.empty:
        plot_event(df_event,ax1,is_leftmost,is_rightmost,is_bottom,event_time)
        lines, labels = ax1.get_legend_handles_labels()  # Get lines and labels from the current subplot
        lines_labels[0][0].extend(lines)  # Add lines to the global list
        lines_labels[0][1].extend(labels)  # Add labels to the global list
    else:
        print(f"No events found in the range {start_time} to {end_time} for year {year}")
    #plot_event(df_event)

    # Save the combined dataframe as a new csv file
    #output_file = os.path.join(folder_path, f'cumulative_rocofevents_{year}.csv')
    #combined_df.to_csv(output_file, index=False)

    # Append the combined_df of each year to all_dfs
    all_dfs.append(combined_df)

    # Call your function to create the subplot for the current year.
# Concatenate all dataframes in all_dfs to create a dataframe that includes data from all years
#all_years_df = pd.concat(all_dfs)
#ax_big.legend(handlelength=1.1, handletextpad=.5, loc=2, fontsize=18)

# Reduce whitespace around the figure
plt.tight_layout()
plt.savefig(f'{os.path.dirname(base_folder_path)}/individual events 2015-2023.pdf') 
