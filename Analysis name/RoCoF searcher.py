#RoCoF searcher

    #smoothed_df.to_csv(save_to + 'smoothed_' + year+'_'+month+'.zip',float_format='%.5f', #, originally float_format='%.0f', this changes to 0 decimals. I want 5 so is put to 5.
    #compression=dict(method='zip', archive_name='smoothed_'+year+'_'+month+'.csv'))
# From paper: Barrios-Gomez et al. - 2020 - RoCoF Calculation Using Low-Cost Hardware in the L.pdf
# RoCoF t dt NT −+ = = (4) where 𝑑𝑑f/dt is the RoCoF at sample t, N is the number of samples in the moving average window, T 
# is the duration of the moving average window and f(t) is the frequency at sample

# packs

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
import py7zr
import lzma
import os
from zipfile import ZipFile 
from tqdm import tqdm
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# Location parameters
year = '2017'
month = '01'
#years = ['2018',]
#months = ['01']

filename = 'smoothed_'+ year + '_' + month + '.zip'
location = r'C:/Users/Tore Tang/Data FinGrid smoothed/'+ year + '/' + month + '/'  # location of the zip file

# location to save file and plot
save_to = r'C:/Users/Tore Tang/Data Fingrid RoCoF events/'+year+'/'#+month+'/'

        # Reading in the file

with ZipFile(location + filename, 'r') as zip_file:
    # loop through each file in the zip archive, for later use with several files
    for file_info in tqdm(zip_file.infolist()):               #tqdm doesnt do much here as it is one zip file with one csv :P
        # check if it's a file 
        if not file_info.is_dir():
            # extract the file content
            with zip_file.open(file_info.filename) as file:
                # reading in the csv file
                df = pd.read_csv(file)                       
                #print('line54')
                df.drop(df.columns[df.columns.str.contains('Unnamed: 0',case = False)],axis = 1, inplace = True) #drop unnamed columns. #Weird that it appears. From the cleaning?
                #print(df)


# reading in raw data same as above.
raw_location = r'C:/Users/Tore Tang/Data FinGrid clean/'+ year + '/' + month + '/'
raw_filename = 'finland' + '_' + year + '_' + month + '.zip'

with ZipFile(raw_location + raw_filename, 'r') as zip_file:
    # loop through each file in the zip archive, for later use with several files
    for raw_file_info in tqdm(zip_file.infolist()):               #tqdm doesnt do much here as it is one zip file with one csv :P
        # check if it's a file 
        if not raw_file_info.is_dir():
            # extract the file content
            with zip_file.open(raw_file_info.filename) as raw_file:
                # reading in the csv file
                raw_df = pd.read_csv(raw_file)                       
                print('line54')
                raw_df.drop(df.columns[df.columns.str.contains('Unnamed: 0',case = False)],axis = 1, inplace = True) #drop unnamed columns. #Weird that it appears. From the cleaning?
                print(raw_df)

print('df loaded as:')
print(df)
print('raw_df loaded as:')
print(raw_df)

# adding in the raw frequency to the smoothed dataframe.
df = df.assign(Raw_Frequency=raw_df['Frequency'])

        #%%trying new finder, gives results.
        #import __builtin__
def calculate_rocof(df, N, T, rocof_limit):
    # Calculate the RoCoF using the symmetric difference quotient
    df['rocof'] = (df['Frequency'].shift(-N) - df['Frequency'].shift(N)) / (2 * T) # shifting to either side. Gives 2 times the time difference. Therefore divided by 2*1. N=10 gives 1 sec,
                                                                                    # shifting to either side gives 2 sec total. Want the results to be in /s.
    
    # Identify the RoCoF events
    df['rocof_event'] = np.where(np.abs(df['rocof']) > rocof_limit, df['rocof'], np.nan) # if the RoCoF is larger than the limit, keep the value, else put NaN

    # Identify the start and end indices of each event
    event_starts = df.loc[df['rocof_event'].notna() & df['rocof_event'].shift().isna()].index
    event_ends = df.loc[df['rocof_event'].notna() & df['rocof_event'].shift(-1).isna()].index

    # Set the rocof_event value to 1 for 5 seconds before and 30 seconds after each event. This to prevent them from dropping out in the code below.
    for start, end in zip(event_starts, event_ends):
        df.loc[start - 10 * N : start, 'rocof_event'] = 1
        df.loc[end : end + 60 * N, 'rocof_event'] = 1

    # Drop rows where RoCoF value is NaN
    df = df.dropna(subset=['rocof_event'])

    return df

N=5     # number of samples in moving average window. 10 points would equal 1 second. In each direction(.shift())
T=0.5      # duration of the moving average window
rocof_limit = 40
df_rocof = calculate_rocof(df, N, T, rocof_limit).copy()                 # copy to avoid SettingWithCopyWarning

        # Plot the frequency and RoCoF events

save_df_to=r'C:/Users/Tore Tang/Data Fingrid RoCoF events/csvfiles/' + year + '/'#+month+'/'

#saving the dataframe to a csv file:
df_rocof.to_csv(save_df_to + 'rocofevents_'+ month + '_' + year + '.csv',float_format='%.5f') #, originally float_format='%.0f', this changes to 0 decimals. I want 5 so is put to 5.


print(df_rocof)

# Plot the frequency and RoCoF events
import matplotlib.pyplot as plt

def plot_event(df_event,event_number):
    fig, ax1 = plt.subplots(figsize=(15, 8))  # Set the figure size here

    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency, [mHz]', color=color)
    ax1.plot(df_event['Time'], df_event['Raw_Frequency'], color='yellowgreen', label='Raw Frequency')  # Add this line
    ax1.plot(df_event['Time'], df_event['Frequency'], color=color, label='Frequency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')  # Adding a legend

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('RoCoF [mHz/s]', color=color)
    ax2.plot(df_event['Time'], df_event['rocof'], color=color, label='RoCoF')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')  # Add this line

    # Calculate the maximum RoCoF value and the date and time of the event
    max_rocof = df_event['rocof'].abs().max()
    max_rocof_time = df_event.loc[df_event['rocof'].idxmax(), 'Time']

    # Format the numbers to include only two decimal points
    max_rocof_str = "{:.2f}".format(max_rocof)
    max_rocof_time_str = max_rocof_time.strftime("%Y-%m-%d %H:%M:%S")

    # Add a title to the figure
    plt.title(f'Maximum RoCoF [mHz/s]: {max_rocof_str} at {max_rocof_time_str}')

    fig.tight_layout()
    #plt.show()

# Before saving the figure
    save_path = os.path.join(save_to,f'{year}_{month}_event_{event_number}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)

    #fig.savefig(os.path.join(save_to,f'{year}_{month}_event_{event_number}.png'))

    plt.close(fig)  # Close the figure to free up memory.

# Convert 'Time' to datetime format
df_rocof.loc[:, 'Time'] = pd.to_datetime(df_rocof['Time'])

# Calculate the time difference between consecutive events
df_rocof.loc[:, 'Time_diff'] = df_rocof['Time'].diff()

# Define a threshold for the time difference (e.g., 1 minute)
threshold = pd.Timedelta(minutes=1.5)

# Find the indices where the time difference is larger than the threshold
indices = df_rocof[df_rocof['Time_diff'] > threshold].index

# Plot each event separately
start = df_rocof.index[0]
for i, end in enumerate(indices,start=1):
    df_event = df_rocof.loc[start:end-1]
    plot_event(df_event,i)
    start = end

# Plot the last event
df_event = df_rocof.loc[start:]
plot_event(df_event,len(indices)+1)



print('done')











#def calculate_rocof(df, N, T, rocof_limit):
    # Calculate the RoCoF
#    rocof = (df['Frequency'].diff(periods=N) / (N - T)).fillna(0)

    # Identify the RoCoF events and store the RoCoF value
#    df['rocof_event'] = np.where(np.abs(rocof) > rocof_limit, rocof, 0)

    # Drop rows where RoCoF value is 0
#    df = df[df['rocof_event'] != 0]

#    return df


"""""
#%%working rocof finder
def calculate_rocof(df, N, T, rocof_limit):
    # Calculate the RoCoF
    rocof = (df['Frequency'].diff(periods=N) / (N - T)).fillna(0)

    # Identify the RoCoF events
    df['rocof_event'] = np.where(np.abs(rocof) > rocof_limit, rocof, 0)

# Drop rows where RoCoF value is NaN
#    df = df.dropna(subset=['rocof_event'])

    return df

# Load your data
#data = pd.read_csv('your_data.csv')
"""""


#%% RoCoF Incremental one step
"""""
periods= 15 # number of steps between calculation
def calculate_rocof(df, periods):
    rocof_limit = 200  # RoCoF limit. The data fed in is in mHz. So putting 100 here would be 0.1 Hz/s.
    # Convert the index to datetime format
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Calculate the difference in frequency and time
    df['freq_diff'] = df['Frequency'].diff(periods)   # Convert to Hz
    df['time_diff'] = df.index.to_series().diff(periods).dt.total_seconds() # the time difference 

    # Calculate RoCoF
    df['rocof'] = df['freq_diff'] / df['time_diff']

    # Identify RoCoF events
    df['rocof_event'] = (df['rocof'].abs() > rocof_limit).astype(int)

    # Drop the intermediate columns
    df = df.drop(columns=['freq_diff', 'time_diff'])

    return df

# Assuming 'df' is your DataFrame with the frequency data
# And 'periods' is the number of points between each RoCoF estimation
df = calculate_rocof(df, periods)

print(df[0:20])
print(df[1000:1020])
print(df[10000:10020])
"""""

"""""
def calculate_rocof(df, window_size, duration):
    # Sort the DataFrame by timestamp
    rocof_limit = 40  # RoCoF limit. The data fed in is in mHz. So putting 100 here would be 0.1 Hz/s.
    df.sort_values(by='Time', inplace=True)

    # Calculate RoCoF using a moving window
    df['rocof'] = (df['Frequency'].diff(periods=window_size) / duration)

    # Drop rows where RoCoF value is NaN
    df = df.dropna(subset=['rocof'])

    # Identify the RoCoF events
    #df['rocof_event'] = np.where(np.abs(df['rocof']) > rocof_limit, 'rocof', 0)

    return df
"""""
# Example usage:
# Assuming df is your DataFrame with 'timestamp' and 'frequency' columns
# You can adjust window_size and duration according to your analysis requirements
#window_size = 10  # Number of samples in the moving average window
#duration = 1.0   # Duration of the moving average window

#result_df = calculate_rocof(df, window_size, duration)

# Print the resulting DataFrame with RoCoF values
#print(result_df)



"""""
# RoCoF Moving window. 
N = 10               # number of samples in moving average window. 10 points would equal 1 second.
T = 1              # duration of the moving average window
rocof_limit = 40  # RoCoF limit. The data fed in is in mHz. So putting 100 here would be 0.1 Hz/s. 

# Calculate RoCoF and identify events
df = calculate_rocof(df, N, T, rocof_limit)
#df = df[df['rocof_event'] != 0]
print(df)
"""""

#plotting
"""""
N=10
import plotly.subplots as sp
import plotly.graph_objects as go

def plot_data(df, N):
    # Find the indices of the RoCoF events
    event_indices = df[df['rocof_event'].notna()].index

    # Initialize a list to store the indices of the points to keep
    indices_to_keep = []

    # For each event, add the indices of the N points before and after the event
    for idx in event_indices:
        start = max(0, idx - N)
        end = min(len(df) - 1, idx + N)
        indices_to_keep.extend(range(start, end + 1))

    # Select the points to keep from the DataFrame
    df_expanded = df.iloc[indices_to_keep]

    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df_expanded.index, 
        y=df_expanded['Frequency'],
        mode='lines',
        name='Frequency'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df_expanded.index, 
        y=df_expanded['rocof_event'],
        mode='markers',
        name='RoCoF mHz/s',
        marker=dict(size=10, color='red')
    ), secondary_y=True)

    fig.update_yaxes(title_text="Frequency", secondary_y=False)
    fig.update_yaxes(title_text="RoCoF", secondary_y=True)

    fig.show()

# Assuming 'df' is your DataFrame with the frequency data and RoCoF events
plot_data(df, 30)
"""""


"""""
import plotly.subplots as sp
import plotly.graph_objects as go

fig_df = df[20000:30000]
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Index')
ax1.set_ylabel('Frequency', color=color)
ax1.plot(fig_df.index, fig_df['Frequency'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('rocof_event', color=color)  # we already handled the x-label with ax1
ax2.plot(fig_df.index, fig_df['rocof'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



df_fig = result_df
print(df_fig)
def plot_data(df_fig):
    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df_fig.index, 
        y=df_fig['Frequency'],
        mode='lines',
        name='Frequency'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df_fig.index, 
        y=df_fig['rocof'],
        mode='lines+markers',
        name='rocof',
        marker=dict(size=8,color='red')
    ), secondary_y=True)

    fig.update_yaxes(title_text="Frequency", secondary_y=False)
    fig.update_yaxes(title_text="RoCoF", secondary_y=True)

    fig.show()

# Assuming 'df' is your DataFrame with the frequency data and RoCoF events
plot_data(df)
"""""



#RoCoF(t_i) = df/dt = (f(t_i) - f(t_i-1 + N)) / (N - T) # N, number of samples in moving average window, T is the duration of the moving average window
                                                        # T is the duration of the moving average window
                                                        # df/dt is the "pure" RoCoF
                                                        # f(t) is the frequency at sample t_i(+1) 

