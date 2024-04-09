#RoCoF searcher WHICH WORKED..


# From paper: Barrios-Gomez et al. - 2020 - RoCoF Calculation Using Low-Cost Hardware in the L.pdf
# RoCoF t dt NT −+ = = (4) where 𝑑𝑑f/dt is the RoCoF at sample t, N is the number of samples in the moving average window, T 
# is the duration of the moving average window and f(t) is the frequency at sample

# packs

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from zipfile import ZipFile 
from tqdm import tqdm
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# Parameters
#year = '2016'
#month = '08'
#filename = 'smoothed_'+ year + '_' + month + '.zip'
#location = r'C:/Users/Tore Tang/Data FinGrid smoothed/'+ year + '/' + month + '/'  # location of the zip file

# Location to save file and plot
#save_to = r'C:/Users/Tore Tang/Data Fingrid RoCoF events/'+year+'/'#+month+'/'

#%% Reading in the smoothed file

#with ZipFile(location + filename, 'r') as zip_file:
    # loop through each file in the zip archive, for later use with several files
#    for file_info in tqdm(zip_file.infolist()):               #tqdm doesnt do much here as it is one zip file with one csv :P
        # check if it's a file 
#        if not file_info.is_dir():
            # extract the file content
#            with zip_file.open(file_info.filename) as file:
                # reading in the csv file
#                df = pd.read_csv(file)                       
                #print('line54')
#                df.drop(df.columns[df.columns.str.contains('Unnamed: 0',case = False)],axis = 1, inplace = True) #drop unnamed columns. #Weird that it appears. From the cleaning?
                #print(df)


#%% Reading in raw data. Added in own column.
#raw_location = r'C:/Users/Tore Tang/Data FinGrid clean/'+ year + '/' + month + '/'
#raw_filename = 'finland' + '_' + year + '_' + month + '.zip'

#with ZipFile(raw_location + raw_filename, 'r') as zip_file:
    # loop through each file in the zip archive, for later use with several files
#    for raw_file_info in tqdm(zip_file.infolist()):               #tqdm doesnt do much here as it is one zip file with one csv :P
        # check if it's a file 
#        if not raw_file_info.is_dir():
            # extract the file content
#            with zip_file.open(raw_file_info.filename) as raw_file:
                # reading in the csv file
#                raw_df = pd.read_csv(raw_file)                       
                #print('line54')
#                raw_df.drop(df.columns[df.columns.str.contains('Unnamed: 0',case = False)],axis = 1, inplace = True) #drop unnamed columns. #Weird that it appears. From the cleaning?
                #print(raw_df)

#print('df loaded as:')
#print(df)
#print('raw_df loaded as:')
#print(raw_df)

# adding in the raw frequency to the smoothed dataframe. In its own column.
#df = df.assign(Raw_Frequency=raw_df['Frequency'])

#%%trying new finder, gives results.
def calculate_rocof(N, T, rocof_limit,freq_change_limit,year,month): #df

    filename = 'smoothed_'+ year + '_' + month + '.zip'
    location = r'C:/Users/Tore Tang/Data FinGrid smoothed/'+ year + '/' + month + '/'  # location of the zip file

    # Location to save file and plot
    save_to = r'C:/Users/Tore Tang/Data Fingrid RoCoF events/'+year+'/'#+month+'/'
    
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
                    #print('line54')
                    raw_df.drop(df.columns[df.columns.str.contains('Unnamed: 0',case = False)],axis = 1, inplace = True) #drop unnamed columns. #Weird that it appears. From the cleaning?
                    #print(raw_df)


    df = df.assign(Raw_Frequency=raw_df['Frequency'])

    df = df.copy()  # copy to avoid SettingWithCopyWarning (makes numpy happy)

    # Calculate the RoCoF using the symmetric difference quotient
    df['rocof'] = (df['Frequency'].shift(-N) - df['Frequency'].shift(N)) / (2 * T)  # shifting to either side. Gives 2 times the time difference. Therefore divided by 2*1. N=10 gives 1 sec,
                                                                                    # shifting to either side gives 2 sec total. Want the results to be in /s.
    
    # Identify the RoCoF values above the limit.
    df['rocof_event'] = np.abs(df['rocof']) > rocof_limit
    # Identify the start of each event
    
    df['event_start'] = (np.abs(df['rocof']) > 40) #& ~df['rocof_event'].shift(1).fillna(False)

    # Initialize cooldown
    cooldown = 0

    # Initialize part_of_event column
    df['part_of_event'] = float(False)

    for i in range(len(df)):
        # If a RoCoF event is found and the cooldown period has ended
        if df.loc[i, 'rocof_event'] == True and cooldown == 0:
            # Start the cooldown period
            cooldown = 60 * N
            # Set event_start to True
            df.loc[i, 'event_start'] = True
            # Set part_of_event to True for 10*N rows before and 60*N rows after the event
            df.loc[max(0, i - 10 * N) : i - 1, 'part_of_event'] = float(True)
            df.loc[i : i + 60 * N, 'part_of_event'] = float(True)
            
            # Set event_start to True
            #df.loc[i, 'event_start'] = True
        elif cooldown > 0:
        # If in the cooldown period, suppress any RoCoF events
            df.loc[i, 'rocof_event'] = False 
            df.loc[i, 'event_start'] = False    
            #cooldown -= 1
        # Decrement the cooldown counter at each iteration
        cooldown = max(0, cooldown - 1)
    
    # Drop rows where RoCoF event is False
    df = df[df['part_of_event'] == True]  
    # Reset the index
    df = df.reset_index(drop=True) 
    # Reset the index
    #df = df.reset_index(drop=True) 

    ff_rows_to_remove = []
    # Iterate over the DataFrame. Checking backwards for forward filled events.
    for i in range(len(df)):
    # If event_start is True
        if df.loc[i, 'rocof_event'] == True:
        # Look at the previous 10*N rows in the raw_frequency column
            raw_frequency_values = df.loc[max(0, i - 10 * N) : i - 1, 'Raw_Frequency']
        # Check if all values are the same
            if raw_frequency_values.nunique() == 1:
            # Add the indices of the rows to be removed to the list
                ff_rows_to_remove.extend(range(max(0, i - 10 * N), i))
                ff_rows_to_remove.extend(range(i, min(i + 1 + 60 * N, len(df))))
                df = df.reset_index(drop=True)
    # Iterate over the DataFrame. Checking forwards for filled events.
    for i in range(len(df)):
    # If event_start is True
        if df.loc[i, 'rocof_event'] == True:
        # Look at the previous 10*N rows in the raw_frequency column
            raw_frequency_values = df.loc[max(0, i + 10) : i + 50, 'Raw_Frequency']
        # Check if all values are the same
            if raw_frequency_values.nunique() == 1:
            # Add the indices of the rows to be removed to the list
                ff_rows_to_remove.extend(range(max(0, i - 10 * N), i))
                ff_rows_to_remove.extend(range(i, min(i + 1 + 60 * N, len(df))))
                df = df.reset_index(drop=True)

# Drop the rows from the DataFrame
    df = df.drop(ff_rows_to_remove)
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        # If a RoCoF event is found
        if df.loc[i, 'rocof_event'] == True:
            # Find the maximum RoCoF value within the event
            max_rocof_idx = df.loc[i : min(i + 60 * N, len(df)), 'rocof'].abs().idxmax()
            df.loc[i, 'max_rocof'] = df.loc[max_rocof_idx, 'rocof']
    
    # Reset the index before doing a new check for frequency changes inside the event.
    #df = df.reset_index(drop=True) 

    #The frequency inside of the event needs to deviate more than 50 from 3 points mean before the event.
    df['nadir'] = np.nan
    df['nadir_time'] = np.nan
    df['Time'] = pd.to_datetime(df['Time'])
    df['nadir_time'] = pd.to_datetime(df['nadir_time'])
    freq_limit_rows_to_remove=[]
    max_diff = 0
    for i in range(len(df)):
    # If a RoCoF event is found   
        max_diff = 0
        if df.loc[i, 'rocof_event'] == True:
            # Update start_freq at the start of each event
            if df.loc[i, 'event_start'] == True:
                start_freq = df.loc[i, 'Frequency']
                    # Shift the frequency for the current event and the next i + 60 * N rows
                for j in range(i, min(i + 60 * N, len(df))):
                    df.loc[j, 'Shifted_Frequency'] = df.loc[j, 'Frequency'] - start_freq

            avg_prev_freq = df.loc[max(0, i - 3) : i - 1, 'Frequency'].mean()

        # Calculate the absolute difference between avg_prev_freq and min_freq_event and max_freq_event
            min_freq_event = df.loc[i : min(i + 60 * N, len(df)), 'Shifted_Frequency'].min()
            max_freq_event = df.loc[i : min(i + 60 * N, len(df)), 'Shifted_Frequency'].max()
            min_diff = abs(avg_prev_freq - min_freq_event)
            max_diff = abs(avg_prev_freq - max_freq_event)
            max_diff = max(min_diff,max_diff)
            
            # Decide which of the max or min freq event is the biggest
            if abs(min_freq_event) > abs(max_freq_event):
                min_freq_event_idx = df.loc[i : min(i + 60 * N, len(df)), 'Shifted_Frequency'].idxmin()
                print(min_freq_event_idx)
                df.loc[i, 'nadir'] = df.loc[min_freq_event_idx, 'Frequency']  # Use 'Frequency' instead of 'Shifted_Frequency'
                df.loc[i, 'nadir_time'] = df.loc[min_freq_event_idx, 'Time']
            else:
                max_freq_event_idx = df.loc[i : min(i + 60 * N, len(df)), 'Shifted_Frequency'].idxmax()
                print(max_freq_event_idx)
                df.loc[i, 'nadir'] = df.loc[max_freq_event_idx, 'Frequency']  # Use 'Frequency' instead of 'Shifted_Frequency'
                df.loc[i, 'nadir_time'] = df.loc[max_freq_event_idx, 'Time']
                        
            # Check if the maximum difference is greater than or equal to freq_change_limit
            if max_diff < freq_change_limit:
                freq_limit_rows_to_remove.extend(range(max(0, i - 10 * N), i))
                freq_limit_rows_to_remove.extend(range(i, min(i + 1 + 60 * N, len(df))))
                max_diff = 0
                #df = df.reset_index(drop=True) #resetting index to avoid index errors.

    # Drop the rows from the DataFrame
    df = df.drop(freq_limit_rows_to_remove)
    df = df.reset_index(drop=True)
    return df

N=5                        # number of samples in moving average window. 10 points would equal 1 second. In each direction(.shift())
T=0.5                      # duration of the moving average window
rocof_limit = 40           # minimum RoCoF value to be considered an event
freq_change_limit = 35     # minimum change in frequency to be considered an event
#df_rocof = calculate_rocof(df, N, T, rocof_limit,freq_change_limit).copy()           #copy to avoid SettingWithCopyWarning

def plot_event(df_event,event_number):
    fig, ax1 = plt.subplots(figsize=(15, 8))  # Set the figure size here
    save_to = r'C:/Users/Tore Tang/Data Fingrid RoCoF events/'+year+'/'
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency, [mHz]', color=color)
    ax1.plot(df_event['Time'], df_event['Raw_Frequency'], color='yellowgreen', label='Raw Frequency') 
    ax1.plot(df_event['Time'], df_event['Frequency'], color=color, label='Frequency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('RoCoF [mHz/s]', color=color)
    ax2.plot(df_event['Time'], df_event['rocof'], color=color, label='RoCoF')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')  

    # Calculate the maximum RoCoF value and the date and time of the event
    max_rocof = df_event['rocof'].abs().max()
    max_rocof_time = df_event.loc[df_event['rocof'].idxmax(), 'Time']

    # Format the numbers to include only two decimal points
    max_rocof_str = "{:.2f}".format(max_rocof)
    max_rocof_time_str = max_rocof_time.strftime("%Y-%m-%d %H:%M:%S")

    # Add a title to the figure
    plt.title(f'RoCoF event with value {max_rocof_str} [mHz/s] at {max_rocof_time_str}')

    fig.tight_layout()
    #plt.show()

    fig.savefig(os.path.join(save_to,f'{year}_{month}_event_{event_number}.png'))
    # Close the figure to free up memory.
    plt.close(fig)                          





#Running loop for all years and months.
years = ['2019','2020','2021','2022'] #'2016','2017','2018','2019','2020','2021'
#years = ['2023']
for year in years:
    months = ['01','02','03','04','05','06','07','08','09','10','11','12'] #,'03','04','05','06','07','08','09','10','11','12'
    #months = ['10']
    for month in months:
        df_rocof = calculate_rocof(N, T, rocof_limit,freq_change_limit,year,month).copy()           #copy to avoid SettingWithCopyWarning
        save_df_to=r'C:/Users/Tore Tang/Data Fingrid RoCoF events/csvfiles/' + year + '/'#+month+'/'
        df_rocof.to_csv(save_df_to + 'rocofevents_'+ month + '_' + year + '.csv',float_format='%.5f')
        
        #plotting stuff below:

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


# Output to show code is done running
print('done')


# saving:
"""""""""""""""""""""""""""""""""""""""
save_df_to=r'C:/Users/Tore Tang/Data Fingrid RoCoF events/csvfiles/' + year + '/'#+month+'/'
#os.makedirs(os.path.dirname(save_df_to), exist_ok=True)                                      #create the folder if it does not exist.

#saving the dataframe to a csv file:
df_rocof.to_csv(save_df_to + 'rocofevents_'+ month + '_' + year + '.csv',float_format='%.5f') #, originally float_format='%.0f', this changes to 0 decimals. I want 5 so is put to 5.


save_to = r'C:/Users/Tore Tang/Data Fingrid RoCoF events/'+year+'/'#+month+'/'
print(df_rocof)

# Plot the frequency and RoCoF events
import matplotlib.pyplot as plt

def plot_event(df_event,event_number):
    fig, ax1 = plt.subplots(figsize=(15, 8))  # Set the figure size here

    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency, [mHz]', color=color)
    ax1.plot(df_event['Time'], df_event['Raw_Frequency'], color='yellowgreen', label='Raw Frequency') 
    ax1.plot(df_event['Time'], df_event['Frequency'], color=color, label='Frequency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('RoCoF [mHz/s]', color=color)
    ax2.plot(df_event['Time'], df_event['rocof'], color=color, label='RoCoF')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')  

    # Calculate the maximum RoCoF value and the date and time of the event
    max_rocof = df_event['rocof'].abs().max()
    max_rocof_time = df_event.loc[df_event['rocof'].idxmax(), 'Time']

    # Format the numbers to include only two decimal points
    max_rocof_str = "{:.2f}".format(max_rocof)
    max_rocof_time_str = max_rocof_time.strftime("%Y-%m-%d %H:%M:%S")

    # Add a title to the figure
    plt.title(f'RoCoF event with value {max_rocof_str} [mHz/s] at {max_rocof_time_str}')

    fig.tight_layout()
    #plt.show()

    fig.savefig(os.path.join(save_to,f'{year}_{month}_event_{event_number}.png'))
    # Close the figure to free up memory.
    plt.close(fig)                          

#plotting stuff below:

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


# Output to show code is done running
print('done')
"""""""""""""""""""""""""""""""""""""""