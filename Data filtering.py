#This python script is built on the 'Base filtring.py' script provided by the supervisor of this thesis, 

# This is a manual cleaner script for the data. You need to manually change the
# months, years, and occasional details. We will not provide an automated script
# since one should carefully check the data each month to ensure there are no
# holes, weird effects, or others.

# This is the cleaner for the Finnish data from FinGrid
# https://data.fingrid.fi/en/dataset/frequency-historical-data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
import py7zr
import lzma
import os
import zipfile
from io import BytesIO
import re


# # European Data Cleaner
# ## Load Data Sets

# Location of the file
location = r'C:/Users/Tore Tang/Data FinGrid/'
# Year
year = r'2023'
# Month
month = r'12'
month_index = int(month) - 1
# File name
file_name = year + '/' + year + '-' + month + '/' + year + '-' + month

# location to save file and plot
save_to = r'C:/Users/Tore Tang/Data FinGrid clean/'+year+'/'+month+'/'

# Date ranges
dates = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']

# Check if leap year
if year in ['2008', '2012', '2016', '2020']:
    start_date = [year+'-01-01 00:00:00', year+'-02-01 00:00:00', year+'-03-01 00:00:00', year+'-04-01 00:00:00', year+'-05-01 00:00:00', year+'-06-01 00:00:00', year+'-07-01 00:00:00', year+'-08-01 00:00:00', year+'-09-01 00:00:00', year+'-10-01 00:00:00',year+'-11-01 00:00:00',year+'-12-01 00:00:00' ]
    end_date = [year+'-01-31 23:59:59', year+'-02-29 23:59:59', year+'-03-31 23:59:59', year+'-04-30 23:59:59', year+'-05-31 23:59:59', year+'-06-30 23:59:59', year+'-07-31 23:59:59', year+'-08-31 23:59:59', year+'-09-30 23:59:59', year+'-10-31 23:59:59',year+'-11-30 23:59:59',year+'-12-31 23:59:59' ]
else:
    start_date = [year+'-01-01 00:00:00', year+'-02-01 00:00:00', year+'-03-01 00:00:00', year+'-04-01 00:00:00', year+'-05-01 00:00:00', year+'-06-01 00:00:00', year+'-07-01 00:00:00', year+'-08-01 00:00:00', year+'-09-01 00:00:00', year+'-10-01 00:00:00',year+'-11-01 00:00:00',year+'-12-01 00:00:00' ]
    end_date = [year+'-01-31 23:59:59', year+'-02-28 23:59:59', year+'-03-31 23:59:59', year+'-04-30 23:59:59', year+'-05-31 23:59:59', year+'-06-30 23:59:59', year+'-07-31 23:59:59', year+'-08-31 23:59:59', year+'-09-30 23:59:59', year+'-10-31 23:59:59',year+'-11-30 23:59:59',year+'-12-31 23:59:59' ]

# locate days of recordings
idx = pd.date_range(start_date[month_index], end_date[month_index], freq = 'D').day

# specify the directory where your .zip files are located
dir_path = location + year + '/'+ year + '-' + month # example, r'C:/Users/Tore Tang/Data FinGrid/2023/2023-08'

# create an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# extract the date from the filename
date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
start=time.time()

# loop through all files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.zip'): #2023 have .7z files
        # construct the full file path
        file_path = os.path.join(dir_path, filename)
        
        # open the zip file
        with zipfile.ZipFile(file_path, 'r') as z:
            # loop through each file in the zip file
            for subfile in tqdm(z.namelist()):
                if subfile.endswith('.csv'):
                    # extract the date from the filename using the regular expression
                    match = date_pattern.search(subfile)
                    if match:
                        date_str = match.group(1)
                        
                        # read the CSV file into a DataFrame
                        with z.open(subfile) as f:
                            df = pd.read_csv(BytesIO(f.read()))
                        
                        # append the DataFrame to the combined DataFrame
                        combined_df = combined_df._append(df, ignore_index=True)
end=time.time()
print(end-start)
# Print the combined DataFrame
print(combined_df)

100%|██████████| 31/31 [00:22<00:00,  1.36it/s]

22.883978366851807
                             Time     Value
0         2023-12-01 00:00:00.000  50.03876
1         2023-12-01 00:00:00.100  50.03952
2         2023-12-01 00:00:00.200  50.03891
3         2023-12-01 00:00:00.300  50.03868
4         2023-12-01 00:00:00.400  50.03822
...                           ...       ...
26776745  2023-12-31 23:59:59.500  50.00964
26776746  2023-12-31 23:59:59.600  50.00930
26776747  2023-12-31 23:59:59.700  50.01034
26776748  2023-12-31 23:59:59.800  50.01019
26776749  2023-12-31 23:59:59.900  50.01012

[26776750 rows x 2 columns]

print(combined_df)
#renaming beacuse of problems with datetime further down..
combined_df = combined_df.rename({'Time':0, 'Value':1}, axis='columns')
print(combined_df)

                             Time     Value
0         2023-12-01 00:00:00.000  50.03876
1         2023-12-01 00:00:00.100  50.03952
2         2023-12-01 00:00:00.200  50.03891
3         2023-12-01 00:00:00.300  50.03868
4         2023-12-01 00:00:00.400  50.03822
...                           ...       ...
26776745  2023-12-31 23:59:59.500  50.00964
26776746  2023-12-31 23:59:59.600  50.00930
26776747  2023-12-31 23:59:59.700  50.01034
26776748  2023-12-31 23:59:59.800  50.01019
26776749  2023-12-31 23:59:59.900  50.01012

[26776750 rows x 2 columns]
                                0         1
0         2023-12-01 00:00:00.000  50.03876
1         2023-12-01 00:00:00.100  50.03952
2         2023-12-01 00:00:00.200  50.03891
3         2023-12-01 00:00:00.300  50.03868
4         2023-12-01 00:00:00.400  50.03822
...                           ...       ...
26776745  2023-12-31 23:59:59.500  50.00964
26776746  2023-12-31 23:59:59.600  50.00930
26776747  2023-12-31 23:59:59.700  50.01034
26776748  2023-12-31 23:59:59.800  50.01019
26776749  2023-12-31 23:59:59.900  50.01012

[26776750 rows x 2 columns]

# Merge dates and times to make a DateTime format. Rename frequency column

#combined_df[0] =  pd.to_datetime(combined_df[0])
#combined_df = combined_df.rename({'Value':'Frequency'}, axis='columns')


#Old code
combined_df[0] =  pd.to_datetime(combined_df[0])
combined_df = combined_df.rename({0:'Time', 1:'Frequency'}, axis='columns')

# Showing head and tail of the dataframe. 
combined_df.tail
combined_df.head

<bound method NDFrame.head of                             Time  Frequency
0        2023-12-01 00:00:00.000   50.03876
1        2023-12-01 00:00:00.100   50.03952
2        2023-12-01 00:00:00.200   50.03891
3        2023-12-01 00:00:00.300   50.03868
4        2023-12-01 00:00:00.400   50.03822
...                          ...        ...
26776745 2023-12-31 23:59:59.500   50.00964
26776746 2023-12-31 23:59:59.600   50.00930
26776747 2023-12-31 23:59:59.700   50.01034
26776748 2023-12-31 23:59:59.800   50.01019
26776749 2023-12-31 23:59:59.900   50.01012

[26776750 rows x 2 columns]>

# Here we remove 50 Hz from the frequency, since it is common to work in a
# reference frame where the nominal frequency is 0 Hz (useful to compare US
# and EU data)
combined_df['Frequency'] = (combined_df['Frequency'] - 50.)*1000 # 60.0 for US and Japan

# use pandas to clean the timeseries.
## First, drop all duplicates entries
combined_df = combined_df.drop_duplicates(subset='Time')

## Now ensure the first entry is the first second of the month and the last
## the last second of the month.

idx = pd.date_range(start_date[month_index], end_date[month_index], freq = '100ms')

#combined_df = combined_df.set_index('Time').rename_axis('datetime')
#combined_df = combined_df.reindex(idx, fill_value=np.nan) #dont want to fill with nan values!!

#New insertion method for datetime and fillforward

import pandas as pd


# Convert the "datetime" column to datetime type if it's not already. Already done above.
# df['datetime'] = pd.to_datetime(df['datetime'])

# Set the "datetime" column as the index
combined_df = combined_df.set_index('Time')
print('Before resampling')
print(combined_df[2398968:2398978])
print(combined_df[2398978:2398998])

# Resample the dataframe to create new rows for missing 0.1-second intervals
df_resampled = combined_df.resample('0.1S').asfreq()
print('After resampling')
print(df_resampled[2398968:2398978])
print(df_resampled[2398978:2398998])

#below here: testing new code for countign fowar filling.

# Create a new column that indicates whether the 'Frequency' value is NaN
df_resampled['filled'] = df_resampled['Frequency'].isna()

#original code below
# Fill missing frequency values with the previous valid frequency value
df_resampled['Frequency'] = df_resampled['Frequency'].ffill()
print('After resampling and forward filling')
print(df_resampled[2398968:2398978])
print(df_resampled[2398978:2398998])

#below here: testing new code for countign fowar filling.
# Calculate the total number of times forward filling has happened
total_fill_count = df_resampled['filled'].sum()

# Calculate the percentage of data that has been forward filled
fill_percentage = (total_fill_count / len(df_resampled)) * 100

#below here: testing new code for countign fowar filling.
# Drop the 'filled' column as it's no longer needed
df_resampled = df_resampled.drop(columns='filled')

#original code below
# Reset the index to make "datetime" a column again and reset the index
df_resampled = df_resampled.reset_index()

# Add these values to the last row
df_resampled.loc[df_resampled.index[-1], 'total_fill_count'] = total_fill_count
df_resampled.loc[df_resampled.index[-1], 'fill_percentage'] = fill_percentage

df_resampled.tail()

Before resampling
                         Frequency
Time                              
2023-12-03 18:38:55.400      -3.25
2023-12-03 18:38:55.500      -4.21
2023-12-03 18:38:55.600      -4.13
2023-12-03 18:38:55.700      -4.01
2023-12-03 18:38:55.800      -4.04
2023-12-03 18:38:55.900      -3.96
2023-12-03 18:38:56.000      -4.28
2023-12-03 18:38:56.100      -4.83
2023-12-03 18:38:56.200      -5.49
2023-12-03 18:38:56.300      -5.48
                         Frequency
Time                              
2023-12-03 18:38:56.400      -5.81
2023-12-03 18:38:56.500      -6.61
2023-12-03 18:38:56.600      -6.64
2023-12-03 18:38:56.700      -6.88
2023-12-03 18:38:56.800      -7.59
2023-12-03 18:38:56.900      -6.94
2023-12-03 18:38:57.000      -7.40
2023-12-03 18:38:57.100      -8.83
2023-12-03 18:38:57.200      -8.48
2023-12-03 18:38:57.300      -9.04
2023-12-03 18:38:57.400      -9.10
2023-12-03 18:38:57.500      -8.62
2023-12-03 18:38:57.600      -9.10
2023-12-03 18:38:57.700      -9.01
2023-12-03 18:38:57.800      -8.37
2023-12-03 18:38:57.900      -8.87
2023-12-03 18:38:58.000      -8.75
2023-12-03 18:38:58.100      -7.78
2023-12-03 18:38:58.200      -8.79
2023-12-03 18:38:58.300      -8.29
After resampling
                         Frequency
Time                              
2023-12-03 18:38:16.800     -22.39
2023-12-03 18:38:16.900     -22.16
2023-12-03 18:38:17.000     -22.34
2023-12-03 18:38:17.100     -22.11
2023-12-03 18:38:17.200     -21.58
2023-12-03 18:38:17.300     -20.79
2023-12-03 18:38:17.400     -20.65
2023-12-03 18:38:17.500     -19.30
2023-12-03 18:38:17.600     -19.04
2023-12-03 18:38:17.700     -18.38
                         Frequency
Time                              
2023-12-03 18:38:17.800     -17.86
2023-12-03 18:38:17.900     -17.36
2023-12-03 18:38:18.000     -17.43
2023-12-03 18:38:18.100     -17.33
2023-12-03 18:38:18.200     -16.39
2023-12-03 18:38:18.300     -16.39
2023-12-03 18:38:18.400     -17.02
2023-12-03 18:38:18.500     -17.24
2023-12-03 18:38:18.600     -16.66
2023-12-03 18:38:18.700     -17.48
2023-12-03 18:38:18.800     -17.10
2023-12-03 18:38:18.900     -17.57
2023-12-03 18:38:19.000     -18.37
2023-12-03 18:38:19.100     -18.04
2023-12-03 18:38:19.200     -17.83
2023-12-03 18:38:19.300     -17.57
2023-12-03 18:38:19.400     -17.39
2023-12-03 18:38:19.500     -17.17
2023-12-03 18:38:19.600     -17.10
2023-12-03 18:38:19.700     -15.99
After resampling and forward filling
                         Frequency  filled
Time                                      
2023-12-03 18:38:16.800     -22.39   False
2023-12-03 18:38:16.900     -22.16   False
2023-12-03 18:38:17.000     -22.34   False
2023-12-03 18:38:17.100     -22.11   False
2023-12-03 18:38:17.200     -21.58   False
2023-12-03 18:38:17.300     -20.79   False
2023-12-03 18:38:17.400     -20.65   False
2023-12-03 18:38:17.500     -19.30   False
2023-12-03 18:38:17.600     -19.04   False
2023-12-03 18:38:17.700     -18.38   False
                         Frequency  filled
Time                                      
2023-12-03 18:38:17.800     -17.86   False
2023-12-03 18:38:17.900     -17.36   False
2023-12-03 18:38:18.000     -17.43   False
2023-12-03 18:38:18.100     -17.33   False
2023-12-03 18:38:18.200     -16.39   False
2023-12-03 18:38:18.300     -16.39   False
2023-12-03 18:38:18.400     -17.02   False
2023-12-03 18:38:18.500     -17.24   False
2023-12-03 18:38:18.600     -16.66   False
2023-12-03 18:38:18.700     -17.48   False
2023-12-03 18:38:18.800     -17.10   False
2023-12-03 18:38:18.900     -17.57   False
2023-12-03 18:38:19.000     -18.37   False
2023-12-03 18:38:19.100     -18.04   False
2023-12-03 18:38:19.200     -17.83   False
2023-12-03 18:38:19.300     -17.57   False
2023-12-03 18:38:19.400     -17.39   False
2023-12-03 18:38:19.500     -17.17   False
2023-12-03 18:38:19.600     -17.10   False
2023-12-03 18:38:19.700     -15.99   False

	Time 	Frequency 	total_fill_count 	fill_percentage
26783995 	2023-12-31 23:59:59.500 	9.64 	NaN 	NaN
26783996 	2023-12-31 23:59:59.600 	9.30 	NaN 	NaN
26783997 	2023-12-31 23:59:59.700 	10.34 	NaN 	NaN
26783998 	2023-12-31 23:59:59.800 	10.19 	NaN 	NaN
26783999 	2023-12-31 23:59:59.900 	10.12 	7254.0 	0.027083

# Showing head and tail of the dataframe. 
print(df_resampled[222980:223010])
combined_df.tail
combined_df.head

# Plot a 'quality plot' with the jumps, fluctuations and dead zones

fig, ax = plt.subplots(1,1, figsize=(12,3))
ax.plot(df_resampled['Frequency'].values, color='black')

from matplotlib.patches import Patch
patch_l = Patch(color='gray', label='Quality of data (impossible to determine)')
fig.text(0.09,0.8, r'Decimals = 0', fontsize=16)
ax.set_ylim([-550,650])
ax.set_yticks([-400,-200,0,200,400])
ax.set_xlabel('Time', fontsize = 18); ax.set_ylabel('F [mHz]', fontsize = 18)
ax.legend(handles=[patch_l], loc=4, ncol=4,fontsize = 14)
fig.subplots_adjust(left=0.07, bottom=0.18, right=.99, top=0.99)
#fig.savefig(save_to + year + '_' + month + '.png', dpi = 400, transparent=True)

# %% Save data into a zipped csv. location is save_to 

# Check if the directory exists
if not os.path.exists(save_to):
    # If not, create the directory
    os.makedirs(save_to)

df_resampled.to_csv(save_to + 'finland_' + year+'_'+month+'.zip',float_format='%.5f', #, originally float_format='%.0f', this changes to 0 decimals. I want 5 so is put to 5.
    compression=dict(method='zip', archive_name=year+'_'+month+'.csv'))

