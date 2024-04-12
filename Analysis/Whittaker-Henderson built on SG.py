#Whittaker-Henderson built on Savitzky Golay


#%% packs, not all used

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
from WhittakerHendersonSmoother import WhittakerHendersonSmoother


# Specifying years, month and location of the file. saveplace used to identify the name of the file.
# could be done in one line but this seemed cleaner





year = '2023'
month = '01'
#years=['2018']
#months = ['01','02','03','04','05','06','07','08','09','10','11','12']

#for year in years:
#    for month in months:



saveplace = 'finland_'+ year + '_' + month + '.zip'
location = r'C:/Users/Tore Tang/Data FinGrid clean/'+ year + '/' + month + '/'

# location to save file and plot
save_to = r'C:/Users/Tore Tang/Data FinGrid smoothed/'+year+'/'+month+'/'

#Whittaker-Henderson smoother class
# Adapted code from javacode in the paper "Why and How Savitzky–Golay Filters Should Be Replaced"
# https://pubs.acs.org/doi/epdf/10.1021/acsmeasuresciau.1c00054
# Look for it in the WhittakerHenderson smoother class

# Static helper methods of the class includes several static helper methods       # Static methods can be used for caching or memoization purposes. When you need to store and reuse calculated results, 
# (make_dprime_d, times_lambda_plus_ident, cholesky_l, solve)                     # static methods can help maintain a cache within the class, making subsequent calculations more efficient.
# are used internally for matrix manipulation and solving linear systems.

#%%Loading the dataframe in from the zip file.

#df = pd.DataFrame()
with ZipFile(location + saveplace, 'r') as zip:
# loop through each file in the zip archive, for later use with several files
    for file_info in tqdm(zip.infolist()):               #tqdm doesnt do much here as it is one zip file with one csv :P
    # check if it's a file 
        if not file_info.is_dir():
        # extract the file content
            with zip.open(file_info.filename) as file:
            # reading in the csv file
                df = pd.read_csv(file)                       
            #print('line54')
                df.drop(df.columns[df.columns.str.contains('Unnamed: 0',case = False)],axis = 1, inplace = True) #drop unnamed columns. #Weird that it appears. From the cleaning?
            #print(df)
            # Concatenate the data to the empty DataFrame
            #df = pd.concat([df, data], ignore_index=True)      #close this? -erlend
            




# Sort the DataFrame by datetime:(should already be sorted, no need to do it again)
# Added because as the paper stated, "This is necessary because the smoothing algorithm assumes equally spaced points."
# But need to rename the first column.

df.rename({'Time':'datetime'}, axis='columns',inplace=True) #,inplace=True #change the name of the first column to 'datetime'. Had issues with "Unnamed 0" and Time appearing. As 2 columns.
#df = df.sort_values(by='datetime')


# Printing again to check that the dataframe looks good.
#print('printing place where the issue with original dataframe occured(should read -8.14 for time 2023-12-01 06:11:38.300 till 2023-12-01 06:11:39.700):')
#print(df)

#print(df[222970:223000])
#print('This is the dataframe read in')
#print(df)

#%% Prepare the data for smoothing

# Extract the 'frequency' column as a NumPy array. 
# If needed later, convert the datetime column to numeric values (e.g., days since the first date). #not needed imo.

#frequency_data = df['Frequency'].values
#frequency_data_limited=frequency_data[0:200000]  #extract some of the frequency data, so the script can run faster. For fail searching purposes.
#time_values = (df['datetime'] - df['datetime'].min()).dt.days.values  #not used, but could be used for time values if needed later.

# Checking data again
#print('printing the frequency data and limited frequency data:')
#print(frequency_data)
#print(frequency_data_limited[100:120])
#print(frequency_data_limited[10000:12000])


#%% Whittaker-Henderson Smoother
# Choose the order of the penalty derivative (order) and the smoothing parameter (lambda_val).

#order = 1
#lambda_val = 1  # Adjust as needed #main point for it not running correctly. Was set at 1000 :OO
#start=time.time()
#Input len of freq data, and penalty order and smoothing parameter (lambda_val).
#smoother = WhittakerHendersonSmoother(len(frequency_data_limited), order, lambda_val)  #ta inn listen ovenfor også?
#smoothed_data = smoother.smooth(frequency_data_limited)                                #assumin the return is also with correct freq to date?
#end=time.time()

#print('The smoothing took seconds to complete.')
#print(end - start)   
#print('printing the smoothed data:')
#print(smoothed_data)

#%%trying parallel processing outside of the class.

from multiprocessing import Pool

def smooth_data(chunk, order, lambda_val):
    smoother = WhittakerHendersonSmoother(len(chunk), order, lambda_val)
    return smoother.smooth(chunk)

if __name__ == '__main__':
    start=time.time()
    order = 2
    lambda_val = 4
    frequency_data = df['Frequency'].values
    #frequency_data_limited=frequency_data[0:25918816]
    #print(frequency_data_limited[5000:5010])
    #print(frequency_data_limited)
    #print('line121')

    # Split your data into chunks
    chunk_size = 10000
    chunks = [(frequency_data[i:i + chunk_size], order, lambda_val) for i in range(0, len(frequency_data), chunk_size)]

    # Use a process pool to smooth the chunks in parallel
    with Pool(3) as pool:
        smoothed_chunks = pool.starmap(smooth_data, chunks) #could also implement chunck_size= here, but it is already implemented in the chunks list.

    # gather the smoothed chunks
    smoothed_data = np.concatenate(smoothed_chunks)
    end=time.time()
    print('The smoothing took seconds to complete.')
    print(end - start)
    #print('line 126')
    #print(smoothed_data[5000:5010])

        # Create a new DataFrame with the smoothed data and the corresponding datetime index.
    smoothed_df = pd.DataFrame({
        'Time': df['datetime'][:len(smoothed_data)],
        'Frequency': smoothed_data
    }, columns=['Time', 'Frequency'])
    #print(smoothed_df[5000:5010])



    # %% Save data into a zipped csv. location is save_to    #took away .csv.zip to just .zip

    if not os.path.exists(save_to):
    # If not, create the directory
        os.makedirs(save_to)
    start=time.time()
    smoothed_df.to_csv(save_to + 'smoothed_' + year+'_'+month+'.zip',float_format='%.5f', #, originally float_format='%.0f', this changes to 0 decimals. I want 5 so is put to 5.
    compression=dict(method='zip', archive_name='smoothed_'+year+'_'+month+'.csv'))
    end=time.time()

    print('The saving took seconds to complete.')
    print(end - start)
#Leaving out figure when running for loop   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    fig = go.Figure()
#    N = 12400  # or whatever number of points you want to plot

# trace for the original data
#    fig.add_trace(go.Scatter(x=np.arange(N), y=frequency_data[-N:],
#                        mode='lines', name='Original data'))

# trace for the smoothed data
#    fig.add_trace(go.Scatter(x=np.arange(N), y=smoothed_data[-N:],
#                        mode='lines', name='Smoothed data'))




# trace for the original data
#    fig.add_trace(go.Scatter(x=np.arange(len(frequency_data_limited[26771599:26783999])), y=frequency_data_limited,
#                     mode='lines', name='Original data'))

# trace for the smoothed data
#    fig.add_trace(go.Scatter(x=np.arange(len(smoothed_data[26771599:26783999])), y=smoothed_data,
#                     mode='lines', name='Smoothed data'))

# update layout if needed
#    fig.update_layout(title='Combined Plot')





# show the combined figure
#    fig.show()




#NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -leo
#executor map (pool?) 


#%% Plotly. Create a figure for the original data and smoothed data 
#fig = go.Figure()

# trace for the original data
#fig.add_trace(go.Scatter(x=np.arange(len(frequency_data_limited[100000:130000])), y=frequency_data_limited,
#                         mode='lines', name='Original data'))

# trace for the smoothed data
#fig.add_trace(go.Scatter(x=np.arange(len(smoothed_data[100000:130000])), y=smoothed_data,
#                         mode='lines', name='Smoothed data'))

# update layout if needed
#fig.update_layout(title='Combined Plot')

# show the combined figure
#fig.show()


#%% Recombine the smoothed data with the according timedate data.

#dummy = df[0:10000]

# Create a new dataframe with 'datetime' and 'smoothed_data' for saving testing purposes
#smoothed_df = pd.DataFrame({'datetime': dummy['datetime'], 'smoothed_data': smoothed_data})

#print(smoothed_data)


#%% Save the smoothed data to a csv file in a Zipped file.
#Save data into a zipped csv. location is save_to
#smoothed_df.to_csv(save_to + 'smoothed_' + year+'_'+month+'.csv.zip',   # float_format='%.0f', made the csv file saved as rounded off numbers. Did this happen in the cleaning as well
#    compression=dict(method='zip', archive_name='smoothed_' + year+'_'+month+'.csv'))

#NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#np.savez_compressed() #use this to compress and open files -leo
