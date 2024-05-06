# Whittaker-Henderson smoother class
# Adapted code from Java in the paper "Why and How Savitzky–Golay Filters Should Be Replaced"
# https://pubs.acs.org/doi/epdf/10.1021/acsmeasuresciau.1c00054

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
from multiprocessing import Pool

#Choose which year and month you would like to smooth.

year = '2023'
month = '01'
#years=['2018']
#months = ['01','02','03','04','05','06','07','08','09','10','11','12']

#Change these location to your desired ones.
saveplace = 'finland_'+ year + '_' + month + '.zip'
location = r'your_folder'+ year + '/' + month + '/'

# location to save file and plot
save_to = r'your_folder/'+year+'/'+month+'/'

# Loading the dataframe in from the zip file.

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
            


df.rename({'Time':'datetime'}, axis='columns',inplace=True) #,inplace=True #change the name of the first column to 'datetime'. Had issues with "Unnamed 0" and Time appearing. As 2 columns.

# Choose the order of the penalty derivative (order) and the smoothing parameter (lambda_val).

def smooth_data(chunk, order, lambda_val):
    smoother = WhittakerHendersonSmoother(len(chunk), order, lambda_val)
    return smoother.smooth(chunk)

if __name__ == '__main__':
    start=time.time()
    order = 2
    lambda_val = 4
    frequency_data = df['Frequency'].values

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

    # Create a new DataFrame with the smoothed data and the corresponding datetime index.
    smoothed_df = pd.DataFrame({
        'Time': df['datetime'][:len(smoothed_data)],
        'Frequency': smoothed_data
    }, columns=['Time', 'Frequency'])



    # %% Save data into a zipped csv.

    if not os.path.exists(save_to):
    # If not, create the directory
        os.makedirs(save_to)
    start=time.time()
    smoothed_df.to_csv(save_to + 'smoothed_' + year+'_'+month+'.zip',float_format='%.5f', #, originally float_format='%.0f', this changes to 0 decimals. I want 5 so is put to 5.
    compression=dict(method='zip', archive_name='smoothed_'+year+'_'+month+'.csv'))
    end=time.time()

    print('The saving took seconds to complete.')
    print(end - start)
    
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# The code below is commented out. But have been used to visualize the smoothing process. This enables tuning of the parameters.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
