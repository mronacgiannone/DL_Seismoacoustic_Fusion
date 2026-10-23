'''
KoreaGeonet - an array-based seismoacoustic deep learning fusion model

Author: Miro Ronac Giannone [mronacgiannone@smu.edu]
Co-authors: Stephen Arrowsmith [sarrowsmith@smu.edu] & Junghyun Park [junghyunp@smu.edu]
----------------------------------------------------------------------------------------------------------------------
This script lays the framework for a deep learning fusion model using seismoacoustic research arrays within the Korean peninsula.
Current capabilites are limited to discriminating between surface explosions and earthquakes using 3 arrays [BRDAR, CHNAR, and KSGAR] within the network.

Necessary files:
    Korea seismic velocity model - "kigam.tvel"
    Seismic and infrasound site files
    Surface explosion events (KIGAM - Che et al., 2019)
    Earthquake events (Korea Meteorological Administration & Han et al., 2023)
'''

import datetime, utm, dask, warnings, cartopy, os, random, math
#-----------------------------------------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#-----------------------------------------------------------------------------------------------------------------------#
from pyproj import Geod
from scipy import signal
from itertools import groupby
from obspy.core import AttribDict
from dask.distributed import Client
from pisces.tables.css3 import Wfdisc
from tensorflow.keras import backend as K
from array_analysis import array_processing
from obspy import read, Stream, UTCDateTime
from matplotlib.colors import ListedColormap
from obspy.taup import taup_create, TauPyModel
from sklearn.preprocessing import StandardScaler
from matplotlib.dates import date2num, set_epoch
from statsmodels.stats.contingency_tables import mcnemar
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#-----------------------------------------------------------------------------------------------------------------------#
# ML Packages 
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
#-----------------------------------------------------------------------------------------------------------------------#
# Using the original matplotlib epoch
set_epoch('0000-12-31T00:00:00')
#-----------------------------------------------------------------------------------------------------------------------#
warnings.filterwarnings("ignore")

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############## Global Variables ###############
############### ############### ###############

# Geographic coordinates of arrays
BRD_coords = [37.9709, 124.6519]
CHN_coords = [38.2773, 127.1228]
KSG_coords = [38.5954, 128.3519]

# Define Geod for Great Circle computations
g = Geod(ellps='sphere')
'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############# Ancillary Functions #############
############### ############### ###############

def YMD_to_JD(YMD, date):
    '''---------------------------------------------------------------------------
    Converts calendar date from YYYY/MM/DD format to julian day and year.
    ---------------------------------------------------------------------------'''
    dt = datetime.datetime.strptime(date, YMD)
    tt = dt.timetuple()
    julian_day = tt.tm_yday; year = dt.year
    #-----------------------------------------------------------------------------------------------------------------------#
    return julian_day, year

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def HMS_to_S(HMS, time):
    '''---------------------------------------------------------------------------
    Converts HH:MM:SS input to total seconds.
    ---------------------------------------------------------------------------'''
    dt = datetime.datetime.strptime(time, HMS)
    hrs_to_secs = dt.hour * 60 * 60
    mins_to_secs = dt.minute * 60
    time_in_seconds = hrs_to_secs + mins_to_secs + dt.second
    #-----------------------------------------------------------------------------------------------------------------------#
    return time_in_seconds

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def read_wfdisc(wfdisc_file,dir):
    '''---------------------------------------------------------------------------
    Description: Function to read wfdisc file returns an obspy stream.
    Note: Put the wfdisc and mwf files in appropriate directory
    
    Input:
        wfdisc_file (str): path of the wfdisc file
    Return:
        st: obspy stream
    ---------------------------------------------------------------------------'''

    tr_list = []

    with open(wfdisc_file, 'r') as f:
        for row in f:
            sta = row[0:6].strip()
            chan = row[7:15].strip()
            time = float(row[16:33].strip())
            wfid = int(row[34:42].strip())
            chanid = int(row[43:51].strip())
            jdate = int(row[52:60].strip())
            endtime = float(row[61:78].strip())
            nsamp = int(row[79:87].strip())
            samprate = float(row[88:99].strip())
            calib = float(row[100:116].strip())
            calper = row[117:133].strip()
            instype = row[134:140].strip()
            segtype = row[141:142].strip()
            datatype = row[143:145].strip()
            clip = row[146:147].strip()
            #dir = row[148:149].strip()
            dir = dir
            dfile = row[213:245].strip()
            foff = int(row[246:256].strip())
            commid = int(row[257:265].strip())
            #lddate = UTCDateTime(row[266:283].strip())
            lddate = UTCDateTime(1697589675.00000)
            try:
                wf = Wfdisc(
                    calib, calper, chan, chanid, clip, commid, datatype,
                    dfile, dir, endtime, foff, instype, jdate, lddate,
                    nsamp, samprate, segtype, sta, time, wfid) # Old
            except:
                wf = Wfdisc(
                    sta, chan, time, wfid, commid)
            
            tr = wf.to_trace()
            tr_list.append(tr)
            # break
            
    st = Stream(tr_list)
    
    return st

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def read_data(julian_day, year, month, day, arrays, data='seismic'):
    '''---------------------------------------------------------------------------
    Pulls seismic/infrasound data from Korean seismoacoustic arrays.
    ---------------------------------------------------------------------------'''
    st = Stream()
    #-----------------------------------------------------------------------------------------------------------------------#
    for array in arrays:
        # Pulling data from directory
        try:
            if julian_day < 10:
                st_tmp = read('/data/Korea/KoreaActive/'+str(year)+'/00'+str(julian_day)+'/'+array+'*.wfdisc')
            elif 10 <= julian_day < 100:
                st_tmp = read('/data/Korea/KoreaActive/'+str(year)+'/0'+str(julian_day)+'/'+array+'*.wfdisc')
            else:
                st_tmp = read('/data/Korea/KoreaActive/'+str(year)+'/'+str(julian_day)+'/'+array+'*.wfdisc')
        except Exception as inst:
            print(inst)
            try:
                if julian_day < 10:
                    data_str = '/data/Korea/KoreaActive/'+str(year)+'/00'+str(julian_day)+'/'+array+'AR_ALL.'+str(year)+str(month)+str(day)+'.0000.wfdisc'
                    dir_input = '/data/Korea/KoreaActive/'+str(year)+'/00'+str(julian_day)+'/'
                elif 10 <= julian_day < 100:
                    data_str = '/data/Korea/KoreaActive/'+str(year)+'/0'+str(julian_day)+'/'+array+'AR_ALL.'+str(year)+str(month)+str(day)+'.0000.wfdisc'
                    dir_input = '/data/Korea/KoreaActive/'+str(year)+'/0'+str(julian_day)+'/'
                else:
                    data_str = '/data/Korea/KoreaActive/'+str(year)+'/'+str(julian_day)+'/'+array+'AR_ALL.'+str(year)+str(month)+str(day)+'.0000.wfdisc'
                    dir_input = '/data/Korea/KoreaActive/'+str(year)+'/'+str(julian_day)+'/'
                    print(data_str)
                    print(dir_input)
                st_tmp = read_wfdisc(data_str,dir_input)
            except Exception as inst:
                print(inst)
                continue
        #-----------------------------------------------------------------------------------------------------------------------#
        if data == 'seismic':
            # Selecting seismic stations from stream
            st_merge = st_tmp.select(channel='BH*')
        elif data == 'infrasound':
            # Selecting infrasound stations from stream
            st_merge = st_tmp.select(channel='BDF')
        #-----------------------------------------------------------------------------------------------------------------------#
        try: 
            st_merge = st_merge.merge()
        except: 
            pass
        for tr in st_merge:
            tr.stats.network = 'XX' # need to do this to match station info on location file
            st.append(tr)
    #-----------------------------------------------------------------------------------------------------------------------#
    return st

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def append_loc_info(stream, df, array):
    '''---------------------------------------------------------------------------
    Appends the location metadata to desired Korean seismoacoustic array.
    ---------------------------------------------------------------------------'''
    st = Stream()
    for tr in stream.select(station=array+'*'):
        st.append(tr)
    st_LOC = Stream(); df_asarray = np.asarray(df)
    for tr in st:
        if tr.id in df_asarray:
            st_LOC.append(tr)
        else:
            continue
    #-----------------------------------------------------------------------------------------------------------------------#
    # Constructing station metadata
    for tr in st_LOC:
        index = np.where(df['stn'] == tr.stats.station)[0]
        sacAttrib = AttribDict({"stla": df['lat'][index],
        "stlo": df['lon'][index]})
        tr.stats.sac = sacAttrib
    for tr in st_LOC:
        lat = (df[df['stn'] == tr.id]['lat']).values[0]
        lon = (df[df['stn'] == tr.id]['lon']).values[0]
        tr.stats.sac.stla = lat
        tr.stats.sac.stlo = lon
        tr.stats.sac.stel = 0
    #----------------------------------------------------------------------------------------------------------------------#
    return st_LOC

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def convert_database_datetimes_to_JD(database_filename):
    '''---------------------------------------------------------------------------
    Converts database dates to julian day format.
    ---------------------------------------------------------------------------'''
    df = pd.read_excel(database_filename, skiprows=0, header=0)
    for event_idx in range(len(df)):
        event = df.loc[event_idx]
        julian_day, year = YMD_to_JD('%Y/%m/%d', event['Event Date/Time'].split(' ')[0])
        #-----------------------------------------------------------------------------------------------------------------------#
        # Formatting julian day for saving waveforms
        if julian_day < 10: julian_day = '00' + str(julian_day)
        elif 10 <= julian_day < 100: julian_day = '0' + str(julian_day)
        else: julian_day = str(julian_day)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Converting to JD    
        df.loc[event_idx,'Event Date/Time'] = str(year) + '/' +str(julian_day) + ' ' + event['Event Date/Time'].split(' ')[1]
    #-----------------------------------------------------------------------------------------------------------------------#
    return df

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def construct_database(dates, times, lats, lons, mags, celerity_range=[0.225, 0.400], database_type='explosions'):
    '''---------------------------------------------------------------------------
    Constructs databases with relevant params needed for processing surface explosion 
    and earthquake seismoacoustic data within the Korean peninsula.
    ---------------------------------------------------------------------------'''
    # Creating Korea seismic velocity model
    taup_create.build_taup_model('kigam.tvel')
    model = TauPyModel(model='kigam')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Storing info needed for subsequent processing
    ev_datetime = []; ev_lat = []; ev_lon = []; ev_mag = []
    BRD_dist = []; CHN_dist = []; KSG_dist = []
    BRD_baz = []; CHN_baz = []; KSG_baz = []
    BRD_seismic_starttime = []; BRD_seismic_endtime = []
    CHN_seismic_starttime = []; CHN_seismic_endtime = []
    KSG_seismic_starttime = []; KSG_seismic_endtime = []
    BRD_infra_starttime = []; BRD_infra_endtime = []
    CHN_infra_starttime = []; CHN_infra_endtime = []
    KSG_infra_starttime = []; KSG_infra_endtime = []
    #-----------------------------------------------------------------------------------------------------------------------#
    # Calculations
    for ev_idx in range(len(dates)):
        # Origin time and location
        ev_datetime.append(dates[ev_idx] + ' ' + times[ev_idx])
        ev_lat_idx = lats[ev_idx]; ev_lat.append(ev_lat_idx)
        ev_lon_idx = lons[ev_idx]; ev_lon.append(ev_lon_idx)
        ev_mag_idx = mags[ev_idx]; ev_mag.append(ev_mag_idx)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Distance and back azimuth
        _, BRD_baz_idx, BRD_dist_idx = g.inv(ev_lon_idx, ev_lat_idx, BRD_coords[1], BRD_coords[0])
        if BRD_baz_idx < 0:
            BRD_baz_idx += 360
        BRD_dist.append(BRD_dist_idx/1000); BRD_baz.append(BRD_baz_idx)
        _, CHN_baz_idx, CHN_dist_idx = g.inv(ev_lon_idx, ev_lat_idx, CHN_coords[1], CHN_coords[0])
        if CHN_baz_idx < 0:
            CHN_baz_idx += 360
        CHN_dist.append(CHN_dist_idx/1000); CHN_baz.append(CHN_baz_idx)
        _, KSG_baz_idx, KSG_dist_idx = g.inv(ev_lon_idx, ev_lat_idx, KSG_coords[1], KSG_coords[0])
        if KSG_baz_idx < 0:
            KSG_baz_idx += 360
        KSG_dist.append(KSG_dist_idx/1000); KSG_baz.append(KSG_baz_idx)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Seismic time windows
        dists = [BRD_dist_idx/1000, CHN_dist_idx/1000, KSG_dist_idx/1000]
        P_arrivals = []
        for dist in dists:
            P_arrivals_array = model.get_ray_paths(source_depth_in_km=0, distance_in_degree=dist/111.1, phase_list=["P"])
            P_arrivals.append(P_arrivals_array[0].time)
        # Making starttime 10 seconds prior to onset and endtime 130 seconds after starttime
        BRD_seismic_starttime.append(P_arrivals[0] - 10); BRD_seismic_endtime.append((P_arrivals[0] - 10) + 130)
        CHN_seismic_starttime.append(P_arrivals[1] - 10); CHN_seismic_endtime.append((P_arrivals[1] - 10) + 130)
        KSG_seismic_starttime.append(P_arrivals[2] - 10); KSG_seismic_endtime.append((P_arrivals[2] - 10) + 130)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Infrasound time windows
        BRD_infra_starttime.append(dists[0]/celerity_range[1]); BRD_infra_endtime.append(dists[0]/celerity_range[0])
        CHN_infra_starttime.append(dists[1]/celerity_range[1]); CHN_infra_endtime.append(dists[1]/celerity_range[0])
        KSG_infra_starttime.append(dists[2]/celerity_range[1]); KSG_infra_endtime.append(dists[2]/celerity_range[0])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Construct dataframe
    data = np.hstack((np.array(ev_datetime).reshape(len(ev_datetime),1), np.array(ev_lat).reshape(len(ev_lat),1), np.array(ev_lon).reshape(len(ev_lon),1), np.array(ev_mag).reshape(len(ev_mag),1), 
                      np.array(BRD_dist).reshape(len(BRD_dist),1), np.array(CHN_dist).reshape(len(CHN_dist),1), np.array(KSG_dist).reshape(len(KSG_dist),1), 
                      np.array(BRD_baz).reshape(len(BRD_baz),1), np.array(CHN_baz).reshape(len(CHN_baz),1), np.array(KSG_baz).reshape(len(KSG_baz),1),
                      np.array(BRD_seismic_starttime).reshape(len(BRD_seismic_starttime),1), np.array(BRD_seismic_endtime).reshape(len(BRD_seismic_endtime),1),
                      np.array(CHN_seismic_starttime).reshape(len(CHN_seismic_starttime),1), np.array(CHN_seismic_endtime).reshape(len(CHN_seismic_endtime),1),
                      np.array(KSG_seismic_starttime).reshape(len(KSG_seismic_starttime),1), np.array(KSG_seismic_endtime).reshape(len(KSG_seismic_endtime),1),
                      np.array(BRD_infra_starttime).reshape(len(BRD_infra_starttime),1), np.array(BRD_infra_endtime).reshape(len(BRD_infra_endtime),1),
                      np.array(CHN_infra_starttime).reshape(len(CHN_infra_starttime),1), np.array(CHN_infra_endtime).reshape(len(CHN_infra_endtime),1),
                      np.array(KSG_infra_starttime).reshape(len(KSG_infra_starttime),1), np.array(KSG_infra_endtime).reshape(len(KSG_infra_endtime),1)))
    df_new = pd.DataFrame(data=data, columns=['Event Date/Time', 'Event Latitude', 'Event Longitude', 'Event Magnitude (ML)',
                                              'BRD Distance (km)', 'CHN Distance (km)', 'KSG Distance (km)', 'BRD Baz', 'CHN Baz', 'KSG Baz',
                                              'BRD Seismic Starttime (s)', 'BRD Seismic Endtime (s)',
                                              'CHN Seismic Starttime (s)', 'CHN Seismic Endtime (s)',
                                              'KSG Seismic Starttime (s)', 'KSG Seismic Endtime (s)',
                                              'BRD Infrasound Starttime (s)', 'BRD Infrasound Endtime (s)',
                                              'CHN Infrasound Starttime (s)', 'CHN Infrasound Endtime (s)',
                                              'KSG Infrasound Starttime (s)', 'KSG Infrasound Endtime (s)'])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Rounding
    for col in df_new:
        if col == 'Event Date/Time':
            continue
        else:
            df_new[col] = np.round(df_new[col].astype(float),2)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Removing events with mags < 0
    df_new.drop(df_new[df_new['Event Magnitude (ML)'] < 0].index, inplace = True)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Sorting by date (earliest to latest)
    df_new = df_new.sort_values(by=['Event Date/Time'])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Saving as Excel
    if database_type == 'explosions':
        df_new.to_excel('Surface_Explosions_Database.xlsx', index=False)
    elif database_type == 'earthquakes':
        df_new.to_excel('Earthquakes_Database.xlsx', index=False)      
    else:
        df_new.to_excel(database_type+'.xlsx', index=False)  
    print('Done saving database.')

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def fix_lengths(t, y):
    '''---------------------------------------------------------------------------
    Fixes uneven data lengths to facilitate plotting of array processing results.
    ---------------------------------------------------------------------------'''
    t_out = t.copy()
    y_out = y.copy()
    if len(t_out) > len(y_out):
        t_out = t_out[0:len(y_out)]
    elif len(y_out) > len(t_out):
        y_out = y_out[0:len(y_out)]
    #-----------------------------------------------------------------------------------------------------------------------#
    return t_out, y_out

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def data_time_window(t, st, t_start, t_end):
    '''---------------------------------------------------------------------------
    Extracts data within set time window.
    ---------------------------------------------------------------------------'''
    ix = np.where((t_start <= t) & (t < t_end))
    data = []
    if len(np.array(st).shape) > 1:
        for i in range(len(st)):
            data.append(st[i].data[ix])
    else:
        data.append(st.data[ix])
    t_data = t[ix] - np.min(t[ix])
    data = np.array(data)
    #-----------------------------------------------------------------------------------------------------------------------#
    return t_data, data

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def bilinear_resize(original_img, new_h, new_w):
    '''---------------------------------------------------------------------------
    Resize 2D array using bilinear interpolation.

    Ref: https://meghal-darji.medium.com/implementing-bilinear-interpolation-for-image-resizing-357cbb2c2722
    ---------------------------------------------------------------------------'''
    #get dimensions of original image
    old_h, old_w = original_img.shape
    # Creating array of desired shape. 
    #We will fill-in the values later.
    resized = np.zeros((new_h, new_w))
    #Calculate horizontal and vertical scaling factor
    w_scale_factor = (old_w ) / (new_w ) if new_h != 0 else 0
    h_scale_factor = (old_h ) / (new_h ) if new_w != 0 else 0
    for i in range(new_h):
        for j in range(new_w):
            #map the coordinates back to the original image
            x = i * h_scale_factor
            y = j * w_scale_factor
            #calculate the coordinate values for 4 surrounding pixels.
            x_floor = math.floor(x)
            x_ceil = min( old_h - 1, math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(old_w - 1, math.ceil(y))
            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = original_img[int(x), int(y)]
            elif (x_ceil == x_floor):
                q1 = original_img[int(x), int(y_floor)]
                q2 = original_img[int(x), int(y_ceil)]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif (y_ceil == y_floor):
                q1 = original_img[int(x_floor), int(y)]
                q2 = original_img[int(x_ceil), int(y)]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
            else:
                v1 = original_img[x_floor, y_floor]
                v2 = original_img[x_ceil, y_floor]
                v3 = original_img[x_floor, y_ceil]
                v4 = original_img[x_ceil, y_ceil]

                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

            resized[i,j] = q
    return resized.astype(float)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def check_detections(detections_filename, event_type, arrays=['BRD','CHN','KSG']):
    '''---------------------------------------------------------------------------
    Checks for inconsistencies between detections spreadsheet and database.
    ---------------------------------------------------------------------------'''
    # Read in detections file
    df = pd.read_excel(detections_filename, skiprows=0, header=0)
    for array in arrays:
        event_labels = []
        directory = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Seismic/Waveforms/'+array
        for filename in os.listdir(directory):
            year = filename.split('_')[0]
            day = filename.split('_')[1]
            time = filename.split('_')[2].replace('-',':').split('.')[0]
            event_labels.append(year + '-' + day + ' ' + time)
        for idx in range(len(df[array])):
            if df[array][idx] in event_labels: 
                continue
            elif str(df[array][idx]) == 'nan':
                continue
            else:
                print(str(array) + ' ' + str(df[array][idx]) + ' does not match')
    database_size = len(np.concatenate((df['BRD'].dropna(axis=0), df['CHN'].dropna(axis=0), df['KSG'])))
    print(str(database_size) + ' ' + event_type + ' array detections in database.')

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_event_waveforms(detections_filename, database_filenames, event_type, freqmin=1, freqmax=10, arrays=['BRD','CHN','KSG'], channels=['BH1','BH2','BHZ'], channels_alt=['BHN','BHE','BHZ'], verbose_error=False):
    '''---------------------------------------------------------------------------
    Extract waveform data and store as array with dimensions:
    [n_events, n_stns, n_pts, n_channels].

    detections_filename: stores days which detections were made at each array
    database_filename: stores metadata for each event

    Filter waveforms to desired bandpass.
    ---------------------------------------------------------------------------'''
    df = pd.read_excel(detections_filename, skiprows=0, header=0)
    if event_type == 'Earthquake':
        df_metadatas = [pd.read_excel(database_filenames[0], skiprows=0, header=0), pd.read_excel(database_filenames[1], skiprows=0, header=0)]
    elif event_type == 'Explosion':
        df_metadata = pd.read_excel(database_filenames[0], skiprows=0, header=0)
    directories = ['/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Seismic/Waveforms/', '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Seismic/Han_2023/Waveforms/']
    data = []; event_metadata = []; stn_labels = []
    #-----------------------------------------------------------------------------------------------------------------------#
    for array in arrays:
        if event_type == 'Earthquake':
            for dir_idx in range(len(directories)):
                directory = directories[dir_idx] 
                directory += array
                df_metadata = df_metadatas[dir_idx]
                for idx in range(len(df[array].dropna(axis=0))):
                    YYYY_DD = df[array][idx].split(' ')[0].replace('-','_')
                    HH_MM_SS = df[array][idx].split(' ')[1].replace(':','-')
                    wavf = directory+'/'+YYYY_DD+'_'+HH_MM_SS+'.mseed'
                    #-----------------------------------------------------------------------------------------------------------------------#
                    # Read and filter waveforms
                    try:
                        st = read(wavf)
                        try:
                            st = st.merge()
                        except:
                            st = st.resample(int(st[0].stats.sampling_rate))
                            st = st.merge()
                    except:
                        continue
                    st_filt = st.copy()
                    st_filt.taper(type='cosine', max_percentage=0.05, max_length=60)
                    try:
                        st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
                    except:
                        # Need to account for possible masked traces
                        st_filt = st_filt.split()
                        st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
                        st_filt.merge()
                    st_tmp = st_filt.copy()
                    if len(st_tmp) < 3*3: continue # if fewer than 3 3-component seismometers then skip
                    #-----------------------------------------------------------------------------------------------------------------------#
                    # Storing each channel at a time
                    data_tmp = np.zeros((3, 5201, 3)) # num stns x num data points x num channels for waveforms
                    stn_idx = random.sample(range(int(len(st_tmp)/3)), 3) # generate list of three random numbers without duplicates to be used to choose stations
                    stn_labels_tmp = []; skip_event = 'no'
                    #-----------------------------------------------------------------------------------------------------------------------#
                    if (array == 'CHN') or (array == 'KSG'):
                        for i, channel in enumerate(channels):
                            st_chn = st_tmp.copy()
                            st_chn = st_chn.select(station=array+'*', channel='*'+channel)
                            data_tmp_chn = np.zeros((3, 5201)) # waveforms
                            try:
                                for j, k in zip(stn_idx, range(data_tmp.shape[0])):
                                    data_tmp_chn[k,:] = st_chn[j].data.copy()
                                data_tmp[:,:,i] = data_tmp_chn
                                if i == 2:
                                    for jj in stn_idx:
                                        stn_labels_tmp.append(st_chn[jj].stats.station)
                            except Exception as inst:
                                if verbose_error == True:
                                    print(array + ' ' + str(inst))
                                skip_event = 'yes'
                                continue
                    #-----------------------------------------------------------------------------------------------------------------------#
                    elif array == 'BRD':
                        for i, channel in enumerate(channels_alt):
                            st_chn = st_tmp.copy()
                            st_chn = st_chn.select(station=array+'*', channel='*'+channel)
                            data_tmp_chn = np.zeros((3, 5201))
                            try:
                                for j, k in zip(stn_idx, range(data_tmp.shape[0])):
                                    data_tmp_chn[k,:] = st_chn[j].data.copy()
                                data_tmp[:,:,i] = data_tmp_chn        
                                if i == 2:
                                    for jj in stn_idx:
                                        stn_labels_tmp.append(st_chn[jj].stats.station)
                            except Exception as inst:
                                if verbose_error == True:
                                    print(array + ' ' + str(inst))
                                skip_event = 'yes'
                                continue
                    #-----------------------------------------------------------------------------------------------------------------------#
                    if skip_event == 'yes': 
                        continue
                    else:
                        # Storing data
                        data.append(data_tmp)
                    #-----------------------------------------------------------------------------------------------------------------------#
                    # Storing event metadata info (array, lat, lon, mag, distance)
                    year = int(YYYY_DD.split('_')[0]); days = int(YYYY_DD.split('_')[1])
                    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)
                    if dt.day < 10: day = '0' + str(dt.day)
                    else: day = str(dt.day)
                    if dt.month < 10: month = '0' + str(dt.month)
                    else: month = str(dt.month)
                    YYYY_MM_DD = str(dt.year) + '/' + month + '/' + day
                    ev_time = df[array][idx].split(' ')[1] # loads as HH:MM:SS
                    try:
                        ev_time_idx = np.where((df_metadata['Event Date/Time'] == YYYY_MM_DD + ' ' + ev_time))[0][0]
                    except:
                        continue
                    ev = df_metadata.loc[ev_time_idx]; julian_day, year = KGN.YMD_to_JD('%Y/%m/%d', ev['Event Date/Time'].split(' ')[0])
                    # Formatting julian day for saving waveforms
                    if julian_day < 10: julian_day = '00' + str(julian_day)
                    elif 10 <= julian_day < 100: julian_day = '0' + str(julian_day)
                    else: julian_day = str(julian_day)
                    event_metadata_i = [event_type, array, str(year)+'/'+str(julian_day)+' '+ev['Event Date/Time'].split(' ')[1], ev['Event Latitude'].astype(float), ev['Event Longitude'].astype(float), 
                                        ev['Event Magnitude (ML)'].astype(float), ev[array+' Distance (km)'].astype(float)]
                    event_metadata.append(event_metadata_i); stn_labels.append(stn_labels_tmp)
        elif event_type == 'Explosion':
            directory = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Seismic/Waveforms/'+array
            for idx in range(len(df[array].dropna(axis=0))):
                YYYY_DD = df[array][idx].split(' ')[0].replace('-','_')
                HH_MM_SS = df[array][idx].split(' ')[1].replace(':','-')
                wavf = directory+'/'+YYYY_DD+'_'+HH_MM_SS+'.mseed'
                #-----------------------------------------------------------------------------------------------------------------------#
                # Read and filter waveforms
                try:
                    st = read(wavf)
                    try:
                        st = st.merge()
                    except:
                        st = st.resample(int(st[0].stats.sampling_rate))
                        st = st.merge()
                except:
                    continue
                st_filt = st.copy()
                st_filt.taper(type='cosine', max_percentage=0.05, max_length=60)
                try:
                    st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
                except:
                    # Need to account for possible masked traces
                    st_filt = st_filt.split()
                    st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
                    st_filt.merge()
                st_tmp = st_filt.copy()
                if len(st_tmp) < 3*3: continue # if fewer than 3 3-component seismometers then skip
                #-----------------------------------------------------------------------------------------------------------------------#
                # Storing each channel at a time
                data_tmp = np.zeros((3, 5201, 3)) # num stns x num data points x num channels for waveforms
                stn_idx = random.sample(range(int(len(st_tmp)/3)), 3) # generate list of 4 random numbers without duplicates to be used to choose stations
                stn_labels_tmp = []; skip_event = 'no'
                #-----------------------------------------------------------------------------------------------------------------------#
                if (array == 'CHN') or (array == 'KSG'):
                    for i, channel in enumerate(channels):
                        st_chn = st_tmp.copy()
                        st_chn = st_chn.select(station=array+'*', channel='*'+channel)
                        data_tmp_chn = np.zeros((3, 5201)) # waveforms
                        try:
                            for j, k in zip(stn_idx, range(data_tmp.shape[0])):
                                data_tmp_chn[k,:] = st_chn[j].data.copy()
                            data_tmp[:,:,i] = data_tmp_chn
                            if i == 2:
                                for jj in stn_idx:
                                    stn_labels_tmp.append(st_chn[jj].stats.station)
                        except Exception as inst:
                            if verbose_error == True:
                                print(array + ' ' + str(inst))
                            skip_event = 'yes'
                            continue
                #-----------------------------------------------------------------------------------------------------------------------#
                elif array == 'BRD':
                    for i, channel in enumerate(channels_alt):
                        st_chn = st_tmp.copy()
                        st_chn = st_chn.select(station=array+'*', channel='*'+channel)
                        data_tmp_chn = np.zeros((3, 5201))
                        try:
                            for j, k in zip(stn_idx, range(data_tmp.shape[0])):
                                data_tmp_chn[k,:] = st_chn[j].data.copy()
                            data_tmp[:,:,i] = data_tmp_chn     
                            if i == 2:
                                for jj in stn_idx:
                                    stn_labels_tmp.append(st_chn[jj].stats.station)
                        except Exception as inst:
                            if verbose_error == True:
                                print(array + ' ' + str(inst))
                            skip_event = 'yes'
                            continue
                #-----------------------------------------------------------------------------------------------------------------------#
                if skip_event == 'yes': 
                    continue
                else:
                    # Storing data
                    data.append(data_tmp)
                    # Storing event metadata info (array, lat, lon, mag, distance)
                    year = int(YYYY_DD.split('_')[0]); days = int(YYYY_DD.split('_')[1])
                    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)
                    if dt.day < 10: day = '0' + str(dt.day)
                    else: day = str(dt.day)
                    if dt.month < 10: month = '0' + str(dt.month)
                    else: month = str(dt.month)
                    YYYY_MM_DD = str(dt.year) + '/' + month + '/' + day
                    ev_time = df[array][idx].split(' ')[1] # loads as HH:MM:SS
                    try:
                        ev_time_idx = np.where((df_metadata['Event Date/Time'] == YYYY_MM_DD + ' ' + ev_time))[0][0]
                    except:
                        continue
                    ev = df_metadata.loc[ev_time_idx]; julian_day, year = KGN.YMD_to_JD('%Y/%m/%d', ev['Event Date/Time'].split(' ')[0])
                    # Formatting julian day for saving waveforms
                    if julian_day < 10: julian_day = '00' + str(julian_day)
                    elif 10 <= julian_day < 100: julian_day = '0' + str(julian_day)
                    else: julian_day = str(julian_day)
                    event_metadata_i = [event_type, array, str(year)+'/'+str(julian_day)+' '+ev['Event Date/Time'].split(' ')[1], ev['Event Latitude'].astype(float), ev['Event Longitude'].astype(float), 
                                        ev['Event Magnitude (ML)'].astype(float), ev[array+' Distance (km)'].astype(float)]
                    event_metadata.append(event_metadata_i); stn_labels.append(stn_labels_tmp)
    #-----------------------------------------------------------------------------------------------------------------------#
    print('There are ' + str(len(data)) + ' total detections with shape: ' +str(np.array(data).shape))
    #-----------------------------------------------------------------------------------------------------------------------#
    return np.array(data), event_metadata, stn_labels

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_noise_waveforms(noise_filename, database_filename, event_type, freqmin=1, freqmax=10, verbose_error=False, arrays=['BRD','CHN','KSG'], channels=['BH1','BH2','BHZ'], channels_alt=['BHN','BHE','BHZ']):
    '''---------------------------------------------------------------------------
    Extract noise data and store as array with dimensions: 
    [n_noise_segments, n_stns, n_pts, n_channels].

    noise_filename: stores days which contained signals and thus had to be removed

    Filter waveforms to desired bandpass (keep consistent at 1-10 Hz).
    ---------------------------------------------------------------------------'''
    # Read database for event magnitude and distance - need to convert dates to JD for easy indexing
    df_eq = pd.read_excel(database_filename, skiprows=0, header=0)
    for event_idx in range(len(df_eq)):
        event = df_eq.loc[event_idx]
        julian_day, year = YMD_to_JD('%Y/%m/%d', event['Event Date/Time'].split(' ')[0])
        #-----------------------------------------------------------------------------------------------------------------------#
        # Formatting julian day for saving waveforms
        if julian_day < 10: julian_day = '00' + str(julian_day)
        elif 10 <= julian_day < 100: julian_day = '0' + str(julian_day)
        else: julian_day = str(julian_day)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Converting to JD    
        df_eq.loc[event_idx,'Event Date/Time'] = str(year) + '/' +str(julian_day) + ' ' + event['Event Date/Time'].split(' ')[1]
    #-----------------------------------------------------------------------------------------------------------------------#
    df = pd.read_excel(noise_filename, skiprows=0, header=0)
    data = []; event_metadata = []; stn_labels = []
    #-----------------------------------------------------------------------------------------------------------------------#
    for array in arrays:
        directory = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Seismic/Waveforms/'+array
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if f.split('/')[-1].split('_')[0] == 'NOISE':
                year = f.split('/')[-1].split('_')[1]
                day = f.split('/')[-1].split('_')[2]
                time = f.split('/')[-1].split('_')[-1].split('.')[0].replace('-',':')
                event_label = year + '-' + day + ' ' + time
                if event_label in str(df[array].values):
                    continue
                else:
                    # Read and filter waveforms
                    try:
                        st = read(f)
                    except:
                        continue
                    st_filt = st.copy()
                    st_filt.taper(type='cosine', max_percentage=0.05, max_length=60)
                    try:
                        st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
                    except:
                        # Need to account for possible masked traces
                        st_filt = st_filt.split()
                        st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
                        st_filt.merge()
                    st_tmp = st_filt.copy()
                    if len(st_tmp) < 3*3: continue # if fewer than 3 3-component seismometers then skip
                    #-----------------------------------------------------------------------------------------------------------------------#
                    # Storing each channel at a time
                    data_tmp = np.zeros((3, 5201, 3)) # num stns x num data points x num channels for waveforms
                    stn_idx = random.sample(range(int(len(st_tmp)/3)), 3) # generate list of three random numbers without duplicates to be used to choose stations
                    stn_labels_tmp = []; skip_event = 'no'
                    #-----------------------------------------------------------------------------------------------------------------------#
                    if (array == 'CHN') or (array == 'KSG'):
                        for i, channel in enumerate(channels):
                            st_chn = st_tmp.copy()
                            st_chn = st_chn.select(station=array+'*', channel='*'+channel)
                            data_tmp_chn = np.zeros((3, 5201)) # waveforms
                            try:
                                for j, k in zip(stn_idx, range(data_tmp.shape[0])):
                                    data_tmp_chn[k,:] = st_chn[j].data.copy()
                                data_tmp[:,:,i] = data_tmp_chn
                                if i == 2:
                                    for jj in stn_idx:
                                        stn_labels_tmp.append(st_chn[jj].stats.station)
                            except Exception as inst:
                                if verbose_error == True:
                                    print(array + ' ' + str(inst))
                                skip_event = 'yes'
                                continue
                    #-----------------------------------------------------------------------------------------------------------------------#
                    elif array == 'BRD':
                        for i, channel in enumerate(channels_alt):
                            st_chn = st_tmp.copy()
                            st_chn = st_chn.select(station=array+'*', channel='*'+channel)
                            data_tmp_chn = np.zeros((3, 5201))
                            try:
                                for j, k in zip(stn_idx, range(data_tmp.shape[0])):
                                    data_tmp_chn[k,:] = st_chn[j].copy()
                                data_tmp[:,:,i] = data_tmp_chn
                                if i == 2:
                                    for jj in stn_idx:
                                        stn_labels_tmp.append(st_chn[jj].stats.station)
                            except Exception as inst:
                                if verbose_error == True:
                                    print(array + ' ' + str(inst))
                                skip_event = 'yes'
                                continue
                    #-----------------------------------------------------------------------------------------------------------------------#
                    if skip_event == 'yes': 
                        continue
                    else:
                        # Storing data
                        data.append(data_tmp)
                        # Storing metadata
                        df_eq_idx = np.where((df_eq['Event Date/Time'] == event_label.replace('-','/')))[0][0]; ev = df_eq.loc[df_eq_idx]
                        event_metadata.append(['Noise', array, ev['Event Date/Time'], ev['Event Latitude'].astype(float), ev['Event Longitude'].astype(float), ev['Event Magnitude (ML)'].astype(float), ev[array+' Distance (km)'].astype(float)])
                        stn_labels.append(stn_labels_tmp)
            else:
                continue
    #-----------------------------------------------------------------------------------------------------------------------#
    return np.array(data), event_metadata, stn_labels

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_infrasound_noise_data(eq_database_filename, exp_database_filename, extra_noise_database_filename, eq_noise_dets_filename, exp_noise_dets_filename, extra_noise_dets_filename, npts_freq=20, npts_time=360,
                              arrays=['BRD','CHN','KSG'], datadir='/Volumes/Extreme SSD/Korea_Events/'):
    
    # Read and convert database datetimes
    # Earthquakes
    df_eq = convert_database_datetimes_to_JD(eq_database_filename)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Explosions
    df_exp = convert_database_datetimes_to_JD(exp_database_filename)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Extra noise
    df_extra = convert_database_datetimes_to_JD(extra_noise_database_filename)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Reading noise detections spreadsheets
    df_eq_noise_dets = pd.read_excel(eq_noise_dets_filename, skiprows=0, header=0)
    df_exp_noise_dets = pd.read_excel(exp_noise_dets_filename, skiprows=0, header=0)
    df_extra_noise_dets = pd.read_excel(extra_noise_dets_filename, skiprows=0, header=0)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Start with earthquake noise data
    noise_data_eq_slowX = []; noise_data_eq_slowY = []; noise_data_eq_S = []; noise_data_eq_T = []; noise_metadata_eq = []
    #-----------------------------------------------------------------------------------------------------------------------#
    for array in arrays:
        # Array processing results directory
        directory_cardinal = datadir+'Earthquakes/Infrasound/Cardinal/'+array
        for idx in range(len(df_eq_noise_dets[array].dropna(axis=0))):
            YYYY_DD = df_eq_noise_dets[array][idx].split(' ')[0].replace('-','_')
            HH_MM_SS = df_eq_noise_dets[array][idx].split(' ')[1].replace(':','-')
            B = np.load(directory_cardinal+'/NOISE_B_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]
            V = np.load(directory_cardinal+'/NOISE_V_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]
            S = np.load(directory_cardinal+'/NOISE_S_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]
            T = np.load(directory_cardinal+'/NOISE_T_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]; T = T.reshape(1,len(T))
            #-----------------------------------------------------------------------------------------------------------------------#
            # Bilinear resize and store
            B = bilinear_resize(B, new_h=npts_freq, new_w=npts_time)
            V = bilinear_resize(V, new_h=npts_freq, new_w=npts_time)
            S = bilinear_resize(S, new_h=npts_freq, new_w=npts_time)
            T = bilinear_resize(T, new_h=T.shape[0], new_w=npts_time)
            #-----------------------------------------------------------------------------------------------------------------------#
            # Convert to slowness and store
            slowX = np.zeros((B.shape)); slowY = np.zeros((B.shape))
            for row_idx in range(B.shape[0]):
                for col_idx in range(B.shape[1]):
                    sl_x, sl_y = convert_to_slowness(B[row_idx, col_idx], V[row_idx, col_idx])
                    slowX[row_idx, col_idx] = sl_x; slowY[row_idx, col_idx] = sl_y
            noise_data_eq_slowX.append(slowX); noise_data_eq_slowY.append(slowY); noise_data_eq_S.append(S); noise_data_eq_T.append(T)
            #-----------------------------------------------------------------------------------------------------------------------#
            # Store metadata
            df_eq_idx = np.where((df_eq['Event Date/Time'] == YYYY_DD.replace('_','/')+' '+HH_MM_SS.replace('-',':')))[0][0]; ev = df_eq.loc[df_eq_idx]
            noise_metadata_eq.append(['Noise', array, ev['Event Date/Time'], ev['Event Latitude'].astype(float), ev['Event Longitude'].astype(float), ev['Event Magnitude (ML)'].astype(float), ev[array+' Distance (km)'].astype(float)])
    noise_data_eq_slowX = np.array(noise_data_eq_slowX); noise_data_eq_slowY = np.array(noise_data_eq_slowY); noise_data_eq_S = np.array(noise_data_eq_S); noise_data_eq_T = np.array(noise_data_eq_T); noise_metadata_eq = np.array(noise_metadata_eq)
    noise_data_eq = np.zeros((noise_data_eq_slowX.shape[0], npts_freq, npts_time, 3)) # num segments X num freq bands X num data points X channel (slowness X, slowness Y, and semblance)
    noise_data_eq[:,:,:,0] = noise_data_eq_slowX.copy(); noise_data_eq[:,:,:,1] = noise_data_eq_slowY.copy(); noise_data_eq[:,:,:,2] = noise_data_eq_S.copy()
    #-----------------------------------------------------------------------------------------------------------------------#
    # Explosion noise data
    noise_data_exp_slowX = []; noise_data_exp_slowY = []; noise_data_exp_S = []; noise_data_exp_T = []; noise_metadata_exp = []
    #-----------------------------------------------------------------------------------------------------------------------#
    for array in arrays:
        # Array processing results directory
        directory_cardinal = datadir+'Explosions/Infrasound/Cardinal/'+array
        for idx in range(len(df_exp_noise_dets[array].dropna(axis=0))):
            YYYY_DD = df_exp_noise_dets[array][idx].split(' ')[0].replace('-','_')
            HH_MM_SS = df_exp_noise_dets[array][idx].split(' ')[1].replace(':','-')
            B = np.load(directory_cardinal+'/NOISE_B_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]
            V = np.load(directory_cardinal+'/NOISE_V_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]
            S = np.load(directory_cardinal+'/NOISE_S_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]
            T = np.load(directory_cardinal+'/NOISE_T_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]; T = T.reshape(1,len(T))
            #-----------------------------------------------------------------------------------------------------------------------#
            # Bilinear resize and store
            B = bilinear_resize(B, new_h=npts_freq, new_w=npts_time)
            V = bilinear_resize(V, new_h=npts_freq, new_w=npts_time)
            S = bilinear_resize(S, new_h=npts_freq, new_w=npts_time)
            T = bilinear_resize(T, new_h=T.shape[0], new_w=npts_time)
            #-----------------------------------------------------------------------------------------------------------------------#
            # Convert to slowness and store
            slowX = np.zeros((B.shape)); slowY = np.zeros((B.shape))
            for row_idx in range(B.shape[0]):
                for col_idx in range(B.shape[1]):
                    sl_x, sl_y = convert_to_slowness(B[row_idx, col_idx], V[row_idx, col_idx])
                    slowX[row_idx, col_idx] = sl_x; slowY[row_idx, col_idx] = sl_y
            noise_data_exp_slowX.append(slowX); noise_data_exp_slowY.append(slowY); noise_data_exp_S.append(S); noise_data_exp_T.append(T)
            #-----------------------------------------------------------------------------------------------------------------------#
            # Store metadata
            df_exp_idx = np.where((df_exp['Event Date/Time'] == YYYY_DD.replace('_','/')+' '+HH_MM_SS.replace('-',':')))[0][0]; ev = df_exp.loc[df_exp_idx]
            noise_metadata_exp.append(['Noise', array, ev['Event Date/Time'], ev['Event Latitude'].astype(float), ev['Event Longitude'].astype(float), ev['Event Magnitude (ML)'].astype(float), ev[array+' Distance (km)'].astype(float)])
    noise_data_exp_slowX = np.array(noise_data_exp_slowX); noise_data_exp_slowY = np.array(noise_data_exp_slowY); noise_data_exp_S = np.array(noise_data_exp_S); noise_data_exp_T = np.array(noise_data_exp_T); noise_metadata_exp = np.array(noise_metadata_exp)
    noise_data_exp = np.zeros((noise_data_exp_slowX.shape[0], npts_freq, npts_time, 3)) # num segments X num freq bands X num data points X channel (slowness X, slowness Y, and semblance)
    noise_data_exp[:,:,:,0] = noise_data_exp_slowX.copy(); noise_data_exp[:,:,:,1] = noise_data_exp_slowY.copy(); noise_data_exp[:,:,:,2] = noise_data_exp_S.copy()
    #-----------------------------------------------------------------------------------------------------------------------#
    # Exrta noise data
    noise_data_extra_slowX = []; noise_data_extra_slowY = []; noise_data_extra_S = []; noise_data_extra_T = []; noise_metadata_extra = []
    #-----------------------------------------------------------------------------------------------------------------------#
    for array in arrays:
        # Array processing results directory
        directory_cardinal = datadir+'Extra_Noise/Infrasound/Cardinal/'+array
        for idx in range(len(df_extra_noise_dets[array].dropna(axis=0))):
            YYYY_DD = df_extra_noise_dets[array][idx].split(' ')[0].replace('-','_')
            HH_MM_SS = df_extra_noise_dets[array][idx].split(' ')[1].replace(':','-')
            B = np.load(directory_cardinal+'/B_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]
            V = np.load(directory_cardinal+'/V_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]
            S = np.load(directory_cardinal+'/S_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]
            T = np.load(directory_cardinal+'/T_'+YYYY_DD+'_'+HH_MM_SS+'.npy')[0]; T = T.reshape(1,len(T))
            #-----------------------------------------------------------------------------------------------------------------------#
            # Bilinear resize and store
            B = bilinear_resize(B, new_h=npts_freq, new_w=npts_time)
            V = bilinear_resize(V, new_h=npts_freq, new_w=npts_time)
            S = bilinear_resize(S, new_h=npts_freq, new_w=npts_time)
            T = bilinear_resize(T, new_h=T.shape[0], new_w=npts_time)
            #-----------------------------------------------------------------------------------------------------------------------#
            # Convert to slowness and store
            slowX = np.zeros((B.shape)); slowY = np.zeros((B.shape))
            for row_idx in range(B.shape[0]):
                for col_idx in range(B.shape[1]):
                    sl_x, sl_y = convert_to_slowness(B[row_idx, col_idx], V[row_idx, col_idx])
                    slowX[row_idx, col_idx] = sl_x; slowY[row_idx, col_idx] = sl_y
            noise_data_extra_slowX.append(slowX); noise_data_extra_slowY.append(slowY); noise_data_extra_S.append(S); noise_data_extra_T.append(T)
            #-----------------------------------------------------------------------------------------------------------------------#
            # Store metadata
            df_extra_idx = np.where((df_extra['Event Date/Time'] == YYYY_DD.replace('_','/')+' '+HH_MM_SS.replace('-',':')))[0][0]; ev = df_extra.loc[df_extra_idx]
            noise_metadata_extra.append(['Noise', array, ev['Event Date/Time'], ev['Event Latitude'].astype(float), ev['Event Longitude'].astype(float), ev['Event Magnitude (ML)'].astype(float), ev[array+' Distance (km)'].astype(float)])
    noise_data_extra_slowX = np.array(noise_data_extra_slowX); noise_data_extra_slowY = np.array(noise_data_extra_slowY); noise_data_extra_S = np.array(noise_data_extra_S); noise_data_extra_T = np.array(noise_data_extra_T); noise_metadata_extra = np.array(noise_metadata_extra)
    noise_data_extra = np.zeros((noise_data_extra_slowX.shape[0], npts_freq, npts_time, 3)) # num segments X num freq bands X num data points X channel (slowness X, slowness Y, and semblance)
    noise_data_extra[:,:,:,0] = noise_data_extra_slowX.copy(); noise_data_extra[:,:,:,1] = noise_data_extra_slowY.copy(); noise_data_extra[:,:,:,2] = noise_data_extra_S.copy()
    #-----------------------------------------------------------------------------------------------------------------------#
    # Concatenate all noise segments
    noise_data = np.concatenate((noise_data_eq, noise_data_exp, noise_data_extra)); noise_data_T = np.concatenate((noise_data_eq_T, noise_data_exp_T, noise_data_extra_T)); noise_metadata = np.concatenate((noise_metadata_eq, noise_metadata_exp, noise_metadata_extra))
    #-----------------------------------------------------------------------------------------------------------------------#
    return noise_data, noise_data_T, noise_metadata # we want to store timestamps because they are needed for shifting start times during augmentation

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_TrainTest_infrasound_data(ev_train, ev_test, eq_database_filename, exp_database_filename, extra_noise_database_filename=None, eq_noise_dets_filename=None, exp_noise_dets_filename=None, extra_noise_dets_filename=None, 
                                  resize=False, npts_freq=10, npts_time=200, local_infrasound=False, make_noise=False, azimuth_deviation=False):
    
    # Get infrasound noise segments (manually selected)
    if make_noise:
        print('Downloading infrasound noise data')
        infra_noise_data, infra_noise_data_T, infra_noise_metadata = get_infrasound_noise_data(eq_database_filename, exp_database_filename, extra_noise_database_filename, eq_noise_dets_filename, exp_noise_dets_filename, extra_noise_dets_filename, 
                                                                                               npts_freq=npts_freq, npts_time=npts_time)
        random_infra_noise_segments_idx = random.sample(range(infra_noise_data.shape[0]), infra_noise_data.shape[0])
        infra_noise_data = infra_noise_data[random_infra_noise_segments_idx]; infra_noise_data_T = infra_noise_data_T[random_infra_noise_segments_idx]; infra_noise_metadata = infra_noise_metadata[random_infra_noise_segments_idx]
        print('Done')
        #-----------------------------------------------------------------------------------------------------------------------#
        # Getting infrasound array noise segments ready 
        BRD_infra_noise_idx = np.where((infra_noise_metadata[:,1] == 'BRD'))[0]; BRD_infra_noise_data = infra_noise_data[BRD_infra_noise_idx]; BRD_infra_noise_data_T = infra_noise_data_T[BRD_infra_noise_idx]
        BRD_infra_noise_metadata = infra_noise_metadata[BRD_infra_noise_idx]; BRD_infra_noise_idx = 0
        CHN_infra_noise_idx = np.where((infra_noise_metadata[:,1] == 'CHN'))[0]; CHN_infra_noise_data = infra_noise_data[CHN_infra_noise_idx]; CHN_infra_noise_data_T = infra_noise_data_T[CHN_infra_noise_idx]
        CHN_infra_noise_metadata = infra_noise_metadata[CHN_infra_noise_idx]; CHN_infra_noise_idx = 0
        KSG_infra_noise_idx = np.where((infra_noise_metadata[:,1] == 'KSG'))[0]; KSG_infra_noise_data = infra_noise_data[KSG_infra_noise_idx]; KSG_infra_noise_data_T = infra_noise_data_T[KSG_infra_noise_idx]
        KSG_infra_noise_metadata = infra_noise_metadata[KSG_infra_noise_idx]; KSG_infra_noise_idx = 0
    #-----------------------------------------------------------------------------------------------------------------------#
    print('Constructing infrasound training dataset')
    # Start with train
    if resize:
        X_train_infra = np.zeros((ev_train.shape[0], npts_freq, npts_time, 3)); X_train_infra_T = np.zeros((ev_train.shape[0], npts_time))
    else:
        X_train_infra = []; X_train_infra_T = []
    for idx in range(ev_train.shape[0]):
        # Event metadata
        event_type=ev_train[idx][0]; array=ev_train[idx][1]; lat = ev_train[idx][3]; lon = ev_train[idx][4]
        #-----------------------------------------------------------------------------------------------------------------------#
        if (event_type == 'Earthquake') or (event_type == 'Explosion'):
            # Loading array processing results
            if local_infrasound:
                directory_cardinal = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Local_Infrasound/Cardinal/'+array
            else:
                directory_cardinal = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Infrasound/Cardinal/'+array
            try:
                B = np.load(directory_cardinal+'/B_'+ev_train[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                V = np.load(directory_cardinal+'/V_'+ev_train[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                S = np.load(directory_cardinal+'/S_'+ev_train[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                T = np.load(directory_cardinal+'/T_'+ev_train[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]; T = T.reshape(1,len(T))
            except:
                directory_cardinal = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Infrasound/Han_2023/Cardinal/'+array
                B = np.load(directory_cardinal+'/B_'+ev_train[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                V = np.load(directory_cardinal+'/V_'+ev_train[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                S = np.load(directory_cardinal+'/S_'+ev_train[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                T = np.load(directory_cardinal+'/T_'+ev_train[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]; T = T.reshape(1,len(T))
            #-----------------------------------------------------------------------------------------------------------------------#
            if azimuth_deviation:
                # Measure GT back azimuth
                if array == 'BRD':
                    _, GT_baz, _ = g.inv(lon, lat, BRD_coords[1], BRD_coords[0])
                if array == 'CHN':
                    _, GT_baz, _ = g.inv(lon, lat, CHN_coords[1], CHN_coords[0])
                if array == 'KSG':
                    _, GT_baz, _ = g.inv(lon, lat, KSG_coords[1], KSG_coords[0])
                # Calculate azimuth deviation
                B -= GT_baz
                for B_row in range(len(B[:,0])):
                    for B_col in range(len(B[0,:])):
                        # Mitigate for angular wrapping
                        if B[B_row, B_col] < -270: B[B_row, B_col] += 360
                        elif B[B_row, B_col] > 270: B[B_row, B_col] -= 360
                        else: pass
            #-----------------------------------------------------------------------------------------------------------------------#
            if resize:
                # Bilinear resize
                B = bilinear_resize(B, new_h=npts_freq, new_w=npts_time)
                V = bilinear_resize(V, new_h=npts_freq, new_w=npts_time)
                S = bilinear_resize(S, new_h=npts_freq, new_w=npts_time)
                T = bilinear_resize(T, new_h=T.shape[0], new_w=npts_time)
            #-----------------------------------------------------------------------------------------------------------------------#
            # Convert to slowness and store
            slowX = np.zeros((B.shape)); slowY = np.zeros((B.shape))
            for row_idx in range(B.shape[0]):
                for col_idx in range(B.shape[1]):
                    sl_x, sl_y = convert_to_slowness(B[row_idx, col_idx], V[row_idx, col_idx])
                    slowX[row_idx, col_idx] = sl_x; slowY[row_idx, col_idx] = sl_y
            #-----------------------------------------------------------------------------------------------------------------------#
            if resize:
                X_train_infra[idx,:,:,0] = slowX; X_train_infra[idx,:,:,1] = slowY; X_train_infra[idx,:,:,2] = S; X_train_infra_T[idx,:] = T
            else:
                X_train_infra.append([slowX, slowY, S]);
                X_train_infra_T.append(T)
        elif event_type == 'Noise':
            # Here we retrieve noise data of corresponding array
            if array == 'BRD':
                X_train_infra[idx,:,:,:] = BRD_infra_noise_data[BRD_infra_noise_idx,:,:,:]; X_train_infra_T[idx,:] = BRD_infra_noise_data_T[BRD_infra_noise_idx,:]; ev_train_infra.append(BRD_infra_noise_metadata[BRD_infra_noise_idx]); BRD_infra_noise_idx += 1
            elif array == 'CHN':
                X_train_infra[idx,:,:,:] = CHN_infra_noise_data[CHN_infra_noise_idx,:,:,:]; X_train_infra_T[idx,:] = CHN_infra_noise_data_T[CHN_infra_noise_idx,:]; ev_train_infra.append(CHN_infra_noise_metadata[CHN_infra_noise_idx]); CHN_infra_noise_idx += 1
            elif array == 'KSG':
                X_train_infra[idx,:,:,:] = KSG_infra_noise_data[KSG_infra_noise_idx,:,:,:]; X_train_infra_T[idx,:] = KSG_infra_noise_data_T[KSG_infra_noise_idx,:]; ev_train_infra.append(KSG_infra_noise_metadata[KSG_infra_noise_idx]); KSG_infra_noise_idx += 1
    print('Done')
    #-----------------------------------------------------------------------------------------------------------------------#
    print('Constructing infrasound test dataset')
    # Now do test
    if resize:
        X_test_infra = np.zeros((ev_test.shape[0], npts_freq, npts_time, 3)); X_test_infra_T = np.zeros((ev_test.shape[0], npts_time))
    else:
        X_test_infra = []; X_test_infra_T = []
    for idx in range(ev_test.shape[0]):
        # Event metadata
        event_type=ev_test[idx][0]; array=ev_test[idx][1]; lat = ev_test[idx][3]; lon = ev_test[idx][4]
        #-----------------------------------------------------------------------------------------------------------------------#
        if (event_type == 'Earthquake') or (event_type == 'Explosion'):
            # Loading array processing results
            if local_infrasound:
                directory_cardinal = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Local_Infrasound/Cardinal/'+array
            else:
                directory_cardinal = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Infrasound/Cardinal/'+array
            try:
                B = np.load(directory_cardinal+'/B_'+ev_test[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                V = np.load(directory_cardinal+'/V_'+ev_test[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                S = np.load(directory_cardinal+'/S_'+ev_test[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                T = np.load(directory_cardinal+'/T_'+ev_test[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]; T = T.reshape(1,len(T))
            except:
                directory_cardinal = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Infrasound/Han_2023/Cardinal/'+array
                B = np.load(directory_cardinal+'/B_'+ev_test[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                V = np.load(directory_cardinal+'/V_'+ev_test[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                S = np.load(directory_cardinal+'/S_'+ev_test[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
                T = np.load(directory_cardinal+'/T_'+ev_test[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]; T = T.reshape(1,len(T))
            #-----------------------------------------------------------------------------------------------------------------------#
            if azimuth_deviation:
                # Measure GT back azimuth
                if array == 'BRD':
                    _, GT_baz, _ = g.inv(lon, lat, BRD_coords[1], BRD_coords[0])
                if array == 'CHN':
                    _, GT_baz, _ = g.inv(lon, lat, CHN_coords[1], CHN_coords[0])
                if array == 'KSG':
                    _, GT_baz, _ = g.inv(lon, lat, KSG_coords[1], KSG_coords[0])
                # Calculate azimuth deviation
                B -= GT_baz
                for B_row in range(len(B[:,0])):
                    for B_col in range(len(B[0,:])):
                        # Mitigate for angular wrapping
                        if B[B_row, B_col] < -270: B[B_row, B_col] += 360
                        elif B[B_row, B_col] > 270: B[B_row, B_col] -= 360
                        else: pass
            #-----------------------------------------------------------------------------------------------------------------------#
            if resize:
                # Bilinear resize
                B = bilinear_resize(B, new_h=npts_freq, new_w=npts_time)
                V = bilinear_resize(V, new_h=npts_freq, new_w=npts_time)
                S = bilinear_resize(S, new_h=npts_freq, new_w=npts_time)
                T = bilinear_resize(T, new_h=T.shape[0], new_w=npts_time)
            #-----------------------------------------------------------------------------------------------------------------------#
            # Convert to slowness and store
            slowX = np.zeros((B.shape)); slowY = np.zeros((B.shape))
            for row_idx in range(B.shape[0]):
                for col_idx in range(B.shape[1]):
                    sl_x, sl_y = convert_to_slowness(B[row_idx, col_idx], V[row_idx, col_idx])
                    slowX[row_idx, col_idx] = sl_x; slowY[row_idx, col_idx] = sl_y
            #-----------------------------------------------------------------------------------------------------------------------#
            if resize:
                X_test_infra[idx,:,:,0] = slowX; X_test_infra[idx,:,:,1] = slowY; X_test_infra[idx,:,:,2] = S; X_test_infra_T[idx,:] = T
            else:
                X_test_infra.append([slowX, slowY, S])
                X_test_infra_T.append(T)
        elif event_type == 'Noise':
            # Here we retrieve noise data of corresponding array
            if array == 'BRD':
                X_test_infra[idx,:,:,:] = BRD_infra_noise_data[BRD_infra_noise_idx,:,:,:]; X_test_infra_T[idx,:] = BRD_infra_noise_data_T[BRD_infra_noise_idx,:]; ev_test_infra.append(BRD_infra_noise_metadata[BRD_infra_noise_idx]); BRD_infra_noise_idx += 1
            elif array == 'CHN':
                X_test_infra[idx,:,:,:] = CHN_infra_noise_data[CHN_infra_noise_idx,:,:,:]; X_test_infra_T[idx,:] = CHN_infra_noise_data_T[CHN_infra_noise_idx,:]; ev_test_infra.append(CHN_infra_noise_metadata[CHN_infra_noise_idx]); CHN_infra_noise_idx += 1
            elif array == 'KSG':
                X_test_infra[idx,:,:,:] = KSG_infra_noise_data[KSG_infra_noise_idx,:,:,:]; X_test_infra_T[idx,:] = KSG_infra_noise_data_T[KSG_infra_noise_idx,:]; ev_test_infra.append(KSG_infra_noise_metadata[KSG_infra_noise_idx]); KSG_infra_noise_idx += 1
    print('Done')
    #-----------------------------------------------------------------------------------------------------------------------#
    return X_train_infra, X_train_infra_T, X_test_infra, X_test_infra_T

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def augment_seismic_training_data(X_train, X_test, y_train, samp_rate=40, win_len=90, nperseg=2**7, spec_win_overlap=0.70, spec_range=[0.9375,10], max_timeshift=5, augmentations=10, 
                                  add_random_noise=True, var_noise_percent=10, spec_shape=(90,90,3), randomly_reorder_stations=True):
    '''---------------------------------------------------------------------------
    Returns:
    X_train_wavf - waveform data augmented by randomly shifting starttimes and adding noise by channel
    X_tran_spec - spectrograms of augmented waveforms
    X_test_wavf - randomly time shifted waveform test set (not augmented)
    X_test_spec - spectrograms of test waveforms
    y_train_new - OHE labels for augmented dataset 
    ---------------------------------------------------------------------------'''
    X_train_wavf = []; X_train_spec = []; y_train_new = []
    nsamp = samp_rate*win_len
    # Augmenting training dataset
    for entry in range(X_train.shape[0]):
        starttime_shifts = random.sample(range(0,max_timeshift*10), augmentations)
        starttime_shifts = [x/10 for x in starttime_shifts] # can be from [0,0.1,0.2...max_timeshift]
        for shift in starttime_shifts:
            #-----------------------------------------------------------------------------------------------------------------------#
            if randomly_reorder_stations:
                # Randomly reordering stations
                tr_reorder_idx = random.sample(range(X_train[entry].shape[0]), 3) # generate randomization indeces for traces
                X_train_tmp = X_train[entry][tr_reorder_idx,:,:].copy()
            else:
                X_train_tmp = X_train[entry].copy()
            #-----------------------------------------------------------------------------------------------------------------------#
            # Shifting starttimes and trimming endtimes to match specified window length
            new_wavf = X_train_tmp[:,int(shift*samp_rate):int(nsamp+shift*samp_rate),:].copy()
            #-----------------------------------------------------------------------------------------------------------------------#
            # Normalizing by trace
            for chn in range(new_wavf.shape[2]):
                for stn in range(new_wavf.shape[0]):
                    new_wavf[stn,:,chn] /= np.abs(new_wavf[stn,:,chn].max())
            #-----------------------------------------------------------------------------------------------------------------------#
            if add_random_noise == True:
                # Add noise by trace
                for chn in range(new_wavf.shape[2]):
                    for stn in range(new_wavf.shape[0]):
                        noise = np.random.normal(0, np.abs(np.var(new_wavf[stn,:,chn]),dtype=np.float64)*(var_noise_percent/100), new_wavf[stn,:,chn].shape)
                        new_wavf[stn,:,chn] += noise
            #-----------------------------------------------------------------------------------------------------------------------#
            # Generating spectrograms
            spec_tmp = np.zeros(spec_shape) # need to fix this
            for chn in range(new_wavf.shape[2]):
                spec_tmp_chn = []
                for stn in range(new_wavf.shape[0]):
                    f, _, Sxx_tmp = signal.spectrogram(new_wavf[stn,:,chn], samp_rate, nperseg=nperseg, noverlap=(nperseg)*spec_win_overlap)
                    freq_slice = np.where((f >= spec_range[0]) & (f <= spec_range[1])); f = f[freq_slice]; Sxx_tmp = Sxx_tmp[freq_slice,:][0]
                    Sxx_tmp /= np.abs(Sxx_tmp.max())# normalize by trace
                    spec_tmp_chn.append(Sxx_tmp)
                spec_tmp_chn = np.vstack((np.array(spec_tmp_chn))) # stacked spectrograms for each station within channel
                if (spec_tmp_chn.shape[0] != spec_shape[0]) or (spec_tmp_chn.shape[1] != spec_shape[1]):
                    print('Computed spectrogram shape and spec shape parameter dont match...using bilinear interpolation')
                    spec_tmp_chn = bilinear_resize(spec_tmp_chn, new_h=spec_shape[0], new_w=spec_shape[1]) # resize
                spec_tmp[:,:,chn] = spec_tmp_chn
            #-----------------------------------------------------------------------------------------------------------------------#
            # Appending data and labels
            X_train_spec.append(np.nan_to_num(spec_tmp))
            X_train_wavf.append(np.nan_to_num(new_wavf))
            # Storing OHE labels
            y_train_new.append(y_train[entry])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Just normalizing and generating spectrograms out of test set here (no shift, no noise, no augmentation)
    X_test_wavf = []; X_test_spec = []
    for entry in range(X_test.shape[0]):
        # Trim to desired time length
        new_wavf = X_test[entry][:,0:int(nsamp),:].copy()
        #-----------------------------------------------------------------------------------------------------------------------#
        # Normalizing by trace
        for chn in range(new_wavf.shape[2]):
            for stn in range(new_wavf.shape[0]):
                new_wavf[stn,:,chn] /= np.abs(new_wavf[stn,:,chn].max())
        #-----------------------------------------------------------------------------------------------------------------------#
        # Generating spectrograms
        spec_tmp = np.zeros(spec_shape)
        for chn in range(new_wavf.shape[2]):
            spec_tmp_chn = []
            for stn in range(new_wavf.shape[0]):
                f, _, Sxx_tmp = signal.spectrogram(new_wavf[stn,:,chn], samp_rate, nperseg=nperseg, noverlap=(nperseg)*spec_win_overlap)
                freq_slice = np.where((f >= spec_range[0]) & (f <= spec_range[1])); f = f[freq_slice]; Sxx_tmp = Sxx_tmp[freq_slice,:][0]
                Sxx_tmp /= np.abs(Sxx_tmp.max())# normalize by trace               
                spec_tmp_chn.append(Sxx_tmp)
            spec_tmp_chn = np.vstack((np.array(spec_tmp_chn))) # stacked spectrograms for each station within channel
            if (spec_tmp_chn.shape[0] != spec_shape[0]) or (spec_tmp_chn.shape[1] != spec_shape[1]):
                print('Computed spectrogram shape and spec shape parameter dont match...using bilinear interpolation')
                spec_tmp_chn = bilinear_resize(spec_tmp_chn, new_h=spec_shape[0], new_w=spec_shape[1]) # resize
            spec_tmp[:,:,chn] = spec_tmp_chn
        #-----------------------------------------------------------------------------------------------------------------------#
        # Appending data
        X_test_spec.append(np.nan_to_num(spec_tmp))
        X_test_wavf.append(np.nan_to_num(new_wavf))
    #-----------------------------------------------------------------------------------------------------------------------#
    X_train_wavf = np.array(X_train_wavf); X_test_wavf = np.array(X_test_wavf)
    X_train_spec = np.array(X_train_spec); X_test_spec = np.array(X_test_spec)
    print('Train dataset size is: ' + str(X_train_wavf.shape[0]))
    print('Test dataset size is: ' + str(X_test_wavf.shape[0]))
    #-----------------------------------------------------------------------------------------------------------------------#    
    return X_train_wavf, X_train_spec, X_test_wavf, X_test_spec, np.array(y_train_new)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def augment_infrasound_training_data(X_train_infra, X_train_infra_T, X_test_infra, max_timeshift=5, augmentations=10, add_random_noise=True, std_noise_percent=5):
     
    # Augmenting training dataset
    X_train_infra_aug = []
    for entry in range(X_train_infra.shape[0]):
        starttime_shifts = random.sample(range(0,max_timeshift*10), augmentations)
        starttime_shifts = [x/10 for x in starttime_shifts] # can be from [0,0.1,0.2...max_timeshift]
        for shift in starttime_shifts:
            percent_trim = shift / (X_train_infra_T[entry].max() - X_train_infra_T[entry].min())
            npts_to_shift = int(X_train_infra[entry].shape[1] * percent_trim) # calculating number of data points to remove to accomodate for shift
            #-----------------------------------------------------------------------------------------------------------------------#
            # Apply shift
            new_infra_data = X_train_infra[entry][:,npts_to_shift:,:].copy()
            #-----------------------------------------------------------------------------------------------------------------------#
            if add_random_noise == True:
                # Add noise by channel
                for chn in range(new_infra_data.shape[2]):
                    # Mean centered around mean of channel (doing this because not every channel will have a mean around 0 like seismic - especially semblance)
                    noise = np.random.normal(np.mean(new_infra_data[:,:,chn]), np.abs(np.std(new_infra_data[:,:,chn]),dtype=np.float64)*(std_noise_percent/100), new_infra_data[:,:,chn].shape)
                    new_infra_data[:,:,chn] += noise
                    # We need to make sure semblance values remain bounded within 0 and 1 after noise augmentation
                    if chn == 2:
                        # Clip values between 0 and 1
                        new_infra_data[:,:,chn] = np.clip(new_infra_data[:,:,chn], 0, 1)                                                                     
            #-----------------------------------------------------------------------------------------------------------------------#
            # We need to resize back to its original shape
            slowX_resize = bilinear_resize(new_infra_data[:,:,0], new_h=X_train_infra[entry].shape[0], new_w=X_train_infra[entry].shape[1])
            slowY_resize = bilinear_resize(new_infra_data[:,:,1], new_h=X_train_infra[entry].shape[0], new_w=X_train_infra[entry].shape[1])
            Semb_resize = bilinear_resize(new_infra_data[:,:,2], new_h=X_train_infra[entry].shape[0], new_w=X_train_infra[entry].shape[1])
            #-----------------------------------------------------------------------------------------------------------------------#
            # Standardize data by channel
            new_infra_data_ss = np.zeros((X_train_infra[entry].shape[0], X_train_infra[entry].shape[1], X_train_infra[entry].shape[2]))
            new_infra_data_ss[:,:,0] = slowX_resize.copy(); new_infra_data_ss[:,:,1] = slowY_resize.copy(); new_infra_data_ss[:,:,2] = Semb_resize.copy()
            for channel in range(new_infra_data_ss.shape[2]):
                ss = StandardScaler().fit(new_infra_data_ss[:,:,channel])
                new_infra_data_ss[:,:,channel] = ss.transform(new_infra_data_ss[:,:,channel])
            #-----------------------------------------------------------------------------------------------------------------------#
            # Appending data
            X_train_infra_aug.append(np.nan_to_num(new_infra_data_ss))
    #-----------------------------------------------------------------------------------------------------------------------#
    # Standardize test set
    X_test_infra_ss = []
    for entry in range(X_test_infra.shape[0]):
        # Standardize data by channel
        new_infra_test_ss = X_test_infra[entry].copy()
        for channel in range(new_infra_test_ss.shape[2]):
            ss = StandardScaler().fit(new_infra_test_ss[:,:,channel])
            new_infra_test_ss[:,:,channel] = ss.transform(new_infra_test_ss[:,:,channel])
        #-----------------------------------------------------------------------------------------------------------------------#
        # Appending data
        X_test_infra_ss.append(np.nan_to_num(new_infra_test_ss))
    #-----------------------------------------------------------------------------------------------------------------------#
    print('Train dataset size is: ' + str(np.array(X_train_infra_aug).shape[0]))
    print('Test dataset size is: ' + str(np.array(X_test_infra_ss).shape[0]))
    #-----------------------------------------------------------------------------------------------------------------------#
    return np.array(X_train_infra_aug), np.array(X_test_infra_ss)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def separate_predictions(ev_test, yhat_pred):
    '''---------------------------------------------------------------------------
    Stores info relating to correct and incorrect model predictions.
    ---------------------------------------------------------------------------'''
    correct_eqs = []; incorrect_eqs = []
    correct_exps = []; incorrect_exps = []
    BRD_correct = 0; CHN_correct = 0; KSG_correct = 0
    BRD_incorrect = 0; CHN_incorrect = 0; KSG_incorrect = 0
    #-----------------------------------------------------------------------------------------------------------------------#
    # Earthquake predictions
    for ev in range(yhat_pred.shape[0]):
        # Correct
        if (np.round(yhat_pred[ev,0]) == 1) and (ev_test[ev,0] == 'Earthquake'):
            correct_eqs.append(ev_test[ev,:])
            if ev_test[ev,1] == 'BRD': 
                BRD_correct += 1
            elif ev_test[ev,1] == 'CHN': 
                CHN_correct += 1
            elif ev_test[ev,1] == 'KSG': 
                KSG_correct += 1
        # Incorrect
        elif (np.round(yhat_pred[ev,0]) != 1) and (ev_test[ev,0] == 'Earthquake'):
            incorrect_eqs.append(ev_test[ev,:])
            if ev_test[ev,1] == 'BRD': 
                BRD_incorrect += 1
            elif ev_test[ev,1] == 'CHN': 
                CHN_incorrect += 1
            elif ev_test[ev,1] == 'KSG': 
                KSG_incorrect += 1
    #-----------------------------------------------------------------------------------------------------------------------#
    # Explosion predictions
    for ev in range(yhat_pred.shape[0]):
        # Correct
        if (np.round(yhat_pred[ev,1]) == 1) and (ev_test[ev,0] == 'Explosion'):
            correct_exps.append(ev_test[ev,:])
            if ev_test[ev,1] == 'BRD': 
                BRD_correct += 1
            elif ev_test[ev,1] == 'CHN': 
                CHN_correct += 1
            elif ev_test[ev,1] == 'KSG': 
                KSG_correct += 1
        # Incorrect
        elif (np.round(yhat_pred[ev,1]) != 1) and (ev_test[ev,0] == 'Explosion'):
            incorrect_exps.append(ev_test[ev,:])
            if ev_test[ev,1] == 'BRD': 
                BRD_incorrect += 1
            elif ev_test[ev,1] == 'CHN': 
                CHN_incorrect += 1
            elif ev_test[ev,1] == 'KSG': 
                KSG_incorrect += 1        
    #-----------------------------------------------------------------------------------------------------------------------#
    correct_eqs = np.array(correct_eqs); incorrect_eqs = np.array(incorrect_eqs)
    correct_exps = np.array(correct_exps); incorrect_exps = np.array(incorrect_exps)
    #-----------------------------------------------------------------------------------------------------------------------#
    return correct_eqs, incorrect_eqs, correct_exps, incorrect_exps, [BRD_correct,BRD_incorrect], [CHN_correct,CHN_incorrect], [KSG_correct,KSG_incorrect]

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def corrected_events(incorrect_seismic, incorrect_seismoacoustic):

    incorrect_exps_seismic_list = []
    for ev_idx_seismic in range(len(incorrect_seismic)):
        incorrect_exps_seismic_list.append(tuple(incorrect_seismic[ev_idx_seismic]))
    incorrect_exps_seismoacoustic_list = []
    for ev_idx_seismoacoustic in range(len(incorrect_seismoacoustic)):
        incorrect_exps_seismoacoustic_list.append(tuple(incorrect_seismoacoustic[ev_idx_seismoacoustic]))
    #-----------------------------------------------------------------------------------------------------------------------#
    # Removing any new incorrect predictions that were not in seismic incorrect predictions
    new_incorrect_evs = []
    for incorrect in incorrect_exps_seismoacoustic_list:
        if incorrect in incorrect_exps_seismic_list:
            continue
        else:
            new_incorrect_evs.append(incorrect)
    for new_incorrect in new_incorrect_evs:
        if new_incorrect in incorrect_exps_seismoacoustic_list:
            incorrect_exps_seismoacoustic_list.remove(new_incorrect)
        else:
            continue
    merged_lists = incorrect_exps_seismic_list + incorrect_exps_seismoacoustic_list
    merged_lists.sort()
    grouped = groupby(merged_lists)
    duplicates = [key for key, group in grouped if len(list(group)) > 1]
    #-----------------------------------------------------------------------------------------------------------------------#
    # These are the ones that stayed incorrect
    print('The events that were not corrected are:')
    not_corrected_evs = []
    for duplicate in duplicates:
        print(duplicate); not_corrected_evs.append(duplicate)
    for duplicate in duplicates:    
        merged_lists.remove(duplicate)
        merged_lists.remove(duplicate)
    print('The events that were corrected are:')
    for ev in merged_lists:
        print(ev)
    print('New incorrect events are:')
    for new_incorrect_ev in new_incorrect_evs:
        print(new_incorrect_ev)
    return not_corrected_evs, merged_lists, new_incorrect_evs

'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############### Array Processing ##############
############### ############### ###############

def get_slowness_vector_time_shifts(st, ref_station, baz, tr_vel, return_params=False):
    '''---------------------------------------------------------------------------
    Get time shifts based on back azimuth and trace velocity

    *From Cardinal (https://github.com/sjarrowsmith/cardinal.git)
    ---------------------------------------------------------------------------'''
    # Computing array coordinates
    X = np.zeros((len(st), 2))
    stnm = []
    for i in range(0, len(st)):
        E, N, _, _ = utm.from_latlon(st[i].stats.sac.stla, st[i].stats.sac.stlo)
        X[i,0] = E; X[i,1] = N
        stnm.append(st[i].stats.station)
    #-----------------------------------------------------------------------------------------------------------------#
    # Adjusting coordinates to reference station
    ref_station_ix = np.where(np.array(stnm) == ref_station)[0][0]
    X[:,0] = (X[:,0] - X[ref_station_ix,0])
    X[:,1] = (X[:,1] - X[ref_station_ix,1])
    #-----------------------------------------------------------------------------------------------------------------#
    # Computing the slowness vector for a specified backazimuth and trace velocity
    sl_y = np.abs(np.sqrt((1/tr_vel**2)/((np.tan(np.deg2rad(baz)))**2+1)))
    sl_x = np.abs(sl_y * np.tan(np.deg2rad(baz)))
    if baz > 180:
        sl_x = -sl_x
    if (baz > 90) and (baz < 270):
        sl_y = -sl_y
    #-----------------------------------------------------------------------------------------------------------------#
    # Computing time shifts for the slowness vector defined by sl_x, sl_y
    st_sc = st.copy()
    t_shifts = []
    for i in range(0, X.shape[0]):
        t_shift = (X[i,0]*sl_x + X[i,1]*sl_y)
        st_sc[i].stats.starttime = st_sc[i].stats.starttime + t_shift
        t_shifts.append(t_shift)
    t_shifts = np.array(t_shifts)
    #-----------------------------------------------------------------------------------------------------------------#
    # Return params
    if return_params == True:
        return t_shifts, st_sc, sl_x, sl_y
    else:
        return t_shifts

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
   
def beamform(t_shifts, st, ref_station, trace_spacing=2, normalize_data=False, normalize_beam=False, divide_by_num_stns=False, plot=False):
    '''---------------------------------------------------------------------------
    Beamform signals based on time shifts [t_shifts]

    *From Cardinal (https://github.com/sjarrowsmith/cardinal.git)
    ---------------------------------------------------------------------------'''
    tr_ref = st.select(station=ref_station)[0]
    ix = 0
    for tr in st:
    #-----------------------------------------------------------------------------------------------------------------#
    # Need to truncate and pad data
        t_ix = np.arange(0, abs(t_shifts[ix]), tr.stats.delta)
        if t_shifts[ix] < 0:
            data = tr.data[len(t_ix)::]
        elif t_shifts[ix] > 0:
            data = np.concatenate((np.zeros((len(t_ix))), tr.data))
        else:
            data = tr.data
    #-----------------------------------------------------------------------------------------------------------------#
    # Computing stack
        if ix == 0:
            stack = data
        else:
            diff = len(stack)-len(data)
            if diff > 0:
                data = np.concatenate((data, np.zeros(diff)))
            elif diff < 0:
                stack = np.concatenate((stack, np.zeros(np.abs(diff))))
            stack = data + stack
        if normalize_data == True:
            data = data/(np.max(np.abs(data)))
        t_beam = np.arange(0, len(data)*tr_ref.stats.delta, tr_ref.stats.delta)
        if plot == True:
            try:
                plt.plot(t_beam, ix*trace_spacing + data, 'k')
            except:
                t_beam = np.arange(0.001, len(data)*tr_ref.stats.delta, tr_ref.stats.delta)
                plt.plot(t_beam, ix*trace_spacing + data, 'k')
        ix += 1
    if divide_by_num_stns == True:
        beam = stack/len(st)
    else:
        beam = stack.copy()
    if normalize_beam == True:
        beam = beam/np.max(np.abs(beam))
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting beam
    if plot == True:
        try:
            plt.plot(np.arange(0, len(data)*tr_ref.stats.delta, tr_ref.stats.delta), ix*trace_spacing + beam, 'r')
        except:
            plt.plot(np.arange(0.001, len(data)*tr_ref.stats.delta, tr_ref.stats.delta), ix*trace_spacing + beam, 'r')
        plt.xlabel('Time (s) after ' + str(tr_ref.stats.starttime).split('.')[0].replace('T', ' '))
        plt.gca().get_yaxis().set_ticks([])
    #-----------------------------------------------------------------------------------------------------------------#
    # Return params
    return t_beam, beam

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def convert_to_slowness(baz, vel):
    '''---------------------------------------------------------------------------
    Converts a backazimuth and phase/trace velocity to a slowness vector

    *From Cardinal (https://github.com/sjarrowsmith/cardinal.git)
    ---------------------------------------------------------------------------'''
    
    sl_y = np.abs(np.sqrt((1/vel**2) / ((np.tan(np.deg2rad(baz)))**2+1)))
    sl_x = np.abs(sl_y * np.tan(np.deg2rad(baz)))
    if baz > 180:
        sl_x = -sl_x
    if (baz > 90) and (baz < 270):
        sl_y = -sl_y
    
    return sl_x, sl_y

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def make_custom_fbands(f_min=0.01, f_max=50, win_min=3, win_max=200, overlap=0.1, type='third_octave'):
    '''---------------------------------------------------------------------------
    Makes a set of custom frequency bands and time windows for processing

    Input parameters:
    f_min is the minimum frequency
    f_max is the maximum frequency
    win_min is the minimum time window (to be used for maximum frequency)
    win_max is the maximum time window (to be used for minimum frequency)
    type is the type of filter band to use

    *From Cardinal (https://github.com/sjarrowsmith/cardinal.git)
    ---------------------------------------------------------------------------'''
    m, b = np.polyfit([1/f_min, 1/f_max], [win_max, win_min], 1)
    column_names = ['band', 'fmin', 'fcenter', 'fmax', 'win', 'step']
    f_bands = pd.DataFrame(columns = column_names)
    #-----------------------------------------------------------------------------------------------------------------------#
    if type == 'third_octave':
        i = 0
        while f_min * np.cbrt(2) <= f_max:
            i = i + 1
            fmin = f_min
            fmax = f_min * np.cbrt(2)
            fcenter = (fmin + fmax)/2
            win = m * (1/fcenter) + b
            step = win * overlap
            f_min = fmax
            f_bands.loc[-1] = [i, fmin, fcenter, fmax, win, step]  # adding a row
            f_bands.index = f_bands.index + 1  # shifting index
    #-----------------------------------------------------------------------------------------------------------------------#
    elif type == 'octave':
        i = 0
        while f_min * 2 <= f_max:
            i = i + 1
            fmin = f_min
            fmax = f_min * 2
            fcenter = (fmin + fmax)/2
            win = m * (1/fcenter) + b
            step = win * overlap
            f_min = fmax
            f_bands.loc[-1] = [i, fmin, fcenter, fmax, win, step]  # adding a row
            f_bands.index = f_bands.index + 1  # shifting index
    #-----------------------------------------------------------------------------------------------------------------------#
    elif type == 'decade':
        i = 0
        while f_min * 10 <= f_max:
            i = i + 1
            fmin = f_min
            fmax = f_min * 10
            fcenter = (fmin + fmax)/2
            win = m * (1/fcenter) + b
            step = win * overlap
            f_min = fmax
            f_bands.loc[-1] = [i, fmin, fcenter, fmax, win, step]  # adding a row
            f_bands.index = f_bands.index + 1  # shifting index
    #-----------------------------------------------------------------------------------------------------------------------#
    f_bands.index = f_bands.index[::-1]
    return f_bands

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def sliding_time_array_fk(st, element, tstart=None, tend=None, win_len=20, win_frac=0.5, frqlow=0.5, frqhigh=4,
                          sll_x=-3.6, slm_x=3.6, sll_y=-3.6, slm_y=3.6, sl_s=0.18, sl_corr=[0.,0.],
                          normalize_waveforms=True, use_geographic_coords=True):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Processes stream (st) with sliding window FK analysis.

    *From Cardinal (https://github.com/sjarrowsmith/cardinal.git)
    ----------------------------------------------------------------------------------------------------------------------------------'''
    tr = st.select(station=element)[0]    # Trace of reference element
    #-----------------------------------------------------------------------------------------------------------------------#
    # Defining t_start, t_end:
    if (tstart == None) and (tend == None):
        tstart = 1
        tend = (tr.stats.npts * tr.stats.delta)-1
    #-----------------------------------------------------------------------------------------------------------------------#
    if use_geographic_coords:
        for st_i in st:
            st_i.stats.coordinates = AttribDict({
                'latitude': st_i.stats.sac.stla,
                'elevation': 0.,
                'longitude': st_i.stats.sac.stlo})
    #-----------------------------------------------------------------------------------------------------------------------#
    kwargs = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sl_s,
            # sliding window properties
            win_len=win_len, win_frac=win_frac,
            # frequency properties
            frqlow=frqlow, frqhigh=frqhigh, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
            stime=tr.stats.starttime+tstart, etime=tr.stats.starttime+tend, verbose=False,
            sl_corr=sl_corr, normalize_waveforms=normalize_waveforms
        )
    slid_fk = array_processing(st, **kwargs)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Convert times to seconds after start time of reference element and adjusting to center of time window:
    T = ((slid_fk[:,0] - date2num(tr.stats.starttime.datetime))*86400) + win_len/2
    # Convert backazimuths to degrees from North:
    B = slid_fk[:,3] % 360.
    # Convert slowness to phase velocity:
    V = 1/slid_fk[:,4]
    # Semblance:
    S = slid_fk[:,1]
    #-----------------------------------------------------------------------------------------------------------------------#
    return T, B, V, S

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def sliding_time_array_fk_multifreq(st, element, f_bands, t_start=None, t_end=None, n_workers=1,
                                    sll_x=-3.6, slm_x=3.6, sll_y=-3.6, slm_y=3.6, sl_s=0.18, sl_corr=[0.,0.],
                                    use_geographic_coords=True):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Processes stream (st) with sliding window FK analysis in multiple frequency bands.

    Required inputs:
    - st contains the ObsPy Stream of the data
    - f_bands is a Pandas dataframe containing the frequency band information

    Optional parameters:
    - t_start is the start time (in seconds after the start of the ObsPy Stream) to process (None processes the whole Stream)
    - t_end is the end time (in seconds after the start of the ObsPy Stream) to process (None processes the whole Stream)
    - n_workers is the number of threads to use to do the computation (Using dask library if n_workers > 1)
    - Remaining parameters define the slowness plane

    *From Cardinal (https://github.com/sjarrowsmith/cardinal.git)
    ----------------------------------------------------------------------------------------------------------------------------------'''
    if n_workers > 1:
        client = Client(threads_per_worker=1, n_workers=n_workers)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Trace of reference element
    tr = st.select(station=element)[0]
    #-----------------------------------------------------------------------------------------------------------------------#
    # Defining t_start, t_end:
    if (t_start == None) and (t_end == None):
        t_start = 1
        t_end = (tr.stats.npts * tr.stats.delta)-1
    #-----------------------------------------------------------------------------------------------------------------------#
    # Processing each frequency band with sliding window FK processing:
    T_all = []; B_all = []; V_all = []; S_all = []; dask_all = []
    for f_band in f_bands['band'].values:
        win_len = f_bands[f_bands['band'] == f_band]['win'].values[0]
        frqlow = f_bands[f_bands['band'] == f_band]['fmin'].values[0]
        frqhigh = f_bands[f_bands['band'] == f_band]['fmax'].values[0]
        win_frac = f_bands[f_bands['band'] == f_band]['step'].values[0]/f_bands[f_bands['band'] == f_band]['win'].values[0]

        if n_workers == 1:
            T, B, V, S = sliding_time_array_fk(st, element, tstart=t_start, tend=t_end, win_len=win_len, win_frac=win_frac, 
                                               frqlow=frqlow, frqhigh=frqhigh,
                                               sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sl_s,
                                               sl_corr=sl_corr, use_geographic_coords=use_geographic_coords)
            T_all.append(T); B_all.append(B); V_all.append(V); S_all.append(S)
        else:
            dask_out = dask.delayed(sliding_time_array_fk)(st, element, tstart=t_start, tend=t_end, 
                                                           win_len=win_len, win_frac=win_frac, 
                                                           frqlow=frqlow, frqhigh=frqhigh,
                                                           sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, 
                                                           slm_y=slm_y, sl_s=sl_s, sl_corr=sl_corr,
                                                           use_geographic_coords=use_geographic_coords)
            dask_all.append(dask_out)
    if n_workers > 1:
        # Organizing output from distributed process:
        out = dask.compute(*dask_all)
        for out_i in out:
            T_all.append(out_i[0]); B_all.append(out_i[1]); V_all.append(out_i[2]); S_all.append(out_i[3])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Extracting the time vector corresponding to the maximum number of values:
    N = []
    for T in T_all:
        N.append(len(T))
    T = T_all[np.argmax(N)]    # Times for f-band with highest number of DOA estimates
    #-----------------------------------------------------------------------------------------------------------------------#
    # Re-sampling array processing results to produce time/frequency matrices:
    NF = len(f_bands)
    NT = len(T)
    B = np.zeros((NF, NT))
    V = np.zeros((NF, NT))
    S = np.zeros((NF, NT))
    for i in range(0, NF):
        T_i = T_all[i]; B_i = B_all[i]; V_i = V_all[i]; S_i = S_all[i]
        for j in range(0, NT):
            ix = np.argmin(np.abs(T[j] - T_i))
            B[i,j] = B_i[ix]
            V[i,j] = V_i[ix]
            S[i,j] = S_i[ix]
    #-----------------------------------------------------------------------------------------------------------------------#
    if n_workers > 1:
        client.close()
    #-----------------------------------------------------------------------------------------------------------------------#
    return T, B, V, S

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############### Data Extraction ###############
############### ############### ###############

def seismic_modality(database_filename, arrays, outwavfdir, site_dir='Korea_Array_Locations', freqmin=1, freqmax=10, skip_events=None, make_noise=False):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Storing seismic waveforms.
    ----------------------------------------------------------------------------------------------------------------------------------'''
    # Reading events info from Excel as Pandas DataFrame
    df = pd.read_excel(database_filename, skiprows=0, header=0)
    # Geographic locations of each station
    df_brd = pd.read_table(site_dir+'/BRDAR_Seismic_Locations', header=None, sep='\s+', names=['stn', 'lat', 'lon'])
    df_chn = pd.read_table(site_dir+'/CHNAR_Seismic_Locations', header=None, sep='\s+', names=['stn', 'lat', 'lon'])
    df_ksg = pd.read_table(site_dir+'/KSGAR_Seismic_Locations', header=None, sep='\s+', names=['stn', 'lat', 'lon'])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Making indeces for events to run through
    if skip_events is None:
        events_idx = np.arange(0, len(df), 1)
    else:
        events_idx = np.arange(skip_events, len(df), 1) # skipping events (helps debugging)
    for event_idx in events_idx:
        # Retrieving data
        print('Begin processing for event number ' + str(event_idx+1) + ' out of ' + str(len(df)))
        #-----------------------------------------------------------------------------------------------------------------------#
        # Parameters needed for data extraction
        event = df.loc[event_idx]
        ev_date = event['Event Date/Time'].split(' ')[0]; month = ev_date.split('/')[1]; day = ev_date.split('/')[2]
        ev_time = event['Event Date/Time'].split(' ')[1]
        BRD_window = [event['BRD Seismic Starttime (s)'], event['BRD Seismic Endtime (s)']]
        CHN_window = [event['CHN Seismic Starttime (s)'], event['CHN Seismic Endtime (s)']]
        KSG_window = [event['KSG Seismic Starttime (s)'], event['KSG Seismic Endtime (s)']]
        #-----------------------------------------------------------------------------------------------------------------------#
        # Assess whether time windows from successive event overlap
        if event_idx == len(df)-1:
            BRD = 'continue'
            CHN = 'continue'
            KSG = 'continue'
            pass
        else:
            # If starttime from next event is before endtime from this event - we can't extract waveforms
            next_event = df.loc[event_idx+1]
            if UTCDateTime(next_event['Event Date/Time']) + next_event['BRD Seismic Starttime (s)'] < UTCDateTime(event['Event Date/Time']) + BRD_window[1]:
                BRD = 'stop'
            else:
                BRD = 'continue'
            if UTCDateTime(next_event['Event Date/Time']) + next_event['CHN Seismic Starttime (s)'] < UTCDateTime(event['Event Date/Time']) + CHN_window[1]:
                CHN = 'stop'
            else:
                CHN = 'continue'
            if UTCDateTime(next_event['Event Date/Time']) + next_event['KSG Seismic Starttime (s)'] < UTCDateTime(event['Event Date/Time']) + KSG_window[1]:
                KSG = 'stop'
            else:
                KSG = 'continue'
        #-----------------------------------------------------------------------------------------------------------------------#
        # Converting event date to julian date so we can read in data from directory
        YMD = '%Y/%m/%d' # Year/Month/Day
        julian_day, year = YMD_to_JD(YMD, ev_date)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Retrieving stream for desired arrays
        if ev_time[:2] == '23': # if close to end of day we need to also pull data from next day (no window is greater than an hour so '23' is fine)
            st = read_data(julian_day, year, month, day, arrays, data='seismic')
            st += read_data(julian_day+1, year, month, day, arrays, data='seismic')
            try:
                st = st.merge()
            except:
                for tr in st:
                    tr.stats.calib = 1.0 # just in case calib factors aren't the same (can't merge if they aren't)
                try:
                    st = st.merge()
                except:
                    print('Could not merge traces!')
        else:
            st = read_data(julian_day, year, month, day, arrays, data='seismic')
            try:
                st = st.merge()
            except:
                for tr in st:
                    tr.stats.calib = 1.0 # just in case calib factors aren't the same (can't merge if they aren't)
                try:
                    st = st.merge()
                except:
                    print('Could not merge traces!')
        #-----------------------------------------------------------------------------------------------------------------------#
        # Filtering
        st_filt = st.copy()
        st_filt.taper(type='cosine', max_percentage=0.05, max_length=60)
        try:
            st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
        except:
            # Need to account for possible masked traces
            st_filt = st_filt.split()
            st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
            st_filt.merge()
        #-----------------------------------------------------------------------------------------------------------------------#
        # Event origin time in seconds
        HMS = '%H:%M:%S' # Hour:Minute:Second
        if make_noise == False:
            origin_time = HMS_to_S(HMS, ev_time)
        elif make_noise == True:
            origin_time = HMS_to_S(HMS, ev_time) - (BRD_window[1]-BRD_window[0]) 
        #-----------------------------------------------------------------------------------------------------------------------#
        # Formatting julian day for saving waveforms
        if julian_day < 10:
            julian_day = '00' + str(julian_day)
        elif 10 <= julian_day < 100:
            julian_day = '0' + str(julian_day)
        else:
            julian_day = str(julian_day)
        #-----------------------------------------------------------------------------------------------------------------------#
        # BRDAR
        if BRD == 'continue':
            try:
                # Append metadata and trim array data
                st_brd = append_loc_info(st_filt, df_brd, 'BRD')
                st_brd = st_brd.trim(st_brd[0].stats.starttime+origin_time+BRD_window[0], st_brd[0].stats.starttime+origin_time+BRD_window[1])
                #-----------------------------------------------------------------------------------------------------------------------#
                # Fixing calibration value of 0.0
                for tr in st_brd:
                    if tr.stats.calib == 0.0:
                        tr.stats.calib = 1.0
                    else:
                        continue
                #-----------------------------------------------------------------------------------------------------------------------#
                # Saving Obspy stream data as miniseed
                try:
                    if make_noise == True:
                        st_brd.write(outwavfdir+'/BRD/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                    else:
                        st_brd.write(outwavfdir+'/BRD/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                except:
                    st_brd = st_brd.split()
                    if make_noise == True:
                        st_brd.write(outwavfdir+'/BRD/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                    else:
                        st_brd.write(outwavfdir+'/BRD/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                print('Done saving BRD waveforms')
            except Exception as inst:
                print(inst)
                print('Could not store BRD array data')
                pass
        elif BRD == 'stop':
            print('Skipping BRD event due to overlapping time windows from successive event')
        #-----------------------------------------------------------------------------------------------------------------------#
        # CHNAR
        if CHN == 'continue':
            try:
                # Append metadata and trim array data
                st_chn = append_loc_info(st_filt, df_chn, 'CHN')
                st_chn = st_chn.trim(st_chn[0].stats.starttime+origin_time+CHN_window[0], st_chn[0].stats.starttime+origin_time+CHN_window[1])
                #-----------------------------------------------------------------------------------------------------------------------#
                # Fixing calibration value of 0.0
                for tr in st_chn:
                    if tr.stats.calib == 0.0:
                        tr.stats.calib = 1.0
                    else:
                        continue
                #-----------------------------------------------------------------------------------------------------------------------#
                # Saving Obspy stream data as miniseed
                try:
                    if make_noise == True:
                        st_chn.write(outwavfdir+'/CHN/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                    else:
                        st_chn.write(outwavfdir+'/CHN/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                except:
                    st_chn = st_chn.split()
                    if make_noise == True:
                        st_chn.write(outwavfdir+'/CHN/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                    else:
                        st_chn.write(outwavfdir+'/CHN/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                print('Done saving CHN waveforms')
            except Exception as inst:
                print(inst)
                print('Could not store CHN array data')
                pass
        elif CHN == 'stop':
            print('Skipping CHN event due to overlapping time windows from successive event')
        #-----------------------------------------------------------------------------------------------------------------------#
        # KSGAR
        if KSG == 'continue':
            try:
                # Append metadata and trim array data
                st_ksg = append_loc_info(st_filt, df_ksg, 'KSG')
                st_ksg = st_ksg.trim(st_ksg[0].stats.starttime+origin_time+KSG_window[0], st_ksg[0].stats.starttime+origin_time+KSG_window[1])
                #-----------------------------------------------------------------------------------------------------------------------#
                # Fixing calibration value of 0.0
                for tr in st_ksg:
                    if tr.stats.calib == 0.0:
                        tr.stats.calib = 1.0
                    else:
                        continue
                #-----------------------------------------------------------------------------------------------------------------------#
                # Saving Obspy stream data as miniseed
                try:
                    if make_noise == True:
                        st_ksg.write(outwavfdir+'/KSG/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                    else:
                        st_ksg.write(outwavfdir+'/KSG/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                except: 
                    st_ksg = st_ksg.split()
                    if make_noise == True:
                        st_ksg.write(outwavfdir+'/KSG/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                    else:
                        st_ksg.write(outwavfdir+'/KSG/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                print('Done saving KSG waveforms')
            except Exception as inst:
                print(inst)
                print('Could not store KSG array data')
                pass
        elif KSG == 'stop':
            print('Skipping KSG event due to overlapping time windows from successive event')

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def infrasound_modality(database_filename, arrays, outresultsdir, site_dir='Korea_Array_Locations', freqmin=0.5, freqmax=10, f_band_type='third_octave', skip_events=None, make_noise=False, array_processing=True, 
                        save_waveforms=False, outwavfdir=None, n_workers=12, local_infrasound=False):
    '''----------------------------------------------------------------------------------------------------------------------------------
    Stores infrasound array processing results (Timestamp [T], Back azimuth [B], Trace velocity [V], and Semblance [S])
    ----------------------------------------------------------------------------------------------------------------------------------'''
    # Reading events info from Excel as Pandas DataFrame
    df = pd.read_excel(database_filename, skiprows=0, header=0)
    # Geographic locations of each station
    df_brd = pd.read_table(site_dir+'/BRDAR_Infrasound_Locations', header=None, sep='\s+', names=['stn', 'lat', 'lon'])
    df_chn = pd.read_table(site_dir+'/CHNAR_Infrasound_Locations', header=None, sep='\s+', names=['stn', 'lat', 'lon'])
    df_ksg = pd.read_table(site_dir+'/KSGAR_Infrasound_Locations', header=None, sep='\s+', names=['stn', 'lat', 'lon'])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Constructing custom frequency bands - for array processing
    f_bands = make_custom_fbands(f_min=freqmin, f_max=freqmax+1, type=f_band_type) 
    f_bands['fmax'][-1:] = np.floor(f_bands['fmax'][-1:]) # round
    #-----------------------------------------------------------------------------------------------------------------------#
    # Making indeces for events to run through
    if skip_events is None:
        events_idx = np.arange(0, len(df), 1)
    else:
        events_idx = np.arange(skip_events, len(df), 1) # skipping events (helps debugging)
    for event_idx in events_idx:
        # Retrieving data
        print('Begin processing for event number ' + str(event_idx+1) + ' out of ' + str(len(df)))
        #-----------------------------------------------------------------------------------------------------------------------#
        # Parameters needed for data extraction
        event = df.loc[event_idx]
        ev_date = event['Event Date/Time'].split(' ')[0]; month = ev_date.split('/')[1]; day = ev_date.split('/')[2]
        ev_time = event['Event Date/Time'].split(' ')[1]
        if local_infrasound == True: # window is 3 mins long after seismic start time
            BRD_window = [np.floor(event['BRD Seismic Starttime (s)']), np.floor(event['BRD Seismic Starttime (s)']) + (3*60)]
            CHN_window = [np.floor(event['CHN Seismic Starttime (s)']), np.floor(event['CHN Seismic Starttime (s)']) + (3*60)]
            KSG_window = [np.floor(event['KSG Seismic Starttime (s)']), np.floor(event['KSG Seismic Starttime (s)']) + (3*60)]
        else:
            BRD_window = [np.floor(event['BRD Infrasound Starttime (s)']), np.ceil(event['BRD Infrasound Endtime (s)'])]
            CHN_window = [np.floor(event['CHN Infrasound Starttime (s)']), np.ceil(event['CHN Infrasound Endtime (s)'])]
            KSG_window = [np.floor(event['KSG Infrasound Starttime (s)']), np.ceil(event['KSG Infrasound Endtime (s)'])]
        #-----------------------------------------------------------------------------------------------------------------------#
        # Need to check if window lengths for each array are smaller than max window in freq bands
        if (BRD_window[1] - BRD_window[0]) < f_bands['win'].max():
            # Extend window equally in both directions to match max freq band window length
            diff = f_bands['win'].max() - (BRD_window[1] - BRD_window[0])
            BRD_window[0] -= np.ceil(diff/2); BRD_window[1] += np.ceil(diff/2)
        if (CHN_window[1] - CHN_window[0]) < f_bands['win'].max():
            # Extend window equally in both directions to match max freq band window length
            diff = f_bands['win'].max() - (CHN_window[1] - CHN_window[0])
            CHN_window[0] -= np.ceil(diff/2); CHN_window[1] += np.ceil(diff/2)
        if (KSG_window[1] - KSG_window[0]) < f_bands['win'].max():
            # Extend window equally in both directions to match max freq band window length
            diff = f_bands['win'].max() - (KSG_window[1] - KSG_window[0])
            KSG_window[0] -= np.ceil(diff/2); KSG_window[1] += np.ceil(diff/2)
        #-----------------------------------------------------------------------------------------------------------------------#
        # We don't want to skip any events for infrasound processing
        BRD = 'continue'
        CHN = 'continue'
        KSG = 'continue'
        #-----------------------------------------------------------------------------------------------------------------------#
        # Converting event date to julian date so we can read in data from directory
        YMD = '%Y/%m/%d' # Year/Month/Day
        julian_day, year = YMD_to_JD(YMD, ev_date)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Retrieving stream for desired arrays
        if ev_time[:2] == '23': # if close to end of day we need to also pull data from next day (no window is greater than an hour so '23' is fine)
            st = read_data(julian_day, year, month, day, arrays, data='infrasound')
            st += read_data(julian_day+1, year, month, day, arrays, data='infrasound')
            try:
                st = st.merge()
            except:
                for tr in st:
                    tr.stats.calib = 1.0 # just in case calib factors aren't the same (can't merge if they aren't)
                try:
                    st = st.merge()
                except:
                    print('Could not merge traces!')
        else:
            st = read_data(julian_day, year, month, day, arrays, data='infrasound')
            try:
                st = st.merge()
            except:
                for tr in st:
                    tr.stats.calib = 1.0 # just in case calib factors aren't the same (can't merge if they aren't)
                try:
                    st = st.merge()
                except:
                    print('Could not merge traces!')
        #-----------------------------------------------------------------------------------------------------------------------#
        # Filtering
        st_filt = st.copy()
        st_filt.taper(type='cosine', max_percentage=0.05, max_length=60)
        try:
            st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
        except:
            # Need to account for possible masked traces
            st_filt = st_filt.split()
            st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
            st_filt.merge()  
        #-----------------------------------------------------------------------------------------------------------------------#
        # Event origin time in seconds
        HMS = '%H:%M:%S' # Hour:Minute:Second
        if make_noise == False:
            origin_time = HMS_to_S(HMS, ev_time)
        elif make_noise == True:
            origin_time_brd = HMS_to_S(HMS, ev_time) - (BRD_window[1]-BRD_window[0]) 
            origin_time_chn = HMS_to_S(HMS, ev_time) - (CHN_window[1]-CHN_window[0]) 
            origin_time_ksg = HMS_to_S(HMS, ev_time) - (KSG_window[1]-KSG_window[0]) 
        #-----------------------------------------------------------------------------------------------------------------------#
        # Formatting julian day for saving results
        if julian_day < 10:
            julian_day = '00' + str(julian_day)
        elif 10 <= julian_day < 100:
            julian_day = '0' + str(julian_day)
        else:
            julian_day = str(julian_day)
        #-----------------------------------------------------------------------------------------------------------------------#
        # BRDAR
        if BRD == 'continue':
            try:
                # Append metadata and trim array data
                st_brd_orig = append_loc_info(st_filt, df_brd, 'BRD')
                st_brd = st_brd_orig.copy()
                if make_noise == True:
                    st_brd = st_brd.trim(st_brd[0].stats.starttime+origin_time_brd+BRD_window[0], st_brd[0].stats.starttime+origin_time_brd+BRD_window[1])
                else:
                    st_brd = st_brd.trim(st_brd[0].stats.starttime+origin_time+BRD_window[0], st_brd[0].stats.starttime+origin_time+BRD_window[1])
                #-----------------------------------------------------------------------------------------------------------------------#
                # Fixing calibration value of 0.0
                for tr in st_brd:
                    if tr.data.sum() == 0.0:
                        st_brd.remove(tr)
                    if tr.stats.calib == 0.0:
                        tr.stats.calib = 1.0
                    else:
                        continue
                #-----------------------------------------------------------------------------------------------------------------------#
                if save_waveforms == True:
                    try:
                        # Saving Obspy stream data as miniseed
                        try:
                            if make_noise == True:
                                st_brd.write(outwavfdir+'/BRD/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                            else:
                                st_brd.write(outwavfdir+'/BRD/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                        except:
                            st_brd = st_brd.split()
                            if make_noise == True:
                                st_brd.write(outwavfdir+'/BRD/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                            else:
                                st_brd.write(outwavfdir+'/BRD/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                        print('Done saving BRD infrasound waveforms')
                    except:
                        if outwavfdir == None:
                            raise Exception('outwavfdir needs to be specified if save_waveforms is set to True')
                        else:
                            raise Exception('Could not save BRD infrasound waveforms')
            except:
                print('Could not store BRD array data')
                pass
        elif BRD == 'stop':
            print('Skipping BRD event due to overlapping time windows from successive event')
        #-----------------------------------------------------------------------------------------------------------------------#
        # CHNAR
        if CHN == 'continue':
            try:
                # Append metadata and trim array data
                st_chn_orig = append_loc_info(st_filt, df_chn, 'CHN')
                st_chn = st_chn_orig.copy()
                if make_noise == True:
                    st_chn = st_chn.trim(st_chn[0].stats.starttime+origin_time_chn+CHN_window[0], st_chn[0].stats.starttime+origin_time_chn+CHN_window[1])
                else:
                    st_chn = st_chn.trim(st_chn[0].stats.starttime+origin_time+CHN_window[0], st_chn[0].stats.starttime+origin_time+CHN_window[1])
                #-----------------------------------------------------------------------------------------------------------------------#
                # Fixing calibration value of 0.0
                for tr in st_chn:
                    if tr.data.sum() == 0.0:
                        st_chn.remove(tr)
                    if tr.stats.calib == 0.0:
                        tr.stats.calib = 1.0
                    else:
                        continue
                #-----------------------------------------------------------------------------------------------------------------------#
                if save_waveforms == True:
                    try:
                        # Saving Obspy stream data as miniseed
                        try:
                            if make_noise == True:
                                st_chn.write(outwavfdir+'/CHN/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                            else:
                                st_chn.write(outwavfdir+'/CHN/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                        except:
                            st_chn = st_chn.split()
                            if make_noise == True:
                                st_chn.write(outwavfdir+'/CHN/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                            else:
                                st_chn.write(outwavfdir+'/CHN/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                        print('Done saving CHN infrasound waveforms')
                    except:
                        if outwavfdir == None:
                            raise Exception('outwavfdir needs to be specified if save_waveforms is set to True')
                        else:
                            raise Exception('Could not save CHN infrasound waveforms')
            except:
                print('Could not store CHN array data')
                pass
        elif CHN == 'stop':
            print('Skipping CHN event due to overlapping time windows from successive event')
        #-----------------------------------------------------------------------------------------------------------------------#
        # KSGAR
        if KSG == 'continue':
            try:
                # Append metadata and trim array data
                st_ksg_orig = append_loc_info(st_filt, df_ksg, 'KSG')
                st_ksg = st_ksg_orig.copy()
                if make_noise == True:
                    st_ksg = st_ksg.trim(st_ksg[0].stats.starttime+origin_time_ksg+KSG_window[0], st_ksg[0].stats.starttime+origin_time_ksg+KSG_window[1])
                else:
                    st_ksg = st_ksg.trim(st_ksg[0].stats.starttime+origin_time+KSG_window[0], st_ksg[0].stats.starttime+origin_time+KSG_window[1])
                #-----------------------------------------------------------------------------------------------------------------------#
                # Fixing calibration value of 0.0
                for tr in st_ksg:
                    if tr.data.sum() == 0.0:
                        st_ksg.remove(tr)
                    if tr.stats.calib == 0.0:
                        tr.stats.calib = 1.0
                    else:
                        continue
                #-----------------------------------------------------------------------------------------------------------------------#
                if save_waveforms == True:
                    try:
                        # Saving Obspy stream data as miniseed
                        try:
                            if make_noise == True:
                                st_ksg.write(outwavfdir+'/KSG/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                            else:
                                st_ksg.write(outwavfdir+'/KSG/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                        except:
                            st_ksg = st_ksg.split()
                            if make_noise == True:
                                st_ksg.write(outwavfdir+'/KSG/NOISE_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                            else:
                                st_ksg.write(outwavfdir+'/KSG/'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.mseed', format='MSEED')
                        print('Done saving KSG infrasound waveforms')
                    except:
                        if outwavfdir == None:
                            raise Exception('outwavfdir needs to be specified if save_waveforms is set to True')
                        else:
                            raise Exception('Could not save KSG infrasound waveforms')
            except:
                print('Could not store KSG array data')
                pass
        elif KSG == 'stop':
            print('Skipping KSG event due to overlapping time windows from successive event')
        #-----------------------------------------------------------------------------------------------------------------------#
        if array_processing == True:
            # Begin array processing using Cardinal
            st_korea = []
            if BRD == 'continue':
                # Remove traces with endtimes before window endtime
                for tr in st_brd:
                    if make_noise == False:
                        if tr.stats.endtime < st_brd_orig[0].stats.starttime+origin_time+BRD_window[1]:
                            st_brd.remove(tr)
                    else:
                        if tr.stats.endtime < st_brd_orig[0].stats.starttime+origin_time_brd+BRD_window[1]:
                            st_brd.remove(tr)
                if len(st_brd) < 3:
                    print('Not enough BRD stations for array processing!')                        
                st_korea.append(st_brd)
                T_BRD, B_BRD, V_BRD, S_BRD = [], [], [], []
            if CHN == 'continue':
                # Remove traces with endtimes before window endtime
                for tr in st_chn:
                    if make_noise == False:
                        if tr.stats.endtime < st_chn_orig[0].stats.starttime+origin_time+CHN_window[1]:
                            st_chn.remove(tr)
                    else:
                        if tr.stats.endtime < st_chn_orig[0].stats.starttime+origin_time_chn+CHN_window[1]:
                            st_chn.remove(tr)  
                if len(st_chn) < 3:
                    print('Not enough CHN stations for array processing!')              
                st_korea.append(st_chn)
                T_CHN, B_CHN, V_CHN, S_CHN = [], [], [], []
            if KSG == 'continue':
                # Remove traces with endtimes before window endtime
                for tr in st_ksg:
                    if make_noise == False:
                        if tr.stats.endtime < st_ksg_orig[0].stats.starttime+origin_time+KSG_window[1]:
                            st_ksg.remove(tr)
                    else:
                        if tr.stats.endtime < st_ksg_orig[0].stats.starttime+origin_time_ksg+KSG_window[1]:
                            st_ksg.remove(tr)
                if len(st_ksg) < 3:
                    print('Not enough KSG stations for array processing!')
                st_korea.append(st_ksg)
                T_KSG, B_KSG, V_KSG, S_KSG = [], [], [], []
            #-----------------------------------------------------------------------------------------------------------------------#
            for st in st_korea:
                if len(st) < 3:
                    continue
                st_tmp = st.copy(); array_name = st_tmp[0].stats.station[:3]
                print('Begin array processing with ' + array_name)
                #-----------------------------------------------------------------------------------------------------------------------#
                # Define signal windows to be used for processing
                if array_name == 'BRD':
                    sig_start, sig_end = BRD_window
                elif array_name == 'CHN':
                    sig_start, sig_end = CHN_window
                elif array_name == 'KSG':
                    sig_start, sig_end = KSG_window
                #-----------------------------------------------------------------------------------------------------------------------#
                # Need to check if remaining stations have gaps in time-series
                stns_w_gaps = []  
                for gg_ix in range(len(st_tmp.get_gaps())):
                    stns_w_gaps.append(st_tmp.get_gaps()[gg_ix][1])
                stns_w_gaps = [*set(stns_w_gaps)] # removing duplicates
                for tr in st_tmp:
                    if tr.stats.station in stns_w_gaps:
                        st_tmp.remove(tr)
                # If no stations remain, move on to next array
                if len(st_tmp) < 3:
                    print('Not enough stations remain after removing gaps, moving on to next array.')
                    continue
                #-----------------------------------------------------------------------------------------------------------------------#
                # Run Cardinal
                try:
                    T, B, V, S = sliding_time_array_fk_multifreq(st_tmp, st_tmp[0].stats.station, f_bands, t_start=0, t_end=(sig_end-sig_start), n_workers=n_workers)
                except:
                    try: # fixing data gap issues that weren't caught by code
                        if array_name == 'KSG':
                            if ev_date == '2013/05/21':
                                for tr in st_tmp:
                                    if tr.stats.station == 'KSG20': st_tmp.remove(tr)
                                    elif tr.stats.station == 'KSG23': st_tmp.remove(tr)
                                    elif tr.stats.station == 'KSG24': st_tmp.remove(tr)
                                    elif tr.stats.station == 'KSG25': st_tmp.remove(tr)
                                    elif tr.stats.station == 'KSG10': st_tmp.remove(tr)
                                    elif tr.stats.station == 'KSG12': st_tmp.remove(tr)
                            else:
                                for tr in st_tmp:
                                    if tr.stats.station == 'KSG20': st_tmp.remove(tr)
                        elif array_name == 'CHN':
                            if ev_date == '2013/06/01':
                                for tr in st_tmp:
                                    if tr.stats.station == 'CHN03': st_tmp.remove(tr)
                                    elif tr.stats.station == 'CHN04': st_tmp.remove(tr)
                                    elif tr.stats.station == 'CHN05': st_tmp.remove(tr)
                            else:
                                for tr in st_tmp:
                                    if tr.stats.station == 'CHN10': st_tmp.remove(tr)
                                    elif tr.stats.station == 'CHN12': st_tmp.remove(tr)
                                    elif tr.stats.station == 'CHN20': st_tmp.remove(tr)
                                    elif tr.stats.station == 'CHN22': st_tmp.remove(tr)
                        elif array_name == 'BRD':
                            if ev_date == '2016/07/24':
                                for tr in st_tmp:
                                    if tr.stats.station == 'BRD10': st_tmp.remove(tr)
                                    elif tr.stats.station == 'BRD13': st_tmp.remove(tr)
                                    elif tr.stats.station == 'BRD22': st_tmp.remove(tr)
                                    elif tr.stats.station == 'BRD32': st_tmp.remove(tr)
                                    elif tr.stats.station == 'BRD33': st_tmp.remove(tr)
                                    elif tr.stats.station == 'BRD36': st_tmp.remove(tr)
                        T, B, V, S = sliding_time_array_fk_multifreq(st_tmp, st_tmp[0].stats.station, f_bands, t_start=0, t_end=(sig_end-sig_start), n_workers=n_workers)
                    except Exception as inst:
                        print(inst)
                        continue
                #-----------------------------------------------------------------------------------------------------------------------# 
                # Appending results based on array stream
                if array_name == 'BRD':
                    T_BRD.append(T); B_BRD.append(B); V_BRD.append(V); S_BRD.append(S)
                elif array_name == 'CHN':
                    T_CHN.append(T); B_CHN.append(B); V_CHN.append(V); S_CHN.append(S)                  
                elif array_name == 'KSG':
                    T_KSG.append(T); B_KSG.append(B); V_KSG.append(V); S_KSG.append(S)                   
            #-----------------------------------------------------------------------------------------------------------------------#
            # Saving BRD results
            if BRD == 'continue':
                try:
                    if make_noise == True:
                        np.save(outresultsdir+'/BRD/NOISE_T_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(T_BRD))
                        np.save(outresultsdir+'/BRD/NOISE_B_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(B_BRD))
                        np.save(outresultsdir+'/BRD/NOISE_V_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(V_BRD))
                        np.save(outresultsdir+'/BRD/NOISE_S_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(S_BRD))
                    else:
                        np.save(outresultsdir+'/BRD/T_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(T_BRD))
                        np.save(outresultsdir+'/BRD/B_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(B_BRD))
                        np.save(outresultsdir+'/BRD/V_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(V_BRD))
                        np.save(outresultsdir+'/BRD/S_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(S_BRD))
                except:
                    if outresultsdir == None:
                        raise Exception('outresultsdir needs to be specified if array_processing is set to True')
                    else:
                        raise Exception('Problem with saving BRD array processing results')    
            #-----------------------------------------------------------------------------------------------------------------------#
            # Saving CHN results                
            if CHN == 'continue':
                try:
                    if make_noise == True:
                        np.save(outresultsdir+'/CHN/NOISE_T_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(T_CHN))
                        np.save(outresultsdir+'/CHN/NOISE_B_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(B_CHN))
                        np.save(outresultsdir+'/CHN/NOISE_V_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(V_CHN))
                        np.save(outresultsdir+'/CHN/NOISE_S_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(S_CHN))
                    else:
                        np.save(outresultsdir+'/CHN/T_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(T_CHN))
                        np.save(outresultsdir+'/CHN/B_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(B_CHN))
                        np.save(outresultsdir+'/CHN/V_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(V_CHN))
                        np.save(outresultsdir+'/CHN/S_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(S_CHN))
                except:
                    if outresultsdir == None:
                        raise Exception('outresultsdir needs to be specified if array_processing is set to True')
                    else:
                        raise Exception('Problem with saving CHN array processing results')
            #-----------------------------------------------------------------------------------------------------------------------#
            # Saving KSG results
            if KSG == 'continue':
                try:
                    if make_noise == True:
                        np.save(outresultsdir+'/KSG/NOISE_T_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(T_KSG))
                        np.save(outresultsdir+'/KSG/NOISE_B_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(B_KSG))
                        np.save(outresultsdir+'/KSG/NOISE_V_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(V_KSG))
                        np.save(outresultsdir+'/KSG/NOISE_S_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(S_KSG))
                    else:
                        np.save(outresultsdir+'/KSG/T_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(T_KSG))
                        np.save(outresultsdir+'/KSG/B_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(B_KSG))
                        np.save(outresultsdir+'/KSG/V_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(V_KSG))
                        np.save(outresultsdir+'/KSG/S_'+str(year)+'_'+julian_day+'_'+ev_time.replace(':','-')+'.npy', np.array(S_KSG))             
                except:
                    if outresultsdir == None:
                        raise Exception('outresultsdir needs to be specified if array_processing is set to True')
                    else:
                        raise Exception('Problem with saving KSG array processing results')

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############## Plotting Functions #############
############### ############### ###############

def plot_hist(mags, dists, plot_array_counts=False, array_counts=None, bins_mags=None, bins_dists=None, mags_xlim=None, dists_xlim=None, return_bins=False, normalize_bar=False, title=None, legend_loc='upper right', figsize=(9,5)):
    '''---------------------------------------------------------------------------
    Generates histograms for event magnitudes and source-receiver distances.
    ---------------------------------------------------------------------------'''
    # Make figure
    fig = plt.figure(figsize=figsize)
    labels = ['BRD', 'CHN', 'KSG']
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plotting magnitudes
    if plot_array_counts == True:
        fig.add_subplot(1,3,1)
    else:
        fig.add_subplot(1,2,1)
    if bins_mags == None:
        bins_mags = []
        for array_mags in mags:
            bins_mags.append(Freedman_Diaconis(array_mags))
    for idx, array_mags in enumerate(mags):
        if len(mags) > 1:
            sns.distplot(array_mags, hist=True, kde=False, bins=int(bins_mags[idx][0]), hist_kws={'edgecolor':'black'}, label=labels[idx])
            plt.legend(loc=legend_loc)
        else:
            sns.distplot(array_mags, hist=True, kde=False, bins=int(bins_mags[idx][0]), hist_kws={'edgecolor':'black'}, color='red')
    if mags_xlim == None:
        plt.xlim(0,np.concatenate(mags).max()+bins_mags[idx][1])
    else:
        plt.xlim(mags_xlim)
    plt.grid(lw=0.25)
    plt.xlabel('Local Magnitude'); plt.ylabel('Counts'); plt.title('Magnitudes')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plotting distances
    if plot_array_counts == True:
        fig.add_subplot(1,3,2)
    else:
        fig.add_subplot(1,2,2)    
    if bins_dists == None:
        bins_dists = []
        for array_dists in dists:
            bins_dists.append(Freedman_Diaconis(array_dists))
    for idx, array_dists in enumerate(dists):
        if len(dists) > 1:
            sns.distplot(array_dists, hist=True, kde=False, bins=int(bins_dists[idx][0]), hist_kws={'edgecolor':'black'}, label=labels[idx])
            plt.legend(loc=legend_loc)
        else:
            sns.distplot(array_dists, hist=True, kde=False, bins=int(bins_dists[idx][0]), hist_kws={'edgecolor':'black'}, color='k')
    if dists_xlim == None:
        plt.xlim(0,np.concatenate(dists).max()+bins_dists[idx][1])
    else:
        plt.xlim(dists_xlim)
    plt.grid(lw=0.25)
    plt.xlabel('Distance (km)'); plt.title('Source-Receiver Distances')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plotting array detections
    if plot_array_counts == True:
        fig.add_subplot(1,3,3)
        if normalize_bar:
            array_counts = array_counts / array_counts.max()
        plt.bar(labels, array_counts, color='grey'); plt.xlabel('Array'); plt.title('Array Detections')
        plt.grid(lw=0.25); plt.xlim([-0.5,2.5])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Title and return params
    if title is not None:
        plt.suptitle(title)
    if return_bins == True:
        return bins_mags, bins_dists

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def generate_map(earthquake_lats, earthquake_lons, explosion_lats, explosion_lons, eq_arrays=None, exp_arrays=None, hide_eqs=False, hide_exps=False, extent=[122.5, 131, 42, 32.5], projection=ccrs.PlateCarree(), transform=ccrs.Geodetic(), gl_lw=0.5, gl_ls='--', 
                 add_axes_labels=True, draw_labels=True, legend_loc='lower left', title=None, title_size=10.5, markersize=5, array_markersize=8, legend_fontsize=7.5, lw=0.15, markerscale=2,
                 figsize=(9,6)):
    '''---------------------------------------------------------------------------
    This function generates a map of the surface explosions and earthquakes used
    for analysis within the Korean peninsula.
    ---------------------------------------------------------------------------'''
    # Define the figure
    fig, axs = plt.subplots(nrows=1,ncols=1,
                            subplot_kw={'projection': projection},
                            figsize=figsize)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plotting earthquakes
    if hide_eqs: alpha_eq = 0; label_eq = None
    else: alpha_eq = 1; label_eq = 'Earthquake'
    axs.plot(earthquake_lons, earthquake_lats, '*', color='blue', transform=transform, markersize=markersize, alpha=alpha_eq, label=label_eq)
    if eq_arrays is not None:
        # Plotting lines connecting earthquake to array 
        for idx, array in enumerate(eq_arrays):
            if array == 'BRD':
                plt.plot([earthquake_lons[idx], BRD_coords[1]], [earthquake_lats[idx], BRD_coords[0]], color='blue', lw=lw, ls='--', transform=transform)
            elif array == 'CHN':
                plt.plot([earthquake_lons[idx], CHN_coords[1]], [earthquake_lats[idx], CHN_coords[0]], color='blue', lw=lw, ls='--', transform=transform)
            elif array == 'KSG':
                plt.plot([earthquake_lons[idx], KSG_coords[1]], [earthquake_lats[idx], KSG_coords[0]], color='blue', lw=lw, ls='--', transform=transform)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plotting surface explosions
    if hide_exps: alpha_exp = 0; label_exp = None
    else: alpha_exp = 1; label_exp = 'Surface Explosion'
    axs.plot(explosion_lons, explosion_lats, '*', color='red', transform=transform, markersize=markersize, alpha=alpha_exp, label=label_exp)
    if exp_arrays is not None:
        # Plotting lines connecting explosion to array 
        for idx, array in enumerate(exp_arrays):
            if array == 'BRD':
                plt.plot([explosion_lons[idx], BRD_coords[1]], [explosion_lats[idx], BRD_coords[0]], color='red', lw=lw, ls='--', transform=transform)
            elif array == 'CHN':
                plt.plot([explosion_lons[idx], CHN_coords[1]], [explosion_lats[idx], CHN_coords[0]], color='red', lw=lw, ls='--', transform=transform)
            elif array == 'KSG':
                plt.plot([explosion_lons[idx], KSG_coords[1]], [explosion_lats[idx], KSG_coords[0]], color='red', lw=lw, ls='--', transform=transform)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plotting arrays
    axs.plot([BRD_coords[1], CHN_coords[1], KSG_coords[1]],
             [BRD_coords[0], CHN_coords[0], KSG_coords[0]], ls='none', marker='^', mec='k', mfc='black', markersize=array_markersize, markeredgewidth=1, label='Array')
    #-----------------------------------------------------------------------------------------------------------------------#
    axs.set_extent(extent, crs=transform)
    if add_axes_labels == True:
        gl = axs.gridlines(crs=projection, draw_labels=draw_labels, linewidth=0)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.top_labels = False
        gl.right_labels = False
    axs.add_feature(cartopy.feature.LAKES, alpha=0.5)  
    axs.add_feature(cartopy.feature.RIVERS, alpha=0.5)
    axs.add_feature(cartopy.feature.OCEAN, alpha=0.5, color='grey')
    axs.add_feature(cartopy.feature.BORDERS, alpha=0.2) 
    axs.add_feature(cartopy.feature.LAND, alpha=0.2, color='white') 
    axs.add_feature(cartopy.feature.COASTLINE)
    axs.gridlines(lw=gl_lw, ls=gl_ls)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize, markerscale=markerscale)
    if title == None:
        axs.set_title('Explosions and Earthquakes within the Korean Peninsula', size=title_size)
    else:
        axs.set_title(title, size=title_size)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_seismoacoustic_array_coords(st, st_infra, ref_station, infra_ref_idx=0, plot=True, kilometers=False, array=None, plot_stnm=False, legend_loc='upper right', xlim=None, ylim=None,
                                     title_fontsize=10, legend_fontsize=10, markerscale=2, figsize=(6,6)):
    '''---------------------------------------------------------------------------
    Plots Korean seismoacoustic array coordinates.
    ---------------------------------------------------------------------------'''
    # Computing seismic array coordinates
    X = np.zeros((len(st), 2))
    stnm = []
    for i in range(0, len(st)):
        E, N, _, _ = utm.from_latlon(st[i].stats.sac.stla, st[i].stats.sac.stlo)
        X[i,0] = E; X[i,1] = N
        stnm.append(st[i].stats.station)
    ref_station_ix = np.where(np.array(stnm) == ref_station)[0][0]
    X[:,0] = (X[:,0] - X[ref_station_ix,0])
    X[:,1] = (X[:,1] - X[ref_station_ix,1])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Computing infrasound array coordinates
    X_infra = np.zeros((len(st_infra), 2))
    stnm_infra = []
    for i in range(0, len(st_infra)):
        E, N, _, _ = utm.from_latlon(st_infra[i].stats.sac.stla, st_infra[i].stats.sac.stlo)
        X_infra[i,0] = E; X_infra[i,1] = N
        stnm_infra.append(st_infra[i].stats.station)
    X_infra[:,0] = (X_infra[:,0] - X_infra[infra_ref_idx,0])
    X_infra[:,1] = (X_infra[:,1] - X_infra[infra_ref_idx,1])
    
    if kilometers ==True:
        ref_station_ix = np.where(np.array(stnm) == ref_station)[0][0]
        X[:,0] = (X[:,0] - X[ref_station_ix,0])/1000
        X[:,1] = (X[:,1] - X[ref_station_ix,1])/1000
        X_infra[:,0] = (X_infra[:,0] - X_infra[infra_ref_idx,0])/1000
        X_infra[:,1] = (X_infra[:,1] - X_infra[infra_ref_idx,1])/1000
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plotting
    if plot==True:
        fig, ax = plt.subplots(figsize=figsize)

        x, y, arrow_length = 0.05, 0.30, 0.1
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=ax.transAxes)

        plt.plot(X[:,0], X[:,1], '+', color='blue', ms=15)
        plt.plot(X_infra[:,0], X_infra[:,1], 'o', color='black', mfc='white')
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if plot_stnm == True:
            for i in range(0,len(stnm)):
                plt.text(X[i,0], X[i,1], stnm[i])
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        if kilometers == True:
            plt.xlabel('X (km)')
            plt.ylabel('Y (km)')
        plt.title('Array Coordinates')
        if array is not None:
            plt.title(array, fontsize=title_fontsize)
        plt.grid(linewidth=0.1)
        plt.legend(['Seismic', 'Infrasoud'], loc=legend_loc, fontsize=legend_fontsize, markerscale=markerscale)
    #-----------------------------------------------------------------------------------------------------------------------#
    return X, stnm

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_seismic_modality(st, array, channel, year, day, event_type, ev_time, dist, mag, savefig=False, outfigdir=None, figsize=(15,5), xlim=[0,100], 
                          spec_range=[0.9375,15], nperseg=2**7, noverlap=None):
    '''---------------------------------------------------------------------------
    Plots seismic waveforms associated with one channel of input vector.
    ---------------------------------------------------------------------------'''
    st = st.select(station=array+'*', channel='*'+channel)
    try:
        st = st.merge()
    except:
        st = st.resample(int(st[0].stats.sampling_rate))
        st = st.merge()
    #-----------------------------------------------------------------------------------------------------------------------#
    fig = plt.figure(figsize=figsize)
    t = np.arange(0, st[0].stats.delta*st[0].stats.npts, st[0].stats.delta)
    cmap = cm.get_cmap('viridis_r', 256)
    cmap = cmap(np.linspace(2,0.001,100))
    cmap = ListedColormap(cmap)
    for i, tr in enumerate(st[::-1]):
        ax_tmp1 = fig.add_subplot(len(st),2,(i*2)+1)
        tr.data /= np.abs(tr.data.max()) # Normalizing waveforms
        try:
            plt.plot(t, tr.data, color='k', label=tr.stats.station)
        except:
            if len(t) > len(tr.data):
                plt.plot(t[:len(tr.data)], tr.data, color='k', label=tr.stats.station)
            elif len(t) < len(tr.data):
                plt.plot(t, tr.data[:len(t)], color='k', label=tr.stats.station)
        plt.legend(loc='upper right')
        #-----------------------------------------------------------------------------------------------------------------------#
        ax_tmp2 = fig.add_subplot(len(st),2,(i*2)+2, sharex=ax_tmp1)
        if noverlap == None:
            f, t_f, Sxx = signal.spectrogram(tr.data, tr.stats.sampling_rate, nperseg=nperseg)
        else:
            f, t_f, Sxx = signal.spectrogram(tr.data, tr.stats.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        # Keep only freqs of interest
        freq_slice = np.where((f >= spec_range[0]) & (f <= spec_range[1]))
        f = f[freq_slice]
        Sxx = Sxx[freq_slice,:][0]
        Sxx /= np.abs(Sxx.max()) # Normalizing spectrogram
        plt.pcolormesh(t_f, f, Sxx, cmap=cmap, shading='gouraud', norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()))
        #-----------------------------------------------------------------------------------------------------------------------#
        if i == len(st)-1:
            ax_tmp1.set_xlabel('Time (s)'); ax_tmp1.set_ylabel('Normalized\nAmplitude')
            ax_tmp2.set_xlabel('Time (s)'); ax_tmp2.set_ylabel('Frequency\n(Hz)')
        else:
            ax_tmp1.tick_params(labelbottom=False)
            ax_tmp2.tick_params(labelbottom=False)
        ax_tmp1.set_xlim(xlim); ax_tmp1.set_ylim([-1,1])
        ax_tmp2.set_yscale('log'); ax_tmp2.set_ylim(spec_range)
    plt.suptitle(array + ' - ' + channel + '\n '+ event_type + ' - ' + year + ' - ' + day + ' - ' + dist + ' km - ' + mag + ' ML')
    #-----------------------------------------------------------------------------------------------------------------------#
    if savefig == True:
        try:
            if event_type == 'Noise':
                plt.savefig(outfigdir+'/'+array+'/NOISE_'+year+'_'+day+'_'+ev_time.replace(':','-')+'_'+channel)
            else:
                plt.savefig(outfigdir+'/'+array+'/'+year+'_'+day+'_'+ev_time.replace(':','-')+'_'+channel)

        except Exception as inst:
            if outfigdir == None:
                raise Exception('outfigdir must be specified if savefig is set to True.')
            else:
                print(inst)
        plt.close()
        
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_infrasound_modality(st, element, f_bands, T, B, V, S, title=None, bandpass=None,
                             semblance_threshold=0, clim_baz=None, clim_vtr=[0,1],
                             log_freq=False, cmap_cyclic='twilight', cmap_sequential='pink_r',
                             twin_plot=None, f_lim=None, plot_real_amplitude=True, amplitude_units='Pa',
                             ix=None, pixels_in_families=None, best_beam=False, figsize=(9,5), beamform_freq_idx=0,
                             normalize_beam=False, timeseries_ylim=None, GT_baz=None, delay_times=None, fname_plot=None, legend_loc='upper right'):
    '''---------------------------------------------------------------------------
    Plots infrasound array processing results.
    ---------------------------------------------------------------------------'''
    S_filt = S.copy()
    tr = st.select(station=element)[0].copy()
    #-----------------------------------------------------------------------------------------------------------------#
    # For plotting families
    if (pixels_in_families is not None) and (ix is not None):
        # Set all semblances to zero where the pixel is not in a family:
        x = np.zeros(S.shape)
        x[ix[0][pixels_in_families],ix[1][pixels_in_families]] = 1   # Makes a mask where 1 means plot value
        S_filt[x == 0] = 0
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting waveforms
    start_time_string = str(tr.stats.starttime).split('.')[0].replace('T',' ')
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(4,1,1)
    if best_beam == False:
        t_tr = np.arange(0, tr.stats.npts*tr.stats.delta, tr.stats.delta)
        if title is not None:
            plt.title(title)    
        if (bandpass is None):
            pass
        else:
            tr.taper(type='cosine', max_percentage=0.05, max_length=60)
            try:
                tr.filter('bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
            except:
                tr = tr.split()
                tr.filter('bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
                tr = tr.merge()[0]
        t_tr, y = fix_lengths(t_tr, tr.data)
        if plot_real_amplitude:
            plt.plot(t_tr, y, 'k-', label=tr.stats.station)
            plt.ylabel(amplitude_units)
            if timeseries_ylim is not None:
                plt.ylim(timeseries_ylim)
            plt.legend(loc=legend_loc)
        else:
            plt.plot(t_tr, y/np.max(np.abs(y)), 'k-', label=tr.stats.station)
            ax1.tick_params(labelleft=False)
            plt.ylim([-1,1])
            plt.legend(loc=legend_loc)
    else:
        if (bandpass is None):
            pass
        else:
            st_filt = st.copy()
            st_filt.filter('bandpass', freqmin=min(bandpass), freqmax=max(bandpass))
        t_shifts = get_slowness_vector_time_shifts(st_filt, element, baz=B[beamform_freq_idx,:][np.argmax(S[beamform_freq_idx,:])],
                                                                    tr_vel=V[beamform_freq_idx,:][np.argmax(S[beamform_freq_idx,:])])
        t_beam, beam = beamform(t_shifts, st_filt, element, normalize_beam=normalize_beam)
        ax1.plot(t_beam, beam, color='red', label='Best Beam')
        if title is not None:
            plt.title(title)    
        if normalize_beam == False:
            plt.ylabel('Pressure (Pa)')
        else:
            plt.ylim([-1,1])
        if timeseries_ylim is not None:
            plt.ylim(timeseries_ylim)
        plt.legend(loc='upper right')
    ax1.tick_params(labelbottom=False)
    if delay_times is not None:
        plt.axvline(x=delay_times[0], lw=0.5, color='red'); plt.axvline(x=delay_times[1], lw=1, color='green'); plt.axvline(x=delay_times[2], lw=0.5, color='blue')
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting back azimuths
    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2.tick_params(labelbottom=False)
    ix_S = np.where(S_filt < semblance_threshold)
    if GT_baz is not None:
        B_dev = B.copy()
        B_dev -= GT_baz
        for B_row in range(len(B_dev[:,0])):
            for B_col in range(len(B_dev[0,:])):
                # Mitigate angular wrapping
                if B_dev[B_row, B_col] < -270:
                    B_dev[B_row, B_col] += 360
                elif B_dev[B_row, B_col] > 270:
                    B_dev[B_row, B_col] -= 360
                else:
                    pass
        B_plt = B_dev.copy()
        if clim_baz is not None:
            ix_low = np.where(B_plt < clim_baz[0])
            ix_high = np.where(B_plt > clim_baz[1])
            B_plt[ix_low] = None; B_plt[ix_high] = None
        else:
            pass
    else:
        B_plt = B.copy()
    if (pixels_in_families is not None) and (ix is not None):
        B_plt[ix] = None
    else:
        B_plt[ix_S] = None
    t_plot = np.hstack((T,T[len(T)-1]+np.diff(T)[0]))
    f_plot = np.hstack((f_bands['fmin'].values, f_bands['fmax'].values[len(f_bands['fmax'])-1]))
    pcm1 = plt.pcolor(t_plot, f_plot, B_plt, cmap=plt.get_cmap(cmap_cyclic), shading='flat')
    if clim_baz is not None and clim_baz[0]<clim_baz[1]:
        plt.clim([clim_baz[0], clim_baz[1]])
    elif clim_baz is not None and clim_baz[0]>clim_baz[1]:
        plt.clim([clim_baz[0], clim_baz[1]+360])
    plt.ylabel('Freq. (Hz)')
    if log_freq:
        plt.yscale('log')
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting trace velocity
    ax3 = plt.subplot(4,1,3, sharex=ax1, sharey=ax2)
    ax3.tick_params(labelbottom=False)
    V_plt = V.copy()
    if (pixels_in_families is not None) and (ix is not None):
        V_plt[ix] = None
    else:
        V_plt[ix_S] = None
    if GT_baz is not None:
        V_plt[ix_low] = None; V_plt[ix_high] = None
    pcm2 = plt.pcolor(t_plot, f_plot, V_plt, cmap=plt.get_cmap(cmap_sequential), shading='flat')
    if clim_vtr is not None:
        plt.clim([clim_vtr[0], clim_vtr[1]])
    plt.ylabel('Freq. (Hz)')
    if log_freq:
        plt.yscale('log')
    #-----------------------------------------------------------------------------------------------------------------#
    # Plotting semblance
    ax4 = plt.subplot(4,1,4, sharex=ax1, sharey=ax2)
    pcm3 = plt.pcolor(t_plot, f_plot, S, cmap=plt.get_cmap(cmap_sequential), shading='flat')
    plt.clim([0,1])
    plt.ylabel('Freq. (Hz)')
    plt.xlabel('Time (s) after ' + start_time_string)
    if log_freq:
        plt.yscale('log')
    if twin_plot is not None:
        plt.xlim(twin_plot)
    else:
        plt.xlim([t_tr[0], t_tr[len(t_tr)-1]])
    if f_lim is not None:
        plt.ylim(f_lim)
    #-----------------------------------------------------------------------------------------------------------------#
    # Manually adding colorbars
    # Colorbar for back azimuth
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.8575, 0.5175, 0.0200, 0.1550])
    fig.colorbar(pcm1, cax=cbar_ax)
    if GT_baz is not None:
        cbar_ax.set_ylabel('Azimuth\nDeviation (\N{DEGREE SIGN})')
    else:
        cbar_ax.set_ylabel('Azimuth (\N{DEGREE SIGN})')
    cbar_ax.locator_params(nbins=6)
    if clim_baz is not None:
        if clim_baz[0]>clim_baz[1]:
            tick_labels= np.array(cbar_ax.get_yticks())
            tick_labels= tick_labels.astype(int)
            new_labels=(np.where((tick_labels > 360), tick_labels-360, tick_labels))
            cbar_ax.set_yticklabels(new_labels)
    # Colorbar for trace velocity
    cbar_ax = fig.add_axes([0.8575, 0.3175, 0.0200, 0.1550])
    fig.colorbar(pcm2, cax=cbar_ax)
    cbar_ax.locator_params(nbins=4)
    cbar_ax.set_ylabel('Velocity\n(km/s)')
    # Colorbar for semblance
    cbar_ax = fig.add_axes([0.8575, 0.1175, 0.0200, 0.1550])
    fig.colorbar(pcm3, cax=cbar_ax)
    cbar_ax.locator_params(nbins=4)
    cbar_ax.set_ylabel('Semblance')
    #-----------------------------------------------------------------------------------------------------------------#
    if fname_plot is not None:
        plt.savefig(fname_plot)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def TrainTest_seismic_plots(ev_metadata, ev_data, ev_stn_labels, outfigdir, samp_rate=40, win_len=90, spectrograms_shape=(90,90,3), train=True, figsize=(12,6)):
    '''---------------------------------------------------------------------------
    Plots and saves train/test seismic waveforms and spectrograms.
    ---------------------------------------------------------------------------'''
    for idx in range(ev_metadata.shape[0]):
        # Indexing
        event_type=ev_metadata[idx][0]; array=ev_metadata[idx][1]
        mag = ev_metadata[idx][5]; dist = ev_metadata[idx][6]
        lat = ev_metadata[idx][3]; lon = ev_metadata[idx][4]
        #-----------------------------------------------------------------------------------------------------------------------#
        # Data
        nsamp = samp_rate*win_len
        data_tmp = ev_data[idx,:,:,:]; t = np.arange(0, (1/samp_rate)*int(nsamp), (1/samp_rate))
        data_tmp = data_tmp[:,0:int(nsamp),:].copy()
        # Normalizing by trace
        for stn in range(data_tmp.shape[0]):
            for chn in range(data_tmp.shape[2]):
                data_tmp[stn,:,chn] = data_tmp[stn,:,chn] / np.abs(data_tmp[stn,:,chn].max())
        #-----------------------------------------------------------------------------------------------------------------------#
        # Generating spectrograms
        spec_tmp = np.zeros(spectrograms_shape); t_f = []
        for chn in range(data_tmp.shape[2]):
            spec_tmp_chn = []; t_f_tmp_chn = []
            for stn in range(data_tmp.shape[0]):
                f, t_f_idx, Sxx_tmp = signal.spectrogram(data_tmp[stn,:,chn], samp_rate, nperseg=2**7, noverlap=(2**7)*0.70)
                freq_slice = np.where((f >= 0.9375) & (f <= 10)); f = f[freq_slice]; Sxx_tmp = Sxx_tmp[freq_slice,:][0]
                Sxx_tmp /= np.abs(Sxx_tmp.max()) # normalize by station
                spec_tmp_chn.append(Sxx_tmp); t_f_tmp_chn.append(t_f_idx)
            spec_tmp_chn = np.vstack((np.array(spec_tmp_chn))) # stacked spectrograms for each station within channel
            spec_tmp[:,:,chn] = spec_tmp_chn
            t_f.append(t_f_tmp_chn)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Plotting
        fig = plt.figure(figsize=figsize)
        for channel_plot in range(3):
            ax_wavf = fig.add_subplot(3,2,(channel_plot*2)+1)
            k = 0
            for i in range(ev_data.shape[1]):
                ax_wavf.plot(t, data_tmp[i,:,channel_plot]+k, color='k')
                k += 2
            plt.xlim([0,win_len])
            if channel_plot != 2:
                ax_wavf.tick_params(labelbottom=False); ax_wavf.set_xticks([])
                if channel_plot == 0: plt.title('E Component')
                else: plt.title('N Component')
            else: 
                plt.xlabel('Time (s)'); plt.title('Z Component')
            #ax_wavf.tick_params(labelleft=False); #ax_wavf.set_yticks([])
            labels = [item.get_text() for item in ax_wavf.get_yticklabels()]
            for ii, stn_label in enumerate(ev_stn_labels[idx]):
                labels[ii+1] = stn_label
            ax_wavf.set_yticklabels(labels)
            plt.xlim([0,win_len])
            #-----------------------------------------------------------------------------------------------------------------------#
            ax_spec = fig.add_subplot(3,2,(channel_plot*2)+2, sharex=ax_wavf)
            plt.pcolormesh(t_f[0][0], np.arange(0,win_len,1), spec_tmp[:,:,channel_plot], cmap='viridis', shading='gouraud',  norm=colors.LogNorm(vmin=spec_tmp[:,:,channel_plot].min(), vmax=spec_tmp[:,:,channel_plot].max()))
            if channel_plot != 2:
                ax_spec.tick_params(labelbottom=False); ax_spec.set_xticks([])
                if channel_plot == 0: plt.title('E Component')
                else: plt.title('N Component')
            else: 
                plt.xlabel('Time (s)'); plt.title('Z Component')
            ax_spec.tick_params(labelleft=False);ax_spec.set_yticks([])
            plt.xlim([0,win_len])
        plt.suptitle(event_type+' - '+array+'\n'+mag+' ML - '+dist+' km - '+str(lat)+'\N{DEGREE SIGN}N - '+str(lon)+'\N{DEGREE SIGN}E')
        #-----------------------------------------------------------------------------------------------------------------------#
        # Saving figure
        if train == True:
            plt.savefig(outfigdir+'Train/'+event_type+'s/'+ev_metadata[idx][2].replace(':','-').replace('/','_').replace(' ','_')+'_'+array+'.png')
        elif train == False:
            plt.savefig(outfigdir+'Test/'+event_type+'s/'+ev_metadata[idx][2].replace(':','-').replace('/','_').replace(' ','_')+'_'+array+'.png')
        elif train == None:
            plt.savefig(outfigdir+'/'+event_type+'s/'+ev_metadata[idx][2].replace(':','-').replace('/','_').replace(' ','_')+'_'+array+'.png')

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def TrainTest_infra_plots(ev_metadata, outfigdir, npts=360, freqmin=0.5, freqmax=10, local_infrasound=False, train=True, baz_dev=90):
    '''---------------------------------------------------------------------------
    Plots and saves train/test epicentral and local infrasound Cardinal results.
    ---------------------------------------------------------------------------'''
    # Constructing frequency bands
    f_bands = make_custom_fbands(f_min=freqmin, f_max=freqmax+1, type='third_octave')
    f_bands['fmax'][-1:] = np.floor(f_bands['fmax'][-1:]) # round
    #-----------------------------------------------------------------------------------------------------------------------#
    for idx in range(ev_metadata.shape[0]):
        # Event metadata
        event_type=ev_metadata[idx][0]; array=ev_metadata[idx][1]
        lat = ev_metadata[idx][3]; lon = ev_metadata[idx][4]
        mag = ev_metadata[idx][5]; dist = ev_metadata[idx][6]
        #-----------------------------------------------------------------------------------------------------------------------#
        # Calculate GT back azimuth
        if array == 'BRD':
            _, GT_baz, _ = g.inv(lon, lat, BRD_coords[1], BRD_coords[0])
        if array == 'CHN':
            _, GT_baz, _ = g.inv(lon, lat, CHN_coords[1], CHN_coords[0])
        if array == 'KSG':
            _, GT_baz, _ = g.inv(lon, lat, KSG_coords[1], KSG_coords[0])
        #-----------------------------------------------------------------------------------------------------------------------#
        # Loading waveforms and array processing results
        if local_infrasound:
            directory_wavf = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Local_Infrasound/Waveforms/'+array
            directory_cardinal = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Local_Infrasound/Cardinal/'+array
        else:
            directory_wavf = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Infrasound/Waveforms/'+array
            directory_cardinal = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Infrasound/Cardinal/'+array
        try:
            st = read(directory_wavf+'/'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.mseed')
            B = np.load(directory_cardinal+'/B_'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
            V = np.load(directory_cardinal+'/V_'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
            S = np.load(directory_cardinal+'/S_'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
            T = np.load(directory_cardinal+'/T_'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]; T = T.reshape(1,len(T))
        except:
            directory_wavf = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Infrasound/Han_2023/Waveforms/'+array
            directory_cardinal = '/Volumes/Extreme SSD/Korea_Events/'+event_type+'s/Infrasound/Han_2023/Cardinal/'+array
            st = read(directory_wavf+'/'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.mseed')
            B = np.load(directory_cardinal+'/B_'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
            V = np.load(directory_cardinal+'/V_'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
            S = np.load(directory_cardinal+'/S_'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]
            T = np.load(directory_cardinal+'/T_'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'.npy')[0]; T = T.reshape(1,len(T))
        #-----------------------------------------------------------------------------------------------------------------------#
        # Bilinear resize
        B = bilinear_resize(B, new_h=B.shape[0], new_w=npts)
        V = bilinear_resize(V, new_h=V.shape[0], new_w=npts)
        S = bilinear_resize(S, new_h=S.shape[0], new_w=npts)
        T = bilinear_resize(T, new_h=T.shape[0], new_w=npts)[0,:]
        #-----------------------------------------------------------------------------------------------------------------------#
        # Plotting and saving figure
        if train == True:
            fname = outfigdir + 'Train/'+event_type+'s/'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'_'+array+'.png'
        elif train == False:
            fname = outfigdir + 'Test/'+event_type+'s/'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'_'+array+'.png'
        elif train == None:
            fname = outfigdir + '/'+event_type+'s/'+ev_metadata[idx][2].replace('/','_').replace(' ','_').replace(':','-')+'_'+array+'.png'
        plot_infrasound_modality(st, st[0].stats.station, f_bands, T, B, V, S, log_freq=True, bandpass=[0.5,5], plot_real_amplitude=False, GT_baz=GT_baz, clim_baz=[-baz_dev,baz_dev], clim_vtr=[0.2,0.45],
                                     title=event_type+' - '+array+'\n'+str(mag)+'ML - '+str(dist)+' km - '+str(lat)+'\N{DEGREE SIGN}N - '+str(lon)+'\N{DEGREE SIGN}E', fname_plot=fname)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_model_results(history, accuracy_ylim=None, loss_ylim=None, title=None, eval_metric='f1', plot_lr=False, figsize=(12,4)):
    '''---------------------------------------------------------------------------
    Plots neural network training results.
    ---------------------------------------------------------------------------'''
    # Combine all the history from training together
    try:
        combined = dict()
        if plot_lr == True:
            for key in [eval_metric,'val_'+eval_metric,'loss','val_loss','lr']:
                combined[key] = np.hstack([x.history[key] for x in history])
        else:
            for key in [eval_metric,'val_'+eval_metric,'loss','val_loss']:
                combined[key] = np.hstack([x.history[key] for x in history])
    except:
        combined = history.copy()
    #-----------------------------------------------------------------------------------------------------------------------#
    # Summarize history for accuracy
    plt.figure(figsize=figsize)
    if plot_lr == True:
        plt.subplot(1,3,1)
    else:
        plt.subplot(1,2,1)
    plt.plot(combined[eval_metric], color='black')
    plt.plot(combined['val_'+eval_metric], color='red')
    plt.xlabel('Epoch'); plt.ylabel(eval_metric)
    plt.grid(lw=0.25); plt.title('Score')
    plt.legend(['Train', 'Test'], loc='upper left', prop = { "size": 8})
    if accuracy_ylim is not None:
        plt.ylim(accuracy_ylim)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Summarize history for loss
    if plot_lr == True:
        plt.subplot(1,3,2)
    else:
        plt.subplot(1,2,2)
    plt.plot(combined['loss'], color='black')
    plt.plot(combined['val_loss'], color='red')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(lw=0.25); plt.title('Loss')
    plt.legend(['Train', 'Test'], loc='upper right', prop = { "size": 8})
    if loss_ylim is not None:
        plt.ylim(loss_ylim)
    #-----------------------------------------------------------------------------------------------------------------------#
    if plot_lr == True:
        # Summarize history for learning rate
        plt.subplot(1,3,3)
        plt.semilogy(combined['lr'], color='black')
        plt.xlabel('Epoch'); plt.title('Learning Rate')
        plt.grid(lw=0.25)
    #-----------------------------------------------------------------------------------------------------------------------#
    if title is not None:
        plt.suptitle(title)
    plt.show()

'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------'
############### ############### ###############
############# Statistical Methods #############
############### ############### ###############

def Freedman_Diaconis(data):
    '''---------------------------------------------------------------------------
    This function calculates bin width and number of bins used for histogram plotting
    using the Freedman-Diaconis method.
    ---------------------------------------------------------------------------'''
    # Calculate using interquartile range
    Q1 = np.quantile(data, 0.25)
    Q3 = np.quantile(data, 0.75)
    IQR = Q3 - Q1
    cube = np.cbrt(len(data))
    bin_width = (2*IQR/cube)
    num_bins = (data.max() - data.min()) / bin_width
    #-----------------------------------------------------------------------------------------------------------------------#
    return num_bins, bin_width

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def confidence_interval(err1, err2, X_test):
    '''---------------------------------------------------------------------------
    This function calculates the confidence interval given error rates of 2 models
    and associated test set.
    ---------------------------------------------------------------------------'''
    d = np.abs(err1 - err2)
    var = (err2 * (1-err2)/X_test.shape[0]) + (err1 * (1-err1)/X_test.shape[0])
    #-----------------------------------------------------------------------------------------------------------------------#
    return (d-1.96*np.sqrt(var), d+1.96*np.sqrt(var))

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def contingency_value(row):
    '''---------------------------------------------------------------------------
    Helper function for contingency_table.
    ---------------------------------------------------------------------------'''
    if row['Net1'] and row['Net2']:
        return 'A'
    elif row['Net1']:
        return 'B'
    elif row['Net2']:
        return 'C'
    else:
        return 'D'

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def contingency_table(y_true, X_test_multimodal, X_test, net1, net2):
    '''---------------------------------------------------------------------------
    Returns contingency table needed for McNemar test.
    ---------------------------------------------------------------------------'''
    yhat1 = np.round(net1.predict(X_test_multimodal)).flatten()
    yhat2 = np.round(net2.predict(X_test)).flatten()
    truth_table = pd.DataFrame({'Target': y_true.flatten(),
                                'Net1': yhat1,
                                'Net2': yhat2},
                                dtype=bool)
    truth_table.loc[truth_table['Target'] == False] = ~truth_table
    contingency_values = truth_table.apply(contingency_value, axis=1)
    #-----------------------------------------------------------------------------------------------------------------------#
    A = len(contingency_values.loc[contingency_values == 'A'])
    B = len(contingency_values.loc[contingency_values == 'B'])
    C = len(contingency_values.loc[contingency_values == 'C'])
    D = len(contingency_values.loc[contingency_values == 'D'])
    #-----------------------------------------------------------------------------------------------------------------------#
    return np.array([[A, B],
                     [C, D]])

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_p_value(y_true, X_test_multimodal, X_test, net1, net2):
    '''---------------------------------------------------------------------------
    Returns P-value correspondig to McNemar test.
    ---------------------------------------------------------------------------'''
    c_table = contingency_table(y_true, X_test_multimodal, X_test, net1, net2)
    #-----------------------------------------------------------------------------------------------------------------------#
    return mcnemar(c_table, exact=False, correction=False).pvalue

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def statistical_significance(alpha, y_true, X_test_multimodal, X_test, net1, net2, net1_title, net2_title):
    '''---------------------------------------------------------------------------
    Assesses statistical significance of model results.
    ---------------------------------------------------------------------------'''
    p_val = get_p_value(y_true, X_test_multimodal, X_test, net1, net2)
    if p_val >= alpha:
        print('There is no significant difference between ' + net1_title + ' and '+ net2_title + '. Results are not statistically significant')
    else:
        print('There is a significant difference between ' + net1_title + ' and ' + net2_title + '. Results are statistically significant')

'----------------------------------------------'
'---------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

############### ############### ###############
############### Neural Networks ###############
############### ############### ###############

def make_seismic_model(X_train_spec, lr=1e-3, loss='binary_crossentropy', eval_metric=None):
    
    # Define optimizer
    adam = Adam(learning_rate=lr, epsilon=1e-6)
    #-----------------------------------------------------------------------------------------------------------------------#
    # We need to define the shape of our input tensor first
    num_freq_samples = X_train_spec.shape[1] # number of data points (frequency axis)
    num_time_samples = X_train_spec.shape[2] # number of data points (time axis)
    num_channels = X_train_spec.shape[3] # number of channels (E, N, Z)
    input_tensor = Input(shape=(num_freq_samples, num_time_samples, num_channels), name='Input')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 1 - Dilation and Stride (decreasing kr size)
    x = Conv2D(8,
               (3,3),
               strides=(1,1), 
               padding='same',
               kernel_initializer='he_uniform', 
               dilation_rate=3, # kr is now 7x7
               name='Conv_1_Dilated')(input_tensor)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.2)(x)
    x = Conv2D(16,
               (5,5),
               strides=(3,3), # [30, 30]
               padding='same',
               kernel_initializer='he_uniform', 
               name='Conv_1')(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 1 - Resblock
    x_res = BatchNormalization()(x)
    x_res = activations.relu(x_res)
    x_res = Dropout(0.2)(x_res)
    x_res = Conv2D(16,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_1a')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = activations.relu(x_res)
    x_res = Conv2D(16,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_1b')(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    x = activations.relu(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 2 - Stride
    x = Conv2D(32,
               (3,3),
               strides=(2,2), # [15, 15]
               padding='same',
               kernel_initializer='he_uniform', 
               name='Conv_2')(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 2 - Resblock
    x_res = BatchNormalization()(x)
    x_res = activations.relu(x_res)
    x_res = Dropout(0.2)(x_res)
    x_res = Conv2D(32,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_2a')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = activations.relu(x_res)
    x_res = Conv2D(32,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_2b')(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    x = activations.relu(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 3 - Stride
    x = Conv2D(64,
               (3,3),
               strides=(5,1), # [3, 15] 
               padding='same',
               kernel_initializer='he_uniform', 
               name='Conv_3')(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 3 - Resblock
    x_res = BatchNormalization()(x)
    x_res = activations.relu(x_res)
    x_res = Dropout(0.2)(x_res)
    x_res = Conv2D(64,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_3a')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = activations.relu(x_res)
    x_res = Conv2D(64,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_3b')(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    x = activations.relu(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # LSTM 1 - Resblock bidirectional
    x_res = TimeDistributed(Bidirectional(LSTM(32,
                                               activation='tanh',
                                               recurrent_activation='sigmoid',
                                               recurrent_dropout=0.2,
                                               return_sequences=True)))(x)
    x_res = Dropout(0.4)(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    #-----------------------------------------------------------------------------------------------------------------------#
    # LSTM 2 - Unidirectional
    x = TimeDistributed((LSTM(16,
                              activation='tanh',
                              recurrent_activation='sigmoid',
                              recurrent_dropout=0.2,
                              return_sequences=True)))(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Stats vector
    x = x_vector(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 1
    x = Dense(units=64, 
              kernel_initializer='he_uniform', 
              name='Dense_1')(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.5)(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 2
    x = Dense(units=32, 
              kernel_initializer='he_uniform', 
              name='Dense_2')(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.5)(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 3
    x = Dense(units=16, 
              kernel_initializer='he_uniform', 
              name='Dense_3')(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.5)(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Predict
    predictions = Dense(2,
                        kernel_initializer='glorot_uniform', 
                        activation='softmax',
                        name='Predictions')(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    model = Model(inputs=input_tensor, outputs=predictions, name='Seismic_Model')
    #-----------------------------------------------------------------------------------------------------------------------#
    model.compile(optimizer=adam, loss=loss, metrics=[eval_metric])
    #-----------------------------------------------------------------------------------------------------------------------#
    return model

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def make_seismoacoustic_model(X_train_spec, X_train_infra_aug, lr=1e-3, loss='binary_crossentropy', eval_metric=None):

    # Define optimizer
    adam = Adam(learning_rate=lr, epsilon=1e-6)
    #-----------------------------------------------------------------------------------------------------------------------#
    # We need to define the shape of our input tensor first
    num_freq_samples = X_train_spec.shape[1] # number of data points (frequency axis)
    num_time_samples = X_train_spec.shape[2] # number of data points (time axis)
    num_channels = X_train_spec.shape[3] # number of channels (E, N, Z)
    input_tensor = Input(shape=(num_freq_samples, num_time_samples, num_channels), name='Input')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 1
    x = Conv2D(8,
               (3,3),
               strides=(1,1), 
               padding='same',
               kernel_initializer='he_uniform', 
               dilation_rate=3,
               name='Conv_1_Dilated')(input_tensor)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.2)(x)
    x = Conv2D(16,
               (5,5),
               strides=(3,3), # [30, 30]
               padding='same',
               kernel_initializer='he_uniform', 
               name='Conv_1')(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 1 - Resblock
    x_res = BatchNormalization()(x)
    x_res = activations.relu(x_res)
    x_res = Dropout(0.2)(x_res)
    x_res = Conv2D(16,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_1a')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = activations.relu(x_res)
    x_res = Conv2D(16,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_1b')(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    x = activations.relu(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 2 - Stride
    x = Conv2D(32,
               (3,3),
               strides=(2,2), # [15, 15]
               padding='same',
               kernel_initializer='he_uniform', 
               name='Conv_2')(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 2 - Resblock
    x_res = BatchNormalization()(x)
    x_res = activations.relu(x_res)
    x_res = Dropout(0.2)(x_res)
    x_res = Conv2D(32,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_2a')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = activations.relu(x_res)
    x_res = Conv2D(32,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_2b')(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    x = activations.relu(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 3 - Stride
    x = Conv2D(64,
               (3,3),
               strides=(5,1), # [3, 15]
               padding='same',
               kernel_initializer='he_uniform', 
               name='Conv_3')(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 3 - Resblock
    x_res = BatchNormalization()(x)
    x_res = activations.relu(x_res)
    x_res = Dropout(0.2)(x_res)
    x_res = Conv2D(64,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_3a')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = activations.relu(x_res)
    x_res = Conv2D(64,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_3b')(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    x = activations.relu(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # LSTM 1 - Resblock bidirectional
    x_res = TimeDistributed(Bidirectional(LSTM(32,
                                               activation='tanh',
                                               recurrent_activation='sigmoid',
                                               recurrent_dropout=0.2,
                                               return_sequences=True)))(x)
    x_res = Dropout(0.4)(x_res)
    out = Add()([x, x_res])
    x = BatchNormalization()(out)
    #-----------------------------------------------------------------------------------------------------------------------#
    # LSTM 2 - Unidirectional
    x = TimeDistributed((LSTM(16,
                              activation='tanh',
                              recurrent_activation='sigmoid',
                              recurrent_dropout=0.2,
                              return_sequences=True)))(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Flatten
    x = x_vector(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 1
    x = Dense(units=64, 
              kernel_initializer='he_uniform', 
              name='Dense_1')(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.5)(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 2
    x = Dense(units=32, 
              kernel_initializer='he_uniform', 
              name='Dense_2')(x)
    x = BatchNormalization()(x)
    x = activations.relu(x)
    x = Dropout(0.5)(x)
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    # Infrasound branch
    width = X_train_infra_aug.shape[1]
    height = X_train_infra_aug.shape[2]
    num_channels = X_train_infra_aug.shape[3]
    input_tensor_infra = Input(shape=(width, height, num_channels), name='Infrasound')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Infra Conv 1
    x_infra = Conv2D(8,
                     (3,3),
                     strides=(1,1), 
                     padding='same', 
                     kernel_initializer='he_uniform', 
                     dilation_rate=3,
                     name='Infra_Conv_1_Dilated')(input_tensor_infra)
    x_infra = BatchNormalization()(x_infra)
    x_infra = activations.relu(x_infra)
    x_infra = Dropout(0.2)(x_infra)
    x_infra = Conv2D(16,
                     (5,5),
                     strides=(1,2), # [10, 100]
                     padding='same', 
                     kernel_initializer='he_uniform', 
                     name='Infra_Conv_1')(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Infra Conv 1 - Resblock
    x_infra_res = BatchNormalization()(x_infra)
    x_infra_res = activations.relu(x_infra_res)
    x_infra_res = Dropout(0.2)(x_infra_res)
    x_infra_res = Conv2D(16,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Infra_Conv_1a')(x_infra_res)
    x_infra_res = BatchNormalization()(x_infra_res)
    x_infra_res = activations.relu(x_infra_res)
    x_infra_res = Conv2D(16,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Infra_Conv_1b')(x_infra_res)
    out = Add()([x_infra, x_infra_res])
    x_infra = BatchNormalization()(out)
    x_infra = activations.relu(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Infra Conv 2
    x_infra = Conv2D(32,
               (5,5),
               strides=(2,2), # [5, 50] 
               padding='same', 
               kernel_initializer='he_uniform', 
               name='Infra_Conv_2')(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Infra Conv 2 - Resblock
    x_infra_res = BatchNormalization()(x_infra)
    x_infra_res = activations.relu(x_infra_res)
    x_infra_res = Dropout(0.2)(x_infra_res)
    x_infra_res = Conv2D(32,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Infra_Conv_2a')(x_infra_res)
    x_infra_res = BatchNormalization()(x_infra_res)
    x_infra_res = activations.relu(x_infra_res)
    x_infra_res = Conv2D(32,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Infra_Conv_2b')(x_infra_res)
    out = Add()([x_infra, x_infra_res])
    x_infra = BatchNormalization()(out)
    x_infra = activations.relu(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Infra Conv 3
    x_infra = Conv2D(64,
               (5,5),
               strides=(1,3), # [5, 17] 
               padding='same', 
               kernel_initializer='he_uniform', 
               name='Infra_Conv_3')(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Infra Conv 3 - Resblock
    x_infra_res = BatchNormalization()(x_infra)
    x_infra_res = activations.relu(x_infra_res)
    x_infra_res = Dropout(0.2)(x_infra_res)
    x_infra_res = Conv2D(64,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Infra_Conv_3a')(x_infra_res)
    x_infra_res = BatchNormalization()(x_infra_res)
    x_infra_res = activations.relu(x_infra_res)
    x_infra_res = Conv2D(64,
                   (3,3),
                   strides=(1,1), 
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Infra_Conv_3b')(x_infra_res)
    out = Add()([x_infra, x_infra_res])
    x_infra = BatchNormalization()(out)
    x_infra = activations.relu(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    # LSTM 1 - Resblock bidirectional
    x_infra_res = TimeDistributed(Bidirectional(LSTM(32,
                                                     activation='tanh',
                                                     recurrent_activation='sigmoid',
                                                     recurrent_dropout=0.2,
                                                     return_sequences=True)))(x_infra)
    x_infra_res = Dropout(0.4)(x_infra_res)
    out = Add()([x_infra, x_infra_res])
    x_infra = BatchNormalization()(out)
    #-----------------------------------------------------------------------------------------------------------------------#
    # LSTM 2 - Unidirectional
    x_infra = TimeDistributed((LSTM(16,
                                    activation='tanh',
                                    recurrent_activation='sigmoid',
                                    recurrent_dropout=0.2,
                                    return_sequences=True)))(x_infra)
    x_infra = Dropout(0.4)(x_infra)
    x_infra = BatchNormalization()(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Stats vector
    x_infra = x_vector(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 1
    x_infra = Dense(units=64, 
              kernel_initializer='he_uniform', 
              name='Infra_Dense_1')(x_infra)
    x_infra = BatchNormalization()(x_infra)
    x_infra = activations.relu(x_infra)
    x_infra = Dropout(0.5)(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Dense layer 2
    x_infra = Dense(units=32, 
              kernel_initializer='he_uniform', 
              name='Infra_Dense_2')(x_infra)
    x_infra = BatchNormalization()(x_infra)
    x_infra = activations.relu(x_infra)
    x_infra = Dropout(0.5)(x_infra)
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    # Residual Gated Multimodal Unit
    x_seismoacoustic = GMU(x, x_infra, HIDDEN_STATE_DIM=32)
    x_seismoacoustic = Add()([x, x_infra, x_seismoacoustic])
    x_seismoacoustic = BatchNormalization()(x_seismoacoustic)
    x_seismoacoustic = Dropout(0.5)(x_seismoacoustic)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Final dense block
    x_seismoacoustic = Dense(units=128, 
                             kernel_initializer='he_uniform', 
                             name='Seismoacoustic_Dense_1')(x_seismoacoustic)
    x_seismoacoustic = BatchNormalization()(x_seismoacoustic)
    x_seismoacoustic = activations.relu(x_seismoacoustic)
    x_seismoacoustic = Dropout(0.5, name='Seismoacoustic_Dropout_1')(x_seismoacoustic)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Predict using softmax activation
    predictions = Dense(2, 
                        kernel_initializer='glorot_uniform', 
                        activation='softmax', name='Predictions')(x_seismoacoustic)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Create Model
    model = Model(inputs=[input_tensor, input_tensor_infra], outputs=predictions, name='Seismoacoustic_Model')
    # Compile
    model.compile(optimizer=adam, loss=loss, metrics=[eval_metric])
    #-----------------------------------------------------------------------------------------------------------------------#
    return model

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def GMU(x_s, x_i, HIDDEN_STATE_DIM=16):
    '''---------------------------------------------------------------------------
    Gated multimodal unit (Arevalo et al., 2020)
    ---------------------------------------------------------------------------'''
        
    h_s = Dense(units=HIDDEN_STATE_DIM,
                kernel_initializer='glorot_uniform')(x_s)
    h_s = activations.tanh(h_s)

    h_i = Dense(units=HIDDEN_STATE_DIM,
                kernel_initializer='glorot_uniform')(x_i)
    h_i = activations.tanh(h_i)

    x = concatenate([x_s, x_i], axis=1)
    z = Dense(HIDDEN_STATE_DIM,
              activation='sigmoid',
              name='z_layer',
              kernel_initializer='glorot_uniform')(x)

    h = z * h_s + (1 - z) * h_i

    return h

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def x_vector(x, two_dim=True):
    '''---------------------------------------------------------------------------
    Compute statistical vector of input (mean and standard deviation) (Snyder et al., 2018).
    ---------------------------------------------------------------------------'''
    if two_dim:
        x_mean = GlobalAveragePooling2D()(x)
        x_tmp = Subtract()([x,x_mean])
        x_std = GlobalAveragePooling2D()(x_tmp**2)
    else:
        try:
            x_mean = GlobalAveragePooling1D()(x)
        except:
            x = Reshape((1,x.shape[1]))(x)
            x_mean = GlobalAveragePooling1D()(x)
        x_tmp = Subtract()([x,x_mean])
        x_std = GlobalAveragePooling1D()(x_tmp**2)
    x = concatenate([x_mean,x_std])
    return x

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def train_models(split, f1_avg='micro', lr=1e-3, epochs=100, batch_size=64, reduce_lr_patience=3, factor=0.1, min_lr=1e-4, early_stopping_patience=10, n_splits=5, save=True):
    '''---------------------------------------------------------------------------
    Trains and saves results for both seismic and seismoacoustic models.
    ---------------------------------------------------------------------------'''
    print('Begin split # ' +str(split+1) + ' of '+ str(n_splits))
    # Train
    X_train_seismic = np.load('/Volumes/Extreme SSD/Korea_Events/Train_Data/6x/Split_'+str(split+1)+'/X_seismic.npy')
    X_train_infra = np.load('/Volumes/Extreme SSD/Korea_Events/Train_Data/6x/Split_'+str(split+1)+'/X_infra.npy')
    y_train = np.load('/Volumes/Extreme SSD/Korea_Events/Train_Data/6x/Split_'+str(split+1)+'/y.npy')
    # Test
    X_test_seismic = np.load('/Volumes/Extreme SSD/Korea_Events/Test_Data/Split_'+str(split+1)+'/X_seismic.npy')
    X_test_infra = np.load('/Volumes/Extreme SSD/Korea_Events/Test_Data/Split_'+str(split+1)+'/X_infra.npy')
    y_test = np.load('/Volumes/Extreme SSD/Korea_Events/Models/Test_Data/Split_'+str(split+1)+'/y.npy')
    #-----------------------------------------------------------------------------------------------------------------------#
    # Eval metric
    f1_score = tf.keras.metrics.F1Score(threshold=0.5, average=f1_avg)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Build seismic model
    seismic_model = make_seismic_model(X_train_seismic, loss='binary_crossentropy', eval_metric=f1_score, lr=lr)
    # Build seismoacoustic model
    seismoacoustic_model = make_seismoacoustic_model(X_train_seismic, X_train_infra, loss='binary_crossentropy', eval_metric=f1_score, lr=lr)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Set training parameters
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=reduce_lr_patience, min_lr=min_lr)
    early_stopping = EarlyStopping(monitor='val_f1_score', patience=early_stopping_patience, min_delta=0.0001, mode='max', restore_best_weights=True)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Train seismoacoustic model
    print('Start training seismoacoustic model')
    history_seismoacoustic = []
    tmp = seismoacoustic_model.fit([X_train_seismic, X_train_infra], y_train, validation_data=([X_test_seismic, X_test_infra], y_test), epochs=epochs, batch_size=batch_size,
                                   callbacks=[early_stopping, reduce_lr])
    history_seismoacoustic.append(tmp)
    # Train seismic model
    print('Start training seismic model')
    history_seismic = []
    tmp = seismic_model.fit(X_train_seismic, y_train, validation_data=(X_test_seismic, y_test), epochs=epochs, batch_size=batch_size,
                            callbacks=[early_stopping, reduce_lr])
    history_seismic.append(tmp)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Storing seismic histories and saving best model
    if save == save:
        seismic_model.save('/Volumes/Extreme SSD/Korea_Events/Models/Seismic/model_split'+str(split+1))
        np.save('/Volumes/Extreme SSD/Korea_Events/Models/Seismic/history_split'+str(split+1)+'.npy',history_seismic[0].history)
    # Storing seismoacoustic histories and saving best model
    if save == save:
        seismoacoustic_model.save('/Volumes/Extreme SSD/Korea_Events/Models/Seismoacoustic/model_split'+str(split+1))
        np.save('/Volumes/Extreme SSD/Korea_Events/Models/Seismoacoustic/history_split'+str(split+1)+'.npy',history_seismoacoustic[0].history)