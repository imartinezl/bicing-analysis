# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:56:10 2018

@author: Inigo
"""
# =============================================================================
# LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
from datetime import datetime
from datetime import timedelta
from keys import *
# =============================================================================
# DATA IMPORTATION
# =============================================================================
csv_file = "data/2018_09_setembre_bicing_estacions.csv"
df = pd.read_csv(csv_file, sep=';',decimal=',')
date_format = '%d/%m/%y %H:%M:%S'

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
df_electric = df[df['type']=='BIKE-ELECTRIC']
#plt.scatter(df_electric['latitude'],df_electric['longitude'])
df_electric['datetime'] = pd.to_datetime(df_electric['updateTime'],format=date_format)
df_electric['date'] = df_electric['datetime'].dt.date
df_electric = df_electric.assign(total_space = df_electric.slots+df_electric.bikes)
df_electric = df_electric.assign(flux = df_electric.groupby('id')['bikes'].diff(1) ) 
df_electric = df_electric[df_electric['flux']!=0].dropna()

# =============================================================================
#  COST ESTIMATION
# =============================================================================

def route_estimation(waypoint0, waypoint1, app_id='', app_code=''):
    service = "https://route.api.here.com/routing/7.2/calculateroute.json"
    options = "?app_id={}&app_code={}&waypoint0=geo!{}&waypoint1=geo!{}&mode=fastest;bicycle"
    query = (service+options).format(app_id, app_code, waypoint0, waypoint1)
    r = requests.get(query)
    travelTime = r.json()['response']['route'][0]['leg'][0]['travelTime']
    distance = r.json()['response']['route'][0]['leg'][0]['length']
    trajectory = [ m['position'] for m in r.json()['response']['route'][0]['leg'][0]['maneuver'] ]
    return [travelTime, distance, trajectory]

stations = df_electric.groupby('id').first()#.reset_index()
stations_file = "stations.json"
if not os.path.isfile(stations_file):
    stations.reset_index().to_json(stations_file,'records')
stations_pairs = [(x, y) for x in stations.index for y in stations.index if x != y]
stations_pairs = list(set((i,j) if i<=j else (j,i) for i,j in stations_pairs))
def get_point(x):
    return x['latitude'].astype(str)+','+x['longitude'].astype(str)

cost_file = 'cost.csv'
if not os.path.isfile(cost_file):
    cost_list = []
    for i,j in stations_pairs:
        print(i,j)
        waypoint0 = get_point(stations.loc[i])
        waypoint1 = get_point(stations.loc[j])
        [travelTime, distance, trajectory] = route_estimation(waypoint0, waypoint1, app_id, app_code)
        cost_list.append([i,j, travelTime, distance, trajectory])
    
    cost_cols = ['origin','destination','travelTime','distance','trajectory']
    df_cost = pd.DataFrame(cost_list,columns=cost_cols)
    df_cost_inv = df_cost.copy()
    df_cost_inv['origin'] = df_cost['destination']
    df_cost_inv['destination'] = df_cost['origin']
    df_cost_inv['trajectory'] = [list(reversed(x)) for x in df_cost['trajectory']]
    df_cost = pd.concat([df_cost,df_cost_inv])
    df_cost['origin_latitude'] = stations.loc[df_cost.origin]['latitude'].values
    df_cost['origin_longitude'] = stations.loc[df_cost.origin]['longitude'].values
    df_cost['destination_latitude'] = stations.loc[df_cost.destination]['latitude'].values
    df_cost['destination_longitude'] = stations.loc[df_cost.destination]['longitude'].values
    df_cost.to_csv(cost_file,index=False)
else:
    df_cost = pd.read_csv(cost_file)
    import ast
    df_cost.trajectory = df_cost.trajectory.apply(ast.literal_eval)


# =============================================================================
# POSSIBLE TRIPS
# =============================================================================

col_names = ["origin","origin_time","destination","destination_time","cost"]
origin_names = {"id": col_names[0], "datetime": col_names[1]}
destiny_names = {"id": col_names[2], "datetime": col_names[3]}
def trips_calculation(row):
    k = 1
    cond1 = (df_electric.id != row.id)
    cond3 = (df_electric.date >= row.date - timedelta(days=1))
    cond4 = (df_electric.date <= row.date + timedelta(days=1))
    if row.flux > 0:
        cond2 = (df_electric.updateTime < row.updateTime)
        trips = df_electric[cond1 & cond2 & cond3 & cond4][['id','datetime']].rename(columns=origin_names)
        trips[col_names[2]] = row['id']
        trips[col_names[3]] = row['datetime']
        extra = [0,0,row['id'],row['datetime'],k]
        
    elif row.flux < 0:
        cond2 = (df_electric.updateTime > row.updateTime)
        trips = df_electric[cond1 & cond2 & cond3 & cond4][['id','datetime']].rename(columns=destiny_names)
        trips[col_names[0]] = row['id']
        trips[col_names[1]] = row['datetime']
        extra = [row['id'],row['datetime'],0,0,k]
    trips = trips.assign(duration=(trips['destination_time']-trips['origin_time']).dt.total_seconds())
        
    trips_ext = trips.merge(df_cost,on=['origin','destination'])
    trips_ext['cost'] = round(abs(trips_ext.duration - trips_ext.travelTime)/trips_ext.travelTime,2)
    trips_ext = trips_ext[trips_ext['cost']<k][col_names]
    exception = pd.DataFrame([extra], columns=col_names)
    trips_ext = trips_ext.append(exception)
    
    return(trips_ext)
    
trips_file = 'trips.csv'
if not os.path.isfile(trips_file):
    tmp = []
    for index, row in df_electric.iterrows():
        tmp.append(row)
    from multiprocessing.dummy import Pool as ThreadPool 
    pool = ThreadPool(16)
    trips_list = pool.map(trips_calculation, tmp)
    pool.close() 
    pool.join() 

    sum([m.shape[0] for m in trips_list])
    df_trips = pd.concat(trips_list)
    df_trips.to_csv(trips_file,index=False)
else:
    df_trips = pd.read_csv(trips_file)

opt_file = "opt.mat"
if not os.path.isfile(opt_file):
    
    from scipy.sparse import lil_matrix, csr_matrix
    c = csr_matrix(df_trips.cost, dtype=np.double )
    b = csr_matrix(abs(df_electric.flux), dtype=np.int16 )
    A = lil_matrix( (df_electric.shape[0],df_trips.shape[0]), dtype=np.int16 )#.todense()
    cont = 0
    for index, row in df_electric.iterrows():
        if cont % 10==0:
            print(cont)
        if row.flux < 0:
            A[cont,:] = ((row.id == df_trips.origin) & (row.datetime == df_trips.origin_time))
        if row.flux > 0:
            A[cont,:] = ((row.id == df_trips.destination) & (row.datetime == df_trips.destination_time))
        cont += 1
        
    from scipy.io import savemat
    savemat(opt_file, {'A':A,'b':b,'c':c})
    
else:
    from scipy.io import loadmat
    d = loadmat(opt_file)
    A = d['A']; b = d['b']; c = d['c']

# Send to cplex matlab

df_trips['flow'] = pd.read_csv('solution.csv')

# Prepare Data for P5.js
df_solution = df_trips[(df_trips['flow']>0) & (df_trips['origin']!=0) & (df_trips['destination']!=0)]
plt.hist(df_solution.flow)
df_solution.groupby('origin').size().sort_values(ascending=False)
df_solution.groupby('destination').size().sort_values(ascending=False)
    
df_complete = pd.merge(df_cost,df_solution,how="right",on=['origin','destination'])
df_complete['origin_datetime'] = pd.to_datetime(df_complete['origin_time'])
df_complete['destination_datetime'] = pd.to_datetime(df_complete['destination_time'])

df_complete['origin_datetime'].value_counts()
df_complete['destination_datetime'].value_counts()

times_origin = pd.Series(df_complete['origin_datetime'].unique()).sort_values().dt.round('15min')
times_destination = pd.Series(df_complete['destination_datetime'].unique()).sort_values().dt.round('15min')

minimum = min([min(times_origin),min(times_destination)])
maximum = max([max(times_origin),max(times_destination)])
timestamps = pd.date_range(minimum,maximum,freq='15T')#.tolist()

def find_nearest(x,elem):
    R = np.abs(x-elem)
    idx = np.where(R==R.min())[0][0]
    return idx

df_complete['origin_idx'] = df_complete['origin_datetime'].apply(lambda x: find_nearest(timestamps,x))
df_complete['origin_timestamp'] = timestamps[df_complete['origin_idx']]
df_complete['destination_idx'] = df_complete['destination_datetime'].apply(lambda x: find_nearest(timestamps,x))
df_complete['destination_timestamp'] = timestamps[df_complete['destination_idx']]

import json
print(json.dumps(json.loads(df_complete.iloc[0:1].to_json(orient='index')), indent=2))
one_day = df_complete[(df_complete['origin_datetime']>"2018-09-09") & (df_complete['origin_datetime']<"2018-09-10")]
with open('one_day.json', 'w') as outfile:
    json.dump(json.loads(one_day.to_json(orient='records')),outfile)

## DATA PREPARATION FOR KEPLER.GL
df_complete.to_csv('complete.csv', index=False)


# Prepare Data for Deck.GL
df_solution = df_trips[(df_trips['flow']>0) & (df_trips['origin']!=0) & (df_trips['destination']!=0)]
df_complete = pd.merge(df_cost,df_solution,how="right",on=['origin','destination'])
df_complete['origin_datetime'] = pd.to_datetime(df_complete['origin_time'])
df_complete['destination_datetime'] = pd.to_datetime(df_complete['destination_time'])

from math import sin, cos, sqrt, atan2, radians
def distance_lat_lon(lat1,lon1,lat2,lon2):    
    # approximate radius of earth in km
    R = 6373.0    
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))    
    distance = R * c    
    return distance


selection = df_complete.sort_values('origin_time')#.iloc[0:500]
tmin = np.min(selection.origin_datetime)
tmin = int(tmin.timestamp())
tmax = np.max(selection.destination_datetime)
tmax = int(tmax.timestamp())
loop = tmax-tmin
loop_deck = 2000
to_save = []
for row, item in selection.iterrows():
    
    for i in range(int(item.flow)):
        vendor = 1 if item.flow > 1 else 0; #item['flow']
        segments = []
        # time interpolation based on distance
        segments.append([item['origin_longitude'] + np.random.rand()*1e-4,
                         item['origin_latitude'] + np.random.rand()*1e-4,
                         np.round((int(item['origin_datetime'].timestamp())-tmin + np.random.rand()*30)/loop*loop_deck,4)  ])
    
        t = item['trajectory']
        n =  len(t)
        d = [distance_lat_lon(t[i]['latitude'],t[i]['longitude'],t[i+1]['latitude'],t[i+1]['longitude']) for i in range(0,n-1)]
        d = np.cumsum(d/np.sum(d))
        
                
    #    segments.append([item['destination_latitude'],
    #                     item['destination_longitude'],
    #                     item['destination_datetime'].timestamp()  ])
    
        tdelta = (item.destination_datetime-item.origin_datetime)
        for i in range(0,n-1):
            pt = item.origin_datetime+d[i]*tdelta 
            segments.append([t[i+1]['longitude'] + np.random.rand()*1e-4,
                             t[i+1]['latitude'] + np.random.rand()*1e-4, 
                             np.round((int(pt.timestamp())-tmin + np.random.rand()*30)/loop*loop_deck,4)])
        to_save.append({"vendor":vendor,"segments":segments})
    
import json
with open('tmp_2.json', 'w') as outfile:
    json.dump(to_save, outfile)
    

        
        
        
        
        
        
        