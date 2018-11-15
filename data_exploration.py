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
    df_cost.to_csv(cost_file,index=False)
else:
    df_cost = pd.read_csv(cost_file)


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
    trips = trips.assign(duration=(trips['destination_time']-trips['origin_time']).dt.seconds)
        
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
if not os.path.isfile(trips_file):
    
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
    
df_bicing = df_trips[(df_trips['flow']>0) & (df_trips['origin']!=0) & (df_trips['destination']!=0)]
plt.hist(df_bicing.flow)
df_bicing.groupby('origin').size().sort_values(ascending=False)
df_bicing.groupby('destination').size().sort_values(ascending=False)
    