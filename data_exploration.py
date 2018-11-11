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
from keys import *
# =============================================================================
# DATA IMPORTATION
# =============================================================================
csv_file = "data/2018_09_setembre_bicing_estacions.csv"
df = pd.read_csv(csv_file, sep=';',decimal=',')

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
df_electric = df[df['type']=='BIKE-ELECTRIC']
#plt.scatter(df_electric['latitude'],df_electric['longitude'])
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

df_trips = []
col_names = ["origin_id","origin_time","destiny_id","destiny_time"]
origin_names = {"id": col_names[0], "updateTime": col_names[1]}
destiny_names = {"id": col_names[2], "updateTime": col_names[3]}
for index, row in df_electric.iterrows():
    print(row)
    if row.flux > 0:
        cond1 = (df_electric.id != row.id)
        cond2 = (df_electric.updateTime < row.updateTime)
        trip = df_electric[cond1 & cond2][['id','updateTime']].rename(columns=origin_names)
        trip[col_names[2]] = row['id']
        trip[col_names[3]] = row['updateTime']
        
        system = pd.DataFrame([ [0,0,row['id'],row['updateTime']] ], columns=col_names)
        trip.append(system)
        
    elif row.flux < 0:
        df_electric.id != row.id
        df_electric.updateTime > row.updateTime
df.trips.append()



