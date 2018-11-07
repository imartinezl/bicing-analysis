# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:56:10 2018

@author: Inigo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file = "2018_09_setembre_bicing_estacions.csv"
df = pd.read_csv(csv_file, sep=';',decimal=',')

df_electric = df[df['type']=='BIKE-ELECTRIC']
plt.scatter(df_electric['latitude'],df_electric['longitude'])
df_electric['total'] = df_electric['slots'] + df_electric['bikes']


df_electric.groupby('id')['total'].value_counts()
