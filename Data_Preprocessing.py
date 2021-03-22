#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# In[82]:


class Preprocessing:
    def __init__(self,datafile):
        self.df = pd.read_csv(datafile)
    
    def haversine_distance(self):
        """
        Calculates the haversine distance between 2 sets of GPS coordinates in df
        """
        r = 6371  # average radius of Earth in kilometers
        lat1 = 'pickup_latitude'
        lat2 = 'pickup_longitude'
        long1 = 'dropoff_latitude'
        long2 = 'dropoff_longitude'
       
        phi1 = np.radians(self.df[lat1])
        phi2 = np.radians(self.df[lat2])
    
        delta_phi = np.radians(self.df[lat2]-self.df[lat1])
        delta_lambda = np.radians(self.df[long2]-self.df[long1])
     
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        self.df['dist_km'] = (r * c) # in kilometers
        
    def create_datetime_columns(self):
        
        self.df['EDTdate'] = pd.to_datetime(self.df['pickup_datetime'].str[:19]) - pd.Timedelta(hours=4)
        self.df['Hour'] = self.df['EDTdate'].dt.hour
        self.df['AMorPM'] = np.where(self.df['Hour']<12,'am','pm')
        self.df['Weekday'] = self.df['EDTdate'].dt.strftime("%a")
        
    def separate_columns(self):
        
        cat_cols = ['Hour', 'AMorPM', 'Weekday']
        cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'dist_km']
        y_col = ['fare_amount']  # this column contains the labels
        
        # Convert our three categorical columns to category dtypes.
        for cat in cat_cols:
            self.df[cat] = self.df[cat].astype('category')
            
        hr = self.df['Hour'].cat.codes.values
        ampm = self.df['AMorPM'].cat.codes.values
        wkdy = self.df['Weekday'].cat.codes.values

        cats = np.stack([hr, ampm, wkdy], 1)
            
        # Convert categorical variables to a tensor
        self.cats = torch.tensor(cats, dtype=torch.int64) 
        
        # This will set embedding sizes for Hours, AMvsPM and Weekdays
        cat_szs = [len(self.df[col].cat.categories) for col in cat_cols]
        self.emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
        
        # Convert continuous variables to a tensor
        conts = np.stack([self.df[col].values for col in cont_cols], 1)
        self.conts = torch.tensor(conts, dtype=torch.float)
        
        # Convert labels to a tensor
        self.y = torch.tensor(self.df[y_col].values, dtype=torch.float).reshape(-1,1)
        
    def split(self, batch_size):
        test_size = int(batch_size * .2)
        self.cat_train = self.cats[:batch_size-test_size]
        self.cat_test = self.cats[batch_size-test_size:batch_size]
        self.con_train = self.conts[:batch_size-test_size]
        self.con_test = self.conts[batch_size-test_size:batch_size]
        self.y_train = self.y[:batch_size-test_size]
        self.y_test = self.y[batch_size-test_size:batch_size]
    


# In[ ]:





# In[ ]:




