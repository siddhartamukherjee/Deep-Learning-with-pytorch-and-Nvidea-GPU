#!/usr/bin/env python
# coding: utf-8

# In[5]:


from Data_Preprocessing import Preprocessing
from model import TabularModel
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np


# In[27]:


class Prediction():
    def __init__(self):
        pass
    
    def get_model(self):
        emb_szs = [(24, 12), (2, 1), (7, 4)]
        model = TabularModel(emb_szs, 6, 1, [200,100], p=0.4).cuda()
        model.load_state_dict(torch.load('TaxiFareRegrModel.pt'));
        model.eval() # be sure to run this step!
        return model
    
    def haversine_distance(self, lat1, long1, lat2, long2):
        r = 6371
        phi1 = np.radians(self.dfx[lat1])
        phi2 = np.radians(self.dfx[lat2])
        delta_phi = np.radians(self.dfx[lat2]-self.dfx[lat1])
        delta_lambda = np.radians(self.dfx[long2]-self.dfx[long1])
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return r * c
    
    def test_data(self): # pass in the name of the new model
        # INPUT NEW DATA
        plat = float(input('What is the pickup latitude?  '))
        plong = float(input('What is the pickup longitude? '))
        dlat = float(input('What is the dropoff latitude?  '))
        dlong = float(input('What is the dropoff longitude? '))
        psngr = int(input('How many passengers? '))
        dt = input('What is the pickup date and time?\nFormat as YYYY-MM-DD HH:MM:SS     ')
    
        # PREPROCESS THE DATA
        dfx_dict = {'pickup_latitude':plat,'pickup_longitude':plong,'dropoff_latitude':dlat,
             'dropoff_longitude':dlong,'passenger_count':psngr,'EDTdate':dt}
        self.dfx = pd.DataFrame(dfx_dict, index=[0])
        self.dfx['dist_km'] = self.haversine_distance('pickup_latitude', 'pickup_longitude',
                                        'dropoff_latitude', 'dropoff_longitude')
        self.dfx['EDTdate'] = pd.to_datetime(self.dfx['EDTdate'])
    
        # We can skip the .astype(category) step since our fields are small,
        # and encode them right away
        self.dfx['Hour'] = self.dfx['EDTdate'].dt.hour
        self.dfx['AMorPM'] = np.where(self.dfx['Hour']<12,0,1) 
        self.dfx['Weekday'] = self.dfx['EDTdate'].dt.strftime("%a")
        self.dfx['Weekday'] = self.dfx['Weekday'].replace(['Fri','Mon','Sat','Sun','Thu','Tue','Wed'],
                                                [0,1,2,3,4,5,6]).astype('int64')
        # CREATE CAT AND CONT TENSORS
        cat_cols = ['Hour', 'AMorPM', 'Weekday']
        cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
                     'dropoff_longitude', 'passenger_count', 'dist_km']
        xcats = np.stack([self.dfx[col].values for col in cat_cols], 1)
        xcats = torch.tensor(xcats, dtype=torch.int64)
        xconts = np.stack([self.dfx[col].values for col in cont_cols], 1)
        xconts = torch.tensor(xconts, dtype=torch.float)
    
        #CONVERT TO CUDA TENSORS
        xcats = xcats.cuda()
        xconts = xconts.cuda()

    
        # PASS NEW DATA THROUGH THE MODEL WITHOUT PERFORMING A BACKPROP
        with torch.no_grad():
            gpumodel = self.get_model()
            z = gpumodel(xcats, xconts)
        print(f'\nThe predicted fare amount is ${z.item():.2f}')


# In[28]:


if __name__ == '__main__':
    data_pred = Prediction()
    data_pred.test_data()


# In[ ]:




