# -*- coding: utf-8 -*-

import pandas as pd
import os 

class data():
    
    def __init__(self):
        
        path = os.path.dirname(__file__)
        
        path = os.sep.join((path,'data.csv'))
        
        data = pd.read_csv(path)
        
        self.data = data
        
        X = data[['Re','Pr']].values
        y = data['Nu'].values
        
        self.X = X
        self.y = y

if __name__ == '__main__':
    
    d = data()
    
    print(d.X)
    print(d.y)