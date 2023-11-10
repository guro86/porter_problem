# -*- coding: utf-8 -*-

import numpy as np

class norm():
    
    def __init__(self,loc=0,scale=1):
        
        self.loc = loc
        self.scale = scale
        
        
    def logpdf(self,x,**kwargs):
        
        loc = kwargs.get('loc',self.loc)
        scale = kwargs.get('scale',self.scale)
                
        chi = (x - loc)/scale 
        
        return -.5 * chi**2 - np.log(scale) - .5 * np.log(2*np.pi)
        
if __name__ == '__main__':

    
    import scipy.stats
    
    #Benchmark implementations against scipy
    
    print(
        scipy.stats.norm(loc=2,scale=4).logpdf(10)
        )
    
    print(
        norm(loc=2,scale=4).logpdf(10)
        )