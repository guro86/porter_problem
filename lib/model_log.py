import numpy as np

class model():
    
    def __init__(self,X):
        
        #Input data
        self.X = X
        
        #A designmatrix that is n_x x 3
        Xd = np.ones((len(X),3))
        
        #Set the log values
        Xd[:,-2:] = np.log(X)
        
        #Save design matrix
        self.Xd = Xd
        
    def predict_log(self,theta,e=None):
                
        #If just one specific experiment is asked for
        if e is not None: 
            
            #Design matrix given log data
            Xd = self.Xd[e,:]
        
        #Else, take all data
        else:
            #Design matrix given log data
            Xd = self.Xd
        
        #Log pred
        return Xd @ theta.T
    
    def predict(self,theta,e=None):
        
        #Predict logs
        log_pred = self.predict_log(theta,e)
        
        #Scale back
        return np.exp(log_pred)
        
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    import sys
    sys.path.append('../')

    #Import data    
    from data import data
    
    #Load data
    d = data()
    
    #Set up the model
    m = model(X=d.X)
    
    #Set some model parameters
    theta = np.array([-5,1,.4])
    
    #Plot predictions againsts output
    plt.plot(
        m.predict(theta),
        d.y,
        'o'
        )
    
    print(m.predict(theta,0))