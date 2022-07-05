from analysis.ecoacoustic_analysis import EcoacousticAnalysis
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

class SiteSelection(EcoacousticAnalysis): 
    def __init__(self, y_pred, MSE, ac_func_name, num_sites):
        super().__init__()
        
        self.path = '../results/site_selection/'
        self.filename = self.path + ac_func_name + '_for_' + str(num_sites) + '_stations2.png' 
        self.long_lats = np.array([list(item) for item in y_pred.index])
        
        self.ac_func_name = ac_func_name        
        ac_func_values = []
        
        for time in y_pred.columns:
            mu = y_pred[time]
            sigma = MSE[time]
            if ac_func_name == 'UCB':
                ac_func_values.append(self.ucb(mu,sigma))
            elif ac_func_name == 'PI':
                ac_func_values.append(self.pi(mu,sigma,y_pred[time]))
            elif ac_func_name == 'EI':
                ac_func_values.append(self.ei(mu,sigma,y_pred[time]))
            elif ac_func_name == 'std':
                ac_func_values.append(self.std(sigma))
            elif ac_func_name == 'ymaxstd':
                ac_func_values.append(self.ymaxstd(sigma,mu))
                                
        mean_ac_func_values = np.mean(ac_func_values,axis=0)
        self.next_loc = self.long_lats[np.argmax(np.nan_to_num(mean_ac_func_values))]
        self.plot_mean_ac_func(mean_ac_func_values)
        
    def std(self,sigma):
        return sigma
    
    def ymaxstd(self,sigma,mu):
        return sigma + mu
    
    def ucb(self,mu,sigma,kappa=4):
        return mu + kappa*sigma
    
    def pi(self,mu,sigma,y_pred,xi=0.01):
        y_max = np.max(y_pred)
        Z = (mu-y_max-xi)/sigma
        return norm.cdf(Z)
    
    def ei(self,mu,sigma,y_pred,xi=0.01):
        y_max = np.max(y_pred)
        Z = (mu-y_max-xi)/sigma
        return sigma*(Z*norm.cdf(Z)+norm.pdf(Z))
    
    def plot_mean_ac_func(self,mean_ac_func_values):
        n_points = int(np.sqrt(self.long_lats[:,0].size))
        X0, X1 = self.long_lats[:,0].reshape(n_points,n_points), self.long_lats[:,1].reshape(n_points,n_points)
        vals = np.reshape(mean_ac_func_values,(n_points,n_points))
        
        fig, ax = plt.subplots()  
        ax.title.set_text(self.ac_func_name)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.text(np.min(self.long_lats[:,0]), np.min(self.long_lats[:,1]), 'Next site location is: ' + str(self.next_loc))
        
        ac_func_plot = ax.pcolormesh(X0,X1,vals,cmap='RdBu_r',shading='auto')     
        ax.scatter([self.next_loc[0]],[self.next_loc[1]],color='g',s=100)
        fig.colorbar(ac_func_plot)
        self.save_results(fig,self.filename)
        