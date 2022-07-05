import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from models.ecoacoustic_model import EcoacousticModel

class SingleFrameGaussianProcessModel(EcoacousticModel):   
    def __init__(self,title):
        super().__init__()

        # Input space
        #kernel = RBF([1,1], (1e-2, 1e-2))
        kernel = Matern([10,10], (1e-2, 1e-2), nu=1.5) +  WhiteKernel(noise_level=0.5)

        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
        
        self.title = ("_".join(str(title).split())).replace(':','')
        self.path = '../results/GPs/single_frames/'
        
        self.trained = False


    def train(self,X,y):
        self.gp.fit(X, y)
        self.X, self.y = X, y
        self.filename = self.path + 'single_frame_gp_'+str(X.shape[0])+'_stations_at_'+self.title +'.png'
        self.trained = True
        
    def predict(self, x1x2,latlons, plot = False):
        if self.trained:
            latlons = np.array(latlons)
            lonlats = np.array([latlons[:,1],latlons[:,0]]).T
            str_lonlats = ["_".join(map(str, x)) for x in lonlats]

            n_points = int(np.sqrt(lonlats.shape[0]))

            y_pred, MSE = self.gp.predict(x1x2, return_std=True)
            y_lower1 = y_pred - 2*MSE
            y_higher1 = y_pred + 2*MSE
            y_lower = [item if item >= 0 else 0 for item in y_lower1]
            y_higher = [item if item >= 0 else 0 for item in y_higher1]

            
            y_pred_df = pd.DataFrame({'long':x1x2[:,0],'lat':x1x2[:,1],'val':y_pred})
            y_pred_df = y_pred_df.set_index(['long','lat'])
            y_pred_df['new_index'] = ["_".join(map(str, x)) for x in y_pred_df.index]
            y_pred_df = y_pred_df.set_index('new_index')
            y_pred_df = y_pred_df.reindex(str_lonlats)
            y_pred_df['longs'] = [float(num.split('_')[0]) for num in y_pred_df.index]
            y_pred_df['lats'] = [float(num.split('_')[1]) for num in y_pred_df.index]
            y_pred_df = y_pred_df.set_index(['longs','lats'])

            
            y_lower_df = pd.DataFrame({'long':x1x2[:,0],'lat':x1x2[:,1],'val':y_lower})
            y_lower_df = y_lower_df.set_index(['long','lat'])
            y_lower_df['new_index'] = ["_".join(map(str, x)) for x in y_lower_df.index]
            y_lower_df = y_lower_df.set_index('new_index')
            y_lower_df = y_lower_df.reindex(str_lonlats)
            y_lower_df['longs'] = [float(num.split('_')[0]) for num in y_lower_df.index]
            y_lower_df['lats'] = [float(num.split('_')[1]) for num in y_lower_df.index]
            y_lower_df = y_lower_df.set_index(['longs','lats'])

            
            y_higher_df = pd.DataFrame({'long':x1x2[:,0],'lat':x1x2[:,1],'val':y_higher})
            y_higher_df = y_higher_df.set_index(['long','lat'])
            y_higher_df['new_index'] = ["_".join(map(str, x)) for x in y_higher_df.index]
            y_higher_df = y_higher_df.set_index('new_index')
            y_higher_df = y_higher_df.reindex(str_lonlats)
            y_higher_df['longs'] = [float(num.split('_')[0]) for num in y_higher_df.index]
            y_higher_df['lats'] = [float(num.split('_')[1]) for num in y_higher_df.index]
            y_higher_df = y_higher_df.set_index(['longs','lats'])

            
            MSE_df = pd.DataFrame({'long':x1x2[:,0],'lat':x1x2[:,1],'val':MSE})
            MSE_df = MSE_df.set_index(['long','lat'])
            MSE_df['new_index'] = ["_".join(map(str, x)) for x in MSE_df.index]
            MSE_df = MSE_df.set_index('new_index')
            MSE_df = MSE_df.reindex(str_lonlats)
            MSE_df['longs'] = [float(num.split('_')[0]) for num in MSE_df.index]
            MSE_df['lats'] = [float(num.split('_')[1]) for num in MSE_df.index]
            MSE_df = MSE_df.set_index(['longs','lats'])     
                                            
            if plot: 
                X0p, X1p = lonlats[:,0].reshape(n_points,n_points), lonlats[:,1].reshape(n_points,n_points)
                Zp_mean = np.reshape(y_pred_df.values,(n_points,n_points))
                
                Zp_lower = np.reshape(y_lower_df.values,(n_points,n_points))
                Zp_higher = np.reshape(y_higher_df.values,(n_points,n_points))

                vmin = np.min(y_lower)
                vmax = np.max(y_higher)

                fig, ax = plt.subplots(1,4,figsize=(20,4),gridspec_kw={"width_ratios":[1,1,1,0.05]})

                pred_plot = ax[1].pcolormesh(X0p, X1p, Zp_mean,vmin=vmin,vmax=vmax,cmap='RdBu_r',shading='auto')
                ax[1].scatter(self.X[:,0],self.X[:,1],s = self.y*100,c='g')
                ax[1].title.set_text('μ')
                ax[1].set_xlabel('Longitude')
                ax[1].set_ylabel('Latitude')

                ax[0].pcolormesh(X0p, X1p, Zp_lower,vmin=vmin,vmax=vmax,cmap='RdBu_r',shading='auto')
                ax[0].scatter(self.X[:,0],self.X[:,1],s = self.y*100,c='g')
                ax[0].title.set_text('-2σ')
                ax[0].set_xlabel('Longitude')
                ax[0].set_ylabel('Latitude')

                ax[2].pcolormesh(X0p, X1p, Zp_higher,vmin=vmin,vmax=vmax,cmap='RdBu_r',shading='auto')
                ax[2].scatter(self.X[:,0],self.X[:,1],s = self.y*100,c='g')
                ax[2].title.set_text('+2σ')
                ax[2].set_xlabel('Longitude')
                ax[2].set_ylabel('Latitude')

                fig.colorbar(pred_plot, cax=ax[3])
                fig.suptitle(str(self.title))

                self.save_results(fig,self.filename)
                plt.show()

            return y_pred, y_lower, y_higher
        else:
            print('Train the model first!')