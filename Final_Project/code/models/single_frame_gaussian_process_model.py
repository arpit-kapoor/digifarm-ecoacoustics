import numpy as np
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
        
    def predict(self, x1x2, plot = False):
        if self.trained:
            
            n_points = int(np.sqrt(x1x2.shape[0]))


            y_pred, MSE = self.gp.predict(x1x2, return_std=True)
            y_lower = y_pred - 2*MSE
            y_higher = y_pred + 2*MSE

            if plot: 
                X0p, X1p = x1x2[:,0].reshape(n_points,n_points), x1x2[:,1].reshape(n_points,n_points)
                Zp_mean = np.reshape(y_pred,(n_points,n_points))
                Zp_lower = np.reshape(y_lower,(n_points,n_points))
                Zp_higher = np.reshape(y_higher,(n_points,n_points))

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