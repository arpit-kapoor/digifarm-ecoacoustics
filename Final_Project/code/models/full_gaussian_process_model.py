import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from models.ecoacoustic_model import EcoacousticModel


class FullGaussianProcessModel(EcoacousticModel): 
#         filename = 
    def __init__(self):
        super().__init__()
        self.path = '../results/GPs/full_gp/'
        self.trained = False
        #self.kernel = RBF([10,10,10], (1e-2, 1e-2))
        self.kernel = Matern([10,10,10], (1e-2, 1e-2), nu=1.5) +  WhiteKernel(noise_level=0.5)
        self.gp = GaussianProcessRegressor(self.kernel, n_restarts_optimizer=15)
        
    def train(self, X, y):
        self.filename = self.path + 'full_gp_'+str(np.unique(X[:,0]).size)+'_stationsnewkernelhalfcrop2.gif'
        self.filename_MSE = self.path + 'full_gp_'+str(np.unique(X[:,0]).size)+'_stationsnewkernel2_MSEhalfcrop.gif'
        self.X = X
        self.y = y
        self.gp.fit(X, y)
        self.trained = True

    def predict(self, x1x2x3, plot = False):
        self.x1x2x3 = x1x2x3
        if self.trained:
            #y_pred, MSE = self.gp.predict(x1x2x3, return_cov=True)
            y_pred, MSE = self.gp.predict(x1x2x3, return_std=True)
            
            y_lower1 = y_pred - 2*MSE
            y_higher1 = y_pred + 2*MSE
            
            y_lower = [item if item >= 0 else 0 for item in y_lower1]
            y_higher = [item if item >= 0 else 0 for item in y_higher1]
            
            self.vmin = np.min(y_lower)
            self.vmax = np.max(y_higher)

            self.vmin_MSE = np.min(MSE)
            self.vmax_MSE = np.max(MSE)
            
            raw_data = pd.DataFrame({'long':self.X[:,0],'lat':self.X[:,1],'time':self.X[:,2],'val':self.y}).set_index(['long','lat','time'])
            wide_raw_data =  raw_data.pivot_table(values=['val'], index= ['long','lat'],columns='time')
            wide_raw_data = wide_raw_data.droplevel(level=0, axis=1)
            wide_raw_data.columns.name = None
                        
            pred_df = pd.DataFrame({'y_pred':y_pred,'y_lower':y_lower,'y_higher':y_higher,'MSE':MSE})
            pred_df[pred_df < 0] = 0
            cov_df = pd.DataFrame({'long':x1x2x3[:,0],'lat':x1x2x3[:,1],'time':x1x2x3[:,2]})

            df = pd.concat([pred_df, cov_df], axis=1)

            wide_ypred =  df.pivot_table(values=['y_pred'], index= ['long','lat'],columns='time')
            wide_ypred = wide_ypred.droplevel(level=0, axis=1)
            wide_ypred.columns.name = None

            wide_ylower =  df.pivot_table(values=['y_lower'], index= ['long','lat'],columns='time')
            wide_ylower = wide_ylower.droplevel(level=0, axis=1)
            wide_ylower.columns.name = None

            wide_yhigher =  df.pivot_table(values=['y_higher'], index= ['long','lat'],columns='time')
            wide_yhigher = wide_yhigher.droplevel(level=0, axis=1)
            wide_yhigher.columns.name = None

            wide_MSE =  df.pivot_table(values=['MSE'], index= ['long','lat'],columns='time')
            wide_MSE = wide_MSE.droplevel(level=0, axis=1)
            wide_MSE.columns.name = None

            if plot:
                self.create_animation(wide_ypred, wide_ylower, wide_yhigher, wide_MSE, wide_raw_data)

            return wide_ypred, wide_ylower, wide_yhigher, wide_MSE
        else:
            print('Train the model first!')
        
    def create_animation(self,pred,lower,higher,MSE, raw_data):
        long_lats = np.array([list(item) for item in pred.index])
        raw_long_lats = np.array([list(item) for item in raw_data.index])

        n_points = int(np.sqrt(long_lats[:,0].size))
        X0, X1 = long_lats[:,0].reshape(n_points,n_points), long_lats[:,1].reshape(n_points,n_points)

        def format_title(datetime):
            return str(dt.datetime.fromtimestamp(int(np.round(datetime,0))).date())
        
        fig1, ax1 = plt.subplots(1,4,figsize=(20,4),gridspec_kw={"width_ratios":[1,1,1,0.05]})
        fig1.subplots_adjust(wspace=0.3)
        
        #for row in ax1:
        #    row.plot(self.x1x2x3[:,0],self.x1x2x3[:,1])

        ax1[0].title.set_text('-2σ')
        ax1[0].set_xlabel('Longitude')
        ax1[0].set_ylabel('Latitude') 
        
        ax1[1].title.set_text('μ')
        ax1[1].set_xlabel('Longitude')
        ax1[1].set_ylabel('Latitude')        

        ax1[2].title.set_text('+2σ')
        ax1[2].set_xlabel('Longitude')
        ax1[2].set_ylabel('Latitude') 
        
        lower_plot = ax1[0].pcolormesh(X0, X1, np.reshape(lower[lower.columns[0]].values,(n_points,n_points)),vmin=self.vmin,vmax=self.vmax,cmap='Greens',shading='auto')
        lower_plot_scatter = ax1[0].scatter(raw_long_lats[:,0], raw_long_lats[:,1], s = raw_data[raw_data.columns[0]].values*100,c='b')

        pred_plot = ax1[1].pcolormesh(X0, X1, np.reshape(pred[pred.columns[0]].values,(n_points,n_points)),vmin=self.vmin,vmax=self.vmax,cmap='Greens',shading='auto')
        pred_plot_scatter = ax1[1].scatter(raw_long_lats[:,0], raw_long_lats[:,1], s = raw_data[raw_data.columns[0]].values*100,c='b')
        
        higher_plot = ax1[2].pcolormesh(X0, X1, np.reshape(higher[higher.columns[0]].values,(n_points,n_points)),vmin=self.vmin,vmax=self.vmax,cmap='Greens',shading='auto')
        higher_plot_scatter = ax1[2].scatter(raw_long_lats[:,0], raw_long_lats[:,1], s = raw_data[raw_data.columns[0]].values*100,c='b')
        fig1.colorbar(pred_plot, cax=ax1[3])
        
        title_1 = fig1.suptitle(format_title(pred.columns[0]))
        
        def animate(i):
            lower_plot.set_array(np.reshape(lower[lower.columns[i+1]].values,(n_points,n_points)).flatten())
            lower_plot_scatter.set_sizes(raw_data[raw_data.columns[i+1]].values*100)
            
            pred_plot.set_array(np.reshape(pred[pred.columns[i+1]].values,(n_points,n_points)).flatten())
            pred_plot_scatter.set_sizes(raw_data[raw_data.columns[i+1]].values*100)

            higher_plot.set_array(np.reshape(higher[higher.columns[i+1]].values,(n_points,n_points)).flatten())
            pred_plot_scatter.set_sizes(raw_data[raw_data.columns[i+1]].values*100)
            
            title_1.set_text(format_title(pred.columns[i+1]))
             
            return lower_plot,pred_plot,higher_plot
        
        anim = FuncAnimation(fig1, animate, interval=500, frames=pred.shape[1] - 1,blit=False)
        self.save_results(anim,self.filename,True)
        
        
        fig2, ax2 = plt.subplots()  
        ax2.title.set_text('MSE')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')

        
        MSE_plot = ax2.pcolormesh(X0, X1, np.reshape(MSE[MSE.columns[0]].values,(n_points,n_points)),vmin=self.vmin_MSE,vmax=self.vmax_MSE,cmap='RdBu_r',shading='auto')        
        title_2 = fig2.suptitle(format_title(MSE.columns[0]))
        fig2.colorbar(MSE_plot)

        def animate_MSE(i):
            MSE_plot.set_array(np.reshape(MSE[MSE.columns[i+1]].values,(n_points,n_points)).flatten())
            title_2.set_text(format_title(MSE.columns[i+1]))
            return MSE_plot
        
        anim_MSE = FuncAnimation(fig2, animate_MSE, interval=500, frames=MSE.shape[1] - 1,blit=False)
        self.save_results(anim_MSE,self.filename_MSE,True)