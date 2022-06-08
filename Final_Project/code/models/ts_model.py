import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from matplotlib import pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from models.ecoacoustic_model import EcoacousticModel

class TSModel(EcoacousticModel): 
    
    def __init__(self, p=1, d=0, q=0):
        super().__init__() 
        self.period = self.p = p
        self.d = d
        self.q = q  
    
    def split_train_test(self, X, test_ratio=0.3):

        self.T = X.shape[0]
        train_size = int(self.T*(1-test_ratio))
        
        train_data = X[:train_size]
        test_data = X[train_size:]
        
        return train_data, test_data       
        
        
    def seasonal_decompose(self, X):
        decomposition = STL(X, period=self.p).fit()
        fig = decomposition.plot()
        fig.set_size_inches(14,7)
        plt.show()
        
        return decomposition
    
    def plot_autocorrelations(self, y, plot_lims=True):
        T = y.shape[0]
        ac = sm.tsa.stattools.acf(y)
        x = range(1, len(ac)+1)
        plt.figure()
        plt.bar(x, ac)
        if plot_lims:
            plt.axhline(y=2/np.sqrt(T), color='r', linestyle='--', alpha=0.5, label='$2/\sqrt{T}$')
            plt.axhline(y=-2/np.sqrt(T), color='r', linestyle='--', alpha=0.5, label='$-2/\sqrt{T}$')
        plt.xlabel('i')
        plt.ylabel('ρ_i')
        plt.legend()
        plt.show()
    
        
    def train(self, X, verbose=False,  plot=False, **kwargs):
        
        # fit model
        model = ARIMA(
            X, 
            order=(kwargs.get('p', self.p), 
                   kwargs.get('d', self.d), 
                   kwargs.get('q', self.q)
                  )
        )
        
        self.fitted_model = model.fit()

        # summary of fit model
        if verbose:
            print(self.fitted_model.summary())

        if plot:
            # line plot of residuals
            residuals = pd.DataFrame(self.fitted_model.resid)
            self.plot_autocorrelations(residuals)

            # density plot of residuals
            residuals.plot(kind='kde')
            plt.show()

        # summary stats of residuals
        if verbose:
            print(residuals.describe())
    
    def predict(self, steps, conf_int=0.1):
        y_hat = self.fitted_model.forecast(steps=steps)
        result = self.fitted_model.get_forecast(steps).conf_int(conf_int)
        if isinstance(result, pd.DataFrame):
            result = result.values
        y_lower = result[:, 0]
        y_upper = result[:, 1]
        return y_hat, y_lower, y_upper

            
    def get_one_period_ahead_pred(self, X, **kwargs):
        
        # Preds
        preds = []
        self.train_data, self.test_data = self.split_train_test(X, test_ratio=kwargs.get('test_ratio', 0.3))

        updated_data = self.train_data.copy()
        
        # Iterate indicies for periods
        for i in tqdm(range(0, len(self.test_data.index), self.period)):
            
            # Define model
            model = ARIMA(
                updated_data, 
                order=(kwargs.get('p', self.period), 
                       kwargs.get('d', 0), 
                       kwargs.get('q', 0)
                      )
            )

            # Fit ARIMA model
            fitted_model = model.fit()
            
            # Get test samples
            indices = self.test_data.index[i: min(i+self.period, len(self.test_data))]
            test_samples = self.test_data[indices]
            
            # Forecast on test indices
            y_hat = fitted_model.forecast(steps=len(indices))
            
            # Store results in dataframe along with confidence interval
            df_pred = fitted_model.get_forecast(len(indices)).conf_int(0.1)
            df_pred.columns = ['5%', '95%']
            df_pred.loc[:, 'y_hat'] = y_hat
            df_pred.loc[:, 'y'] = test_samples.values
            
            # Append preds
            preds.append(df_pred)
            
            # Update train dataset with period data
            updated_data = pd.concat([updated_data, test_samples], axis=0)
        
        # Concatenate results for all periods
        df_test_pred = pd.concat(preds, axis=0)
 
        # Plot predictions
        fig, ax = plt.subplots(figsize=(20, 5))
        plt.plot(df_test_pred.loc[:, 'y'], label='y', c='tab:red')
        plt.plot(df_test_pred.loc[:, 'y_hat'], label='y_hat', c='tab:green')
        plt.plot(df_test_pred.loc[:, '5%'], label='5%', color='g', linestyle='--', alpha=0.5)
        plt.plot(df_test_pred.loc[:, '95%'], label='95%', color='g', linestyle='--', alpha=0.5)
        plt.fill_between(df_test_pred.index, df_test_pred.loc[:, '5%'], df_test_pred.loc[:, '95%'], 
                         color='g', alpha=0.25)
        plt.legend()
        
        ## Residuals
        df_test_pred.loc[:, 'resid'] = df_test_pred.y_hat - df_test_pred.y
        self.plot_autocorrelations(df_test_pred.resid)
        
        return df_test_pred