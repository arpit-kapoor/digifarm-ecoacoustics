import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from analysis.ecoacoustic_analysis import EcoacousticAnalysis

class Visualise(EcoacousticAnalysis):
    def __init__(self):

        self.path = '../results/raw_data/'
        
    def plot_all_ts(self,single_site_data,site_name,suffix=''):
        filename = self.path + 'station_time_series/time_series_' + site_name + '_' + suffix + '.png'
        fig, ax = plt.subplots(figsize=(15, 3))  
        ax.plot(single_site_data.values)
        ax.set_title(site_name)
        self.save_results(fig,filename)
        
        plt.show()
    
    def point_animation(self,ecoacoustic_data,metadata,suffix = ''):  
        names = list(ecoacoustic_data.columns)
        num_sites = len(names)
        
        filename = self.path + 'station_animations/animation_' + str(num_sites) + '_sites_' + suffix + '.png'
        fig,ax = plt.subplots()
        lats, longs = [],[]
        
        for name in names:
            lats.append(metadata['latitude'][name])
            longs.append(metadata['longitude'][name])

        scatter = ax.scatter(longs, lats,s=np.zeros(len(names)))
        ax.title.set_text('Raw Ecoacoustic Data Animation')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        def init():
            scatter.set_sizes(np.zeros(len(names)))
            return scatter

        def update(frame):
            data = frame
            scatter.set_sizes(data*100)
            return scatter

        anim = FuncAnimation(fig, update, interval=1,frames = ecoacoustic_data.values,init_func=init)
        self.save_results(anim,filename,True)