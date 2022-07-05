import numpy as np
from matplotlib import pyplot as plt

from analysis.ecoacoustic_analysis import EcoacousticAnalysis

class SVDAnalysis(EcoacousticAnalysis): 
    def __init__(self, data_from_full_gp,num_sites,data_type):
        super().__init__()
        self.num_sites = num_sites
        self.data_type = data_type
        self.eof, self.d, u = np.linalg.svd(data_from_full_gp.values)
        self.path = '../results/GPs/full_gp/EOFs/'
        long_lats = np.array([list(item) for item in data_from_full_gp.index])
        n_points = int(np.sqrt(long_lats[:,0].size))
        self.X0, self.X1 = long_lats[:,0].reshape(n_points,n_points), long_lats[:,1].reshape(n_points,n_points)
        
    def show_EOF(self,resolution,eof_number):
        suf = lambda n: "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
        filename = self.path + 'EOF_number_' + suf(eof_number) + '_from_GP_'+ self.data_type +'_data_with_' + str(self.num_sites) + '_sites.png'
        index = 0
        formatted_eof = np.zeros((resolution,resolution))
        for i in range(formatted_eof.shape[0]):
            for j in range(formatted_eof.shape[1]):
                formatted_eof[i,j] = self.eof[index,eof_number-1]
                index+=1
        
        fig, ax = plt.subplots()  
        eof_plot = ax.pcolor(self.X0, self.X1, formatted_eof,cmap='RdBu_r')
        ax.title.set_text('Colour Map of the '+suf(eof_number)+' EOF')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.colorbar(eof_plot)

        self.save_results(fig,filename)
        
        plt.show()
        
    def show_variance_weights(self):
        filename = self.path + '_from_GP_'+ self.data_type +'_data_with_' + str(self.num_sites) + '_sites.png'

        fig, ax = plt.subplots()  
        ax.plot(self.d/np.sum(self.d)*100)
        ax.set_xlabel('EOF number')
        ax.set_ylabel('Weighting (%)')
        ax.title.set_text('Weighting of each EOF')

        self.save_results(fig,filename)
        
        plt.show()
        
