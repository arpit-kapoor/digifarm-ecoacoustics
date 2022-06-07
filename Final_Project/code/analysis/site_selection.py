from analysis.ecoacoustic_analysis import EcoacousticAnalysis

class SiteSelection(EcoacousticAnalysis): 
    def __init__(self, data_from_full_gp,num_sites,data_type):
        super().__init__()
        