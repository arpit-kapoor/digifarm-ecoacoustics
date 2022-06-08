from os.path import exists

class EcoacousticAnalysis:
    def __init__(self):
        print('Analyzing!')
        
    def save_results(self,fig_anim,filename,animation=False):
        if exists(filename):
            print('This file already exists')
        else:
            if animation:
                fig_anim.save(filename)
            else:
                fig_anim.savefig(filename)