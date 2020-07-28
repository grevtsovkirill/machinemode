import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MachineData:
    def __init__(self, data_dir = './', filename = 'data_case_study.csv'):
        self.data_dir=data_dir
        self.filename=filename



    def load_data(self):
        data = pd.read_csv(self.data_dir+self.filename)
        data = data.fillna(0)
        self.cleandata = data
        
        
    def data_prep(self,varlist=[]):
        self.load_data()
        Y = self.cleandata['activity']
        n_class = len(Y.value_counts().index)
        self.n_class = n_class
        if len(varlist)==0:
            varlist0 = list(self.cleandata.select_dtypes(include='number').columns)
            todel = ['activity','timestamp','Unnamed: 0']
            varlist = [it for it in varlist0 if it not in todel]
        
        X = self.cleandata[varlist]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 123)
        self.fracs(Y,y_test,y_train)
        
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train

    def fracs(self, y, y_test,y_train):
        act_type= list(y.value_counts().index)
        df = y.value_counts().to_frame(name="tot")
        df=df/len(y)
        df['test'] = y_test.value_counts()/len(y_test)
        df['train'] = y_train.value_counts()/len(y_train)
        df.plot.barh().invert_yaxis()
        if not os.path.exists('Plots'):
            os.makedirs('Plots')
        plt.savefig("Plots/split.png", transparent=True)
