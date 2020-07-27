import pandas as pd
from sklearn.model_selection import train_test_split

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
        if len(varlist)==0:
            varlist0 = list(self.cleandata.select_dtypes(include='number').columns)
            todel = ['activity','timestamp','Unnamed: 0']
            varlist = [it for it in varlist0 if it not in todel]
        
        X = self.cleandata[varlist]
        n_class = len(Y.value_counts().index)
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
        #return x_train, x_test, y_train, y_test, n_class
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
