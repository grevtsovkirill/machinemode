import pandas as pd
from sklearn.model_selection import train_test_split



def load_data(name = 'data_case_study.csv',path = './'):
    data = pd.read_csv("data_case_study.csv")
    data = data.fillna(0)
    return data

def data_prep(ds,varlist=[]):
    Y = ds['activity']
    if len(varlist)==0:
        varlist0 = list(ds.select_dtypes(include='number').columns)
        todel = ['activity','timestamp','Unnamed: 0']
        varlist = [it for it in varlist0 if it not in todel]

    X = ds[varlist]
    n_class = len(Y.value_counts().index)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    return x_train, x_test, y_train, y_test, n_class
