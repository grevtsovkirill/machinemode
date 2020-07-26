import pandas as pd
import xgboost
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

def build_model(opt='def'):
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 4
    param['nthread'] = 4
    param['num_class'] = 6
    model = xgboost.XGBClassifier(**param)
    return model
    
def main():
    data = load_data()
    x_train, x_test, y_train, y_test, n_class = data_prep(data)
    model = build_model()
    model.fit(x_train,y_train)
    
    
if __name__ == "__main__":
    main()
