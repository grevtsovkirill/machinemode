import pickle
import os
import xgboost
from sklearn.metrics import classification_report,confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

def get_model(x,y,num_class,opt=''):

    if opt == 'load' and os.path.exists('model_xgb_def.pickle'):
        print("read pre-trained model")
        with open('model_xgb_def.pickle', 'rb') as f:
            model = pickle.load(f)
    else:
        print("build model")
        model = build_model(num_class)
        print("train model")
        model.fit(x,y)
        with open('model_xgb_def.pickle', 'wb') as f:
            pickle.dump(model, f)
        
    return model


def build_model(num_class):
    param = {}
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.1
    param['max_depth'] = 4
    param['nthread'] = 4
    param['num_class'] = num_class
    model = xgboost.XGBClassifier(**param)    
    return model


def model_perf(model,x,y,pred):
    print(classification_report(y, pred))
    print(confusion_matrix(y, pred))
    plot_confusion_matrix(model, x, y)
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    plt.savefig("Plots/conf_m.png", transparent=True)
