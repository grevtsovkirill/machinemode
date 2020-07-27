import xgboost
from sklearn.metrics import classification_report,confusion_matrix, plot_confusion_matrix
import pickle
import matplotlib.pyplot as plt


def build_model(x,y,opt='def'):
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 4
    param['nthread'] = 4
    param['num_class'] = 6
    model = xgboost.XGBClassifier(**param)

    if opt=='def':
        model.fit(x,y)
        with open('model_xgb_def.pickle', 'wb') as f:
            pickle.dump(model, f)

    elif opt=='load':
        with open('model_xgb_def.pickle', 'rb') as f:
            model = pickle.load(f)
    
    return model

def model_perf(model,x,y,pred):
    print(classification_report(y, pred))
    print(confusion_matrix(y, pred))
    plot_confusion_matrix(model, x, y)
    plt.savefig("Plots/conf_m.png", transparent=True)