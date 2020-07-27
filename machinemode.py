import pandas as pd
import xgboost
from sklearn.metrics import classification_report,confusion_matrix, plot_confusion_matrix
import pickle
import matplotlib.pyplot as plt

from machinemode import dataprep

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
    
def main():
    data = dataprep.load_data()
    x_train, x_test, y_train, y_test, n_class = dataprep.data_prep(data)
    model = build_model(x_train,y_train,'load')    
    

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    plot_confusion_matrix(model, x_test, y_test)
    plt.savefig("Plots/conf_m.png", transparent=True)
    
if __name__ == "__main__":
    main()
