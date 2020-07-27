import pandas as pd
import matplotlib.pyplot as plt

from machinemode import dataprep,modeldef

    
def main():
    data = dataprep.load_data()
    x_train, x_test, y_train, y_test, n_class = dataprep.data_prep(data)
    model = modeldef.build_model(x_train,y_train,'load')    
    
    predictions = model.predict(x_test)
    print(modeldef.classification_report(y_test, predictions))
    print(modeldef.confusion_matrix(y_test, predictions))
    modeldef.plot_confusion_matrix(model, x_test, y_test)
    plt.savefig("Plots/conf_m.png", transparent=True)
    
if __name__ == "__main__":
    main()
