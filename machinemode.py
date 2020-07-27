from machinemode import dataprep,modeldef

    
def main():
    data = dataprep.load_data()
    x_train, x_test, y_train, y_test, n_class = dataprep.data_prep(data)
    model = modeldef.build_model(x_train,y_train,'load')    
    
    predictions = model.predict(x_test)
    modeldef.model_perf(model,x_test,y_test,predictions)
if __name__ == "__main__":
    main()
