from machinemode import dataprep,modeldef

    
def main():

    data = dataprep.MachineData()
    '''
    Prepare data for training: split train/test
    '''
    data.data_train()

    model = modeldef.get_model(data.x_train,data.y_train,data.n_class,'load')       
    predictions = model.predict(data.x_test)
    modeldef.model_perf(model,data.x_test,data.y_test,predictions)


    '''
    Alternatively - usage for existing model, to apply on new data:
    
        data2 = dataprep.MachineData()
        data2.data_apply()
        model2 = modeldef.load()
        print(model2.predict(data2.X))
    '''
if __name__ == "__main__":
    main()
