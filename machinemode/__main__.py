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

    
if __name__ == "__main__":
    main()
