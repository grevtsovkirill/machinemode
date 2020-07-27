from machinemode import dataprep,modeldef

    
def main():

    data = dataprep.MachineData()
    data.data_prep()

    model = modeldef.build_model(data.x_train,data.y_train,'load')    
    
    predictions = model.predict(data.x_test)
    modeldef.model_perf(model,data.x_test,data.y_test,predictions)
if __name__ == "__main__":
    main()
