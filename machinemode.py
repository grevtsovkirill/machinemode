import pandas as pd

def load_data(name = 'data_case_study.csv',path = './'):
    data = pd.read_csv("data_case_study.csv")
    return data
def main():
    data = load_data()
    print(data.head())

if __name__ == "__main__":
    main()
