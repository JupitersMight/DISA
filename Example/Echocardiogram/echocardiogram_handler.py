import pandas as pd
from Example.DI2 import *
from Example.auxiliary import *

csvs = ["..\\Echocardiogram\\\dataset\\DISA\\echocardiogram_3.csv","..\\Echocardiogram\\\dataset\\DISA\\echocardiogram_5.csv","..\\Echocardiogram\\\dataset\\DISA\\echocardiogram_7.csv"]
results_70_plots = ["..\\Echocardiogram\\results\\3\\70\\plots\\","..\\Echocardiogram\\results\\5\\70\\plots\\","..\\Echocardiogram\\results\\7\\70\\plots\\"]
results_70 = ["..\\Echocardiogram\\results\\3\\70\\patterns.txt","..\\Echocardiogram\\results\\5\\70\\patterns.txt","..\\Echocardiogram\\results\\7\\70\\patterns.txt"]
results_100_plots = ["..\\Echocardiogram\\results\\3\\100\\plots\\","..\\Echocardiogram\\results\\5\\100\\plots\\","..\\Echocardiogram\\results\\7\\100\\plots\\"]
results_100 = ["..\\Echocardiogram\\results\\3\\100\\patterns.txt","..\\Echocardiogram\\results\\5\\100\\patterns.txt","..\\Echocardiogram\\results\\7\\100\\patterns.txt"]

for index in range(len(csvs)):
    data = pd.read_csv(csvs[index])
    class_information = {
        "values": data["survival"],
        "outcome_value": 1,
        "type": "Numerical",
        "method": "empirical"
    }

    data = data.drop(columns=["survival"])
    print("################################# Empirical #########################################")
    print("#################################### 70% ############################################")
    patterns = retrive_patterns(results_70[index])
    stats(data, patterns, class_information)

    print("#################################### 100% ############################################")
    patterns = retrive_patterns(results_100[index])
    stats(data, patterns, class_information)

    class_information["method"] = "gaussian"
    print("################################# Gaussian ##########################################")
    print("#################################### 70% ############################################")
    patterns = retrive_patterns(results_70[index])
    stats(data, patterns, class_information)
    print("#################################### 100% ############################################")
    patterns = retrive_patterns(results_100[index])
    stats(data, patterns, class_information)

bins = [3,5,7]

for index in range(len(csvs)):
    data = pd.read_csv(csvs[index])
    print("####################################"+ str(bins[index]) +"############################################")
    class_cat = distribution_discretizer(data.copy(), number_of_bins=bins[index], cutoff_margin=0.0)
    class_information = {
        "values": class_cat["survival"],
        "type": "Categorical"
    }
    print("#################################### 70% ############################################")
    patterns = retrive_patterns(results_70[index])
    stats(data, patterns, class_information)
    print("#################################### 100% ############################################")
    patterns = retrive_patterns(results_100[index])
    stats(data, patterns, class_information)


