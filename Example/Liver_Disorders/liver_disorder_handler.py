import pandas as pd
from Example.DI2 import *
from Example.auxiliary import *


csvs = ["..\\Liver_Disorders\\\dataset\\DISA\\liver_disorder_3.csv","..\\Liver_Disorders\\\dataset\\DISA\\liver_disorder_5.csv","..\\Liver_Disorders\\\dataset\\DISA\\liver_disorder_7.csv"]
results_70 = ["..\\Liver_Disorders\\results\\3\\70\\patterns.txt","..\\Liver_Disorders\\results\\5\\70\\patterns.txt","..\\Liver_Disorders\\results\\7\\70\\patterns.txt"]
results_100 = ["..\\Liver_Disorders\\results\\3\\100\\patterns.txt","..\\Liver_Disorders\\results\\5\\100\\patterns.txt","..\\Liver_Disorders\\results\\7\\100\\patterns.txt"]

for index in range(len(csvs)):
    data = pd.read_csv(csvs[index])
    class_information = {
        "values": data["drinks"],
        "outcome_value": 1,
        "type": "Numerical",
        "method": "empirical"
    }

    data = data.drop(columns=["drinks"])
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

print("################################# DI2 ##########################################")
for index in range(len(csvs)):
    data = pd.read_csv(csvs[index])
    print("####################################"+ str(bins[index]) +"############################################")
    class_cat = distribution_discretizer(data.copy(), number_of_bins=bins[index], cutoff_margin=0.0)
    class_information = {
        "values": class_cat["drinks"],
        "type": "Categorical"
    }
    print("#################################### 70% ############################################")
    patterns = retrive_patterns(results_70[index])
    stats(data, patterns, class_information)
    print("#################################### 100% ############################################")
    patterns = retrive_patterns(results_100[index])
    stats(data, patterns, class_information)



