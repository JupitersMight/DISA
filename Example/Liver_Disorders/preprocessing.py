import pandas as pd
from DISA.DISA import DISA
import DI2
import arff
import numpy as np


col_names = ["mcv","alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
dropped_cols = ["selector"]
class_variable = "drinks"

data = []
# Read data
with open("..\\Liver_Disorders\\bupa.txt", "r") as a_file:
    for line in a_file:
        data.append(line.split(","))

#Drop uninformative columns
liver_data = pd.DataFrame(data=data, columns=col_names).drop(dropped_cols, axis=1)
class_variable_numerical = liver_data[class_variable].copy()

#normalize and categorize columns
liver_data_3 = DI2.distribution_discretizer(liver_data.copy(), number_of_bins=3, cutoff_margin=0.0)
liver_data_5 = DI2.distribution_discretizer(liver_data.copy(), number_of_bins=5, cutoff_margin=0.0)
liver_data_7 = DI2.distribution_discretizer(liver_data.copy(), number_of_bins=7, cutoff_margin=0.0)

#write to arff and CSV file

arff.dump('..\\Liver_Disorders\\csv\\liver_disorder_3.arff', liver_data_3.values, names=liver_data_3.columns)
arff.dump('..\\Liver_Disorders\\csv\\liver_disorder_5.arff', liver_data_5.values, names=liver_data_5.columns)
arff.dump('..\\Liver_Disorders\\csv\\liver_disorder_7.arff', liver_data_7.values, names=liver_data_7.columns)
liver_data_3[class_variable] = class_variable_numerical
liver_data_5[class_variable] = class_variable_numerical
liver_data_7[class_variable] = class_variable_numerical
liver_data_3.to_csv('..\\Liver_Disorders\\csv\\liver_disorder_3.csv', index=False)
liver_data_5.to_csv('..\\Liver_Disorders\\csv\\liver_disorder_5.csv', index=False)
liver_data_7.to_csv('..\\Liver_Disorders\\csv\\liver_disorder_7.csv', index=False)




