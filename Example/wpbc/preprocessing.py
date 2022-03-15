import pandas as pd
from DISA.DISA import DISA
import DI2
import arff
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



origin_file = open("..\\wpbc\\wpbc.data")
origin_data = []
for line in origin_file:
    curr_array = line.split(',')
    curr_array[len(curr_array)-1] = curr_array[len(curr_array)-1].replace("\n","")
    origin_data.append(curr_array)

origin_dataframe = pd.DataFrame(data=origin_data[1:], columns=origin_data[0]).drop(columns=["id"])

for column in origin_dataframe:
    origin_dataframe[column] = origin_dataframe[column].astype(float)
non_rec = origin_dataframe[origin_dataframe.outcome == 0].drop(columns=["outcome"]).reset_index(drop=True)
rec = origin_dataframe[origin_dataframe.outcome == 1].drop(columns=["outcome"]).reset_index(drop=True)

class_variable_numerical_non_rec = non_rec["time"].copy()
class_variable_numerical_rec = rec["time"].copy()

#non_rec_csv = pd.read_csv("..\\wpbc\\csv\\wpbc_non.csv")
#rec_csv = pd.read_csv("..\\wpbc\\csv\\wpbc_rec.csv")

#rec_csv["time"] = class_variable_numerical_rec
#non_rec_csv["time"] = class_variable_numerical_non_rec

#non_rec_csv.to_csv('..\\wpbc\\csv\\wpbc_non.csv', index=False)
#rec_csv.to_csv('..\\wpbc\\csv\\wpbc_rec.csv', index=False)



#normalize and categorize columns
non_rec_7 = DI2.distribution_discretizer(non_rec.copy(), number_of_bins=7, cutoff_margin=0.0)
rec_7 = DI2.distribution_discretizer(rec.copy(), number_of_bins=7, cutoff_margin=0.0)
non_rec_5 = DI2.distribution_discretizer(non_rec.copy(), number_of_bins=5, cutoff_margin=0.0)
rec_5 = DI2.distribution_discretizer(rec.copy(), number_of_bins=5, cutoff_margin=0.0)
non_rec_3 = DI2.distribution_discretizer(non_rec.copy(), number_of_bins=3, cutoff_margin=0.0)
rec_3 = DI2.distribution_discretizer(rec.copy(), number_of_bins=3, cutoff_margin=0.0)

#write to arff and CSV file

arff.dump('..\\wpbc\\csv\\wpbc_non_3.arff', non_rec_3.values, names=non_rec_3.columns)
arff.dump('..\\wpbc\\csv\\wpbc_rec_3.arff', rec_3.values, names=rec_3.columns)
arff.dump('..\\wpbc\\csv\\wpbc_non_5.arff', non_rec_5.values, names=non_rec_5.columns)
arff.dump('..\\wpbc\\csv\\wpbc_rec_5.arff', rec_5.values, names=rec_5.columns)
arff.dump('..\\wpbc\\csv\\wpbc_non_7.arff', non_rec_7.values, names=non_rec_7.columns)
arff.dump('..\\wpbc\\csv\\wpbc_rec_7.arff', rec_7.values, names=rec_7.columns)

non_rec_7["time"] = class_variable_numerical_non_rec
rec_7["time"] = class_variable_numerical_rec
non_rec_5["time"] = class_variable_numerical_non_rec
rec_5["time"] = class_variable_numerical_rec
non_rec_3["time"] = class_variable_numerical_non_rec
rec_3["time"] = class_variable_numerical_rec

non_rec_3.to_csv('..\\wpbc\\csv\\wpbc_non_3.csv', index=False)
rec_3.to_csv('..\\wpbc\\csv\\wpbc_rec_3.csv', index=False)
non_rec_5.to_csv('..\\wpbc\\csv\\wpbc_non_5.csv', index=False)
rec_5.to_csv('..\\wpbc\\csv\\wpbc_rec_5.csv', index=False)
non_rec_7.to_csv('..\\wpbc\\csv\\wpbc_non_7.csv', index=False)
rec_7.to_csv('..\\wpbc\\csv\\wpbc_rec_7.csv', index=False)





