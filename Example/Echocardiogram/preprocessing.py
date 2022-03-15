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



col_names = ["survival","still-alive", "age-at-heart-attack", "pericardial-effusion", "fractional-shortening", "epss", "lvdd", "wall-motion-score", "wall-motion-index", "mult", "name", "group", "alive-at-1"]
numerical_cols = ["age-at-heart-attack", "fractional-shortening", "epss", "lvdd", "wall-motion-index"]
non_numerical_cols = ["pericardial-effusion"]
dropped_cols = ["alive-at-1", "group", "name", "mult", "wall-motion-score", "still-alive"]
class_variable = "survival"

data = []
# Read data
echo_data = pd.read_csv("..\\Echocardiogram\\input.csv")

#Drop uninformative columns
echo_data = echo_data.drop(dropped_cols, axis=1)
class_variable_numerical = echo_data["survival"].copy()

#normalize and categorize columns
temp_holder = echo_data["pericardial-effusion"]
echo_data = echo_data.drop(["pericardial-effusion"], axis=1)
for column in echo_data.columns:
    echo_data[column] = echo_data[column].replace("?", np.nan).replace("",np.nan)
echo_data_7 = DI2.distribution_discretizer(echo_data.copy(), number_of_bins=7, cutoff_margin=0.0)
echo_data_5 = DI2.distribution_discretizer(echo_data.copy(), number_of_bins=5, cutoff_margin=0.0)
echo_data_3 = DI2.distribution_discretizer(echo_data.copy(), number_of_bins=3, cutoff_margin=0.0)
echo_data_3_arff = pd.concat([echo_data_3, temp_holder], axis=1)
echo_data_5_arff = pd.concat([echo_data_5, temp_holder], axis=1)
echo_data_7_arff = pd.concat([echo_data_7, temp_holder], axis=1)

#write to arff and CSV file

arff.dump('..\\Echocardiogram\\csv\\echocardiogram_3.arff', echo_data_3_arff.values, names=echo_data_3_arff.columns)
arff.dump('..\\Echocardiogram\\csv\\echocardiogram_5.arff', echo_data_5_arff.values, names=echo_data_5_arff.columns)
arff.dump('..\\Echocardiogram\\csv\\echocardiogram_7.arff', echo_data_7_arff.values, names=echo_data_7_arff.columns)
echo_data_3_arff["survival"] = class_variable_numerical
echo_data_5_arff["survival"] = class_variable_numerical
echo_data_7_arff["survival"] = class_variable_numerical
echo_data_3_arff.to_csv('..\\Echocardiogram\\csv\\echocardiogram_3.csv', index=False)
echo_data_5_arff.to_csv('..\\Echocardiogram\\csv\\echocardiogram_5.csv', index=False)
echo_data_7_arff.to_csv('..\\Echocardiogram\\csv\\echocardiogram_7.csv', index=False)


