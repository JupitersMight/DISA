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

input = pd.read_csv("..\\dodecanol\\input.csv")
transposed_input = pd.DataFrame(data=np.zeros((238, 6), dtype=float), columns=["AHR_ECOLI", "LCFA_ECOLI", "Dodecanoyl-[acyl-carrier-protein] hydrolase, chloroplastic", "Fatty acyl-CoA reductase", "A1U2T0", "A1U3L3"])
i = -1
for index, row in input.iterrows():
    if index % 6 == 0:
        i += 1
    transposed_input.iloc[i, transposed_input.columns.get_loc(row["names"])] = row["values"]

assay_name = []
for index, row in input.iterrows():
    if row["assay_name"] not in assay_name:
        assay_name.append(row["assay_name"])

transposed_input = pd.concat([transposed_input, pd.DataFrame(data=assay_name,columns=["assay_name"])], axis=1)

outcome = pd.read_csv("..\\dodecanol\\outcome_class.csv")

outcome = outcome[outcome["names"] == "dodecan-1-ol"].set_index("assay_name")["values"]

transposed_input = transposed_input.set_index('assay_name').join(outcome)

temp_class = transposed_input["values"].copy()

#transposed_input = transposed_input.drop(columns=["values"])

#normalize and categorize columns
transposed_input_7 = DI2.distribution_discretizer(transposed_input.copy(), number_of_bins=7, cutoff_margin=0.0)
transposed_input_5 = DI2.distribution_discretizer(transposed_input.copy(), number_of_bins=5, cutoff_margin=0.0)
transposed_input_3 = DI2.distribution_discretizer(transposed_input.copy(), number_of_bins=3, cutoff_margin=0.0)

#write to arff and CSV file

arff.dump('..\\dodecanol\\csv\\dodecanol_7.arff', transposed_input_7.values, names=transposed_input_7.columns)
arff.dump('..\\dodecanol\\csv\\dodecanol_5.arff', transposed_input_5.values, names=transposed_input_5.columns)
arff.dump('..\\dodecanol\\csv\\dodecanol_3.arff', transposed_input_3.values, names=transposed_input_3.columns)

transposed_input_3["values"] = temp_class
transposed_input_5["values"] = temp_class
transposed_input_7["values"] = temp_class

transposed_input_3.to_csv('..\\dodecanol\\csv\\modified_input_3.csv', index=False)
transposed_input_5.to_csv('..\\dodecanol\\csv\\modified_input_5.csv', index=False)
transposed_input_7.to_csv('..\\dodecanol\\csv\\modified_input_7.csv', index=False)

'''

'''
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt

# 300 represents number of points to make between T.min and T.max
T = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
power_1 = [2,2,2,2,4,5,4,2,2,2,2,2,4,5,4,2,2,2]
power_2 =[1,1,1,1,5,6,5,1,1,1,1,1,5,6,5,1,1,1,]
xnew = np.linspace(1, 18, 300)

spl = make_interp_spline(T, power_1, k=3)  # type: BSpline
power_smooth = spl(xnew)

plt.plot(xnew, power_smooth)

spl = make_interp_spline(T, power_2, k=3)  # type: BSpline
power_smooth = spl(xnew)

plt.plot(xnew, power_smooth)
plt.show()



