import pandas as pd
from DISA.DISA import DISA
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def retrive_patterns(file):
    results = open(file, "r", encoding="utf-8")

    number_of_biclusters_unfiltered = int(results.readline().split("#")[1])

    json = {}
    i = 0
    bic = 0
    while bic < number_of_biclusters_unfiltered:
        bic += 1
        line = results.readline().split("Y=[")
        values_of_cols = line[0].split("I=[")[1].split("]")[0]
        cols = line[1].split("]")[0]
        index = line[1].split("X=[")[1].split("]")[0]
        p_value = line[1].split("pvalue=")[1].split("(")[0].split("Lifts=[")[0]
        json[i] = {}
        json[i]["columns"] = cols.split(",")
        json[i]["col_vals"] = values_of_cols.split(",")
        json[i]["indexs"] = index.split(",")
        json[i]["pvalue"] = float(p_value)
        json[i]["skip"] = False if float(p_value) <= 0.05 else True

        j = 1
        skip = False
        while j < i:
            if json[j]["columns"] == json[i]["columns"] and json[j]["col_vals"] == json[i]["col_vals"] and json[i]["indexs"] == json[j]["indexs"]:
                skip = True
            if len(json[i]["indexs"]) <= 2:
                skip = True
            j += 1
        if skip:
            continue
        i += 1

    results.close()

    patterns = []
    count = 0
    number_of_cols = []
    number_of_rows = []
    for item in json.keys():
        if json[item]["skip"]:
            continue
        else:
            count += 1
            number_of_cols.append(len(json[item]["columns"]))
            number_of_rows.append(len(json[item]["indexs"]))
            patterns.append({
                "lines": json[item]["indexs"],
                "columns": json[item]["columns"],
                "column_values": json[item]["col_vals"],
                "type": "Constant"
            })

    print("Total number of bics")
    print(count)
    print("Average number of columns")
    print(np.average(number_of_cols))
    print("Standard deviation of columns")
    print(np.std(number_of_cols))
    print("Average number of rows")
    print(np.average(number_of_rows))
    print("Standard deviation of rows")
    print(np.std(number_of_rows))

    return patterns


def stats(data, patterns, class_information):
    discriminative_scores = DISA(data, patterns, class_information).assess_patterns(print_table=True)

    information_gain = []
    gini_index = []
    chi_squared = []
    lift = []
    std_lift = []
    stat_sig = []
    for dictionary in discriminative_scores:
        information_gain.append(dictionary["Information Gain"])
        gini_index.append(dictionary["Gini index"])
        chi_squared.append(dictionary["Chi-squared"])
        lift.append(dictionary["Lift"])
        std_lift.append(dictionary["Standardised Lift"])
        stat_sig.append(dictionary["Statistical Significance"])

    print("Average Information Gain")
    print(np.average(information_gain))
    print("Standard deviation of Information Gain")
    print(np.std(information_gain))
    print("Average Gini Index")
    print(np.average(gini_index))
    print("Standard deviation of Gini Index")
    print(np.std(gini_index))
    print("Average Chi-Squared")
    print(np.average(chi_squared))
    print("Standard deviation of Chi-Squared")
    print(np.std(chi_squared))
    print("Average Lift")
    print(np.average(lift))
    print("Standard deviation of Lift")
    print(np.std(lift))
    print("Average Standardised Lift")
    print(np.average(std_lift))
    print("Standard deviation of Standardised Lift")
    print(np.std(std_lift))
    print("Average Statistical Significance")
    print(np.average(stat_sig))
    print("Standard deviation of Statistical Significance")
    print(np.std(stat_sig))

