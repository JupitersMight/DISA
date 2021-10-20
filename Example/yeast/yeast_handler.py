import pandas as pd
from DISA.DISA import DISA


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


data = pd.read_csv("..\\yeast\\yeast.csv")

class_information = {
    "values": data["mit"],
    "outcome_value": 1,
    "type": "Numerical"
}

data = data.drop(columns=["mit"])

results = open("..\\yeast\\patterns.txt", "r")

number_of_biclusters_unfiltered = int(results.readline().split("#")[1])

json = {}
i=0
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
        j += 1
    if skip:
        continue
    i += 1

results.close()

patterns = []
count = 0
for item in json.keys():
    if json[item]["skip"]:
        continue
    else:
        count += 1
        patterns.append({
            "lines": json[item]["indexs"],
            "columns": json[item]["columns"],
            "column_values": json[item]["col_vals"],
            "type": "Constant"
        })

print("Total number of bics")
print(count)

discriminative_scores = DISA(data, patterns, class_information).assess_patterns(print_table=True)
