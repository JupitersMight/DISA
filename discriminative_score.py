import pandas as pd
import scipy.stats as scp
import scipy.stats
import scipy
import numpy as np
import math
import scipy.stats
from math import log2
from scipy.stats import hypergeom
from decimal import Decimal
import matplotlib.pyplot as plt
from scipy.stats import norm



class TempPower:

    #data = Dataframe
    # Inferir o column_values caso seja constante
    #Patterns [ { lines, columns, type, column_values, noise }, ...]
    #Outcome { values, outcome_value, type=Categorical}
    #Mostrar para o Lift maior caso não haja outcome_value
    def __init__(self, data, patterns, outcome, border_values=False):
        self.border_values = border_values
        self.data = data
        self.size_of_dataset = len(outcome["values"])
        self.y_column = outcome["values"]
        self.outcome_type = outcome["type"]

        self.y_value = outcome["outcome_value"] if "outcome_value" in list(outcome.keys()) else None
        # Check if numerical to binarize or categorical to determine the categories
        if outcome["type"] == "Numerical":
            self.unique_classes = [0, 1]
        else:
            self.unique_classes = []

            for value in outcome["values"].unique():
                if np.issubdtype(value, np.integer):
                    self.unique_classes.append(value)
                elif value.is_integer():
                    self.unique_classes.append(value)

        self.patterns = []
        for i in range(len(patterns)):
            column_values = patterns[i]["column_values"] if "column_values" in list(patterns[i].keys()) else None
            patterns[i]["lines"] = list(map(int, patterns[i]["lines"]))
            outcome_to_assess = self.y_value

            # If no column values then infer from data
            if column_values is None:
                column_values = []
                for col in patterns[i]["columns"]:
                    temp_array = []
                    for line in patterns[i]["lines"]:
                        temp_array.append(self.data.at[line, col])
                    column_values.append(np.median(temp_array))

            # If no noise inputted then all column contain 0 noise
            noise = patterns[i]["noise"] if "noise" in list(patterns[i].keys()) else None
            if noise is None:
                noise = []
                for col in patterns[i]["columns"]:
                    noise.append(0)

            # If no type then assume its a constant subspace
            type = patterns[i]["type"] if "type" in list(patterns[i].keys()) else "Constant"
            nr_cols = len(patterns[i]["columns"])
            x_space = outcome["values"].filter(axis=0, items=patterns[i]["lines"])
            _x_space = outcome["values"].drop(axis=0, labels=patterns[i]["lines"])
            x_data = data.drop(columns=data.columns.difference(patterns[i]["columns"])).filter(axis=0, items=patterns[i]["lines"])

            Cx = len(patterns[i]["lines"])
            C_x = self.size_of_dataset - Cx
            intervals = None
            if outcome["type"] == "Numerical":
                outcome_to_assess = 1
                intervals = self.handle_numerical_outcome(x_space)
                c1 = 0
                for value in outcome["values"]:
                    if intervals[0] <= float(value) <= intervals[1]:
                        c1 += 1
                Cy = c1
                C_y = self.size_of_dataset - Cy
                c1 = 0
                for value in x_space:
                    if intervals[0] <= float(value) <= intervals[1]:
                        c1 += 1
                Cxy = c1
                Cx_y = len(x_space) - Cxy
                c1 = 0
                for value in _x_space:
                    if intervals[0] <= float(value) <= intervals[1]:
                        c1 += 1
                C_xy = c1
                C_x_y = len(_x_space) - C_xy
            else:
                if outcome_to_assess is None:
                    maxLift = 0
                    discriminative_unique_class = 0
                    for unique_class in self.unique_classes:
                        testY = len(outcome["values"][outcome["values"] == unique_class])
                        omega = max(Cx + testY - 1, 1 / self.size_of_dataset)
                        v = 1 / max(Cx, testY)
                        testXY = len(x_space[x_space == unique_class])
                        if testXY == 0:
                            continue
                        lift_of_pattern = testXY / (Cx * testY)
                        curr_lift = (lift_of_pattern - omega) / (v - omega)
                        if curr_lift > maxLift:
                            maxLift = curr_lift
                            discriminative_unique_class = unique_class
                    outcome_to_assess = discriminative_unique_class

                Cy = len(outcome["values"][outcome["values"] == outcome_to_assess])
                Cxy = len(x_space[x_space == outcome_to_assess])
                C_xy = len(_x_space[_x_space == outcome_to_assess])
                Cx_y = len(x_space) - len(x_space[x_space == outcome_to_assess])
                C_x_y = len(_x_space) - len(_x_space[_x_space == outcome_to_assess])
                if border_values:
                    Cy += len(outcome["values"][outcome["values"] == outcome_to_assess-0.5]) \
                     + len(outcome["values"][outcome["values"] == outcome_to_assess+0.5])
                    Cxy += len(x_space[x_space == outcome_to_assess-0.5]) \
                      + len(x_space[x_space == outcome_to_assess+0.5])
                    C_xy = len(_x_space[_x_space == outcome_to_assess-0.5]) \
                           + len(_x_space[_x_space == outcome_to_assess+0.5])
                    Cx_y -= len(x_space[x_space == outcome_to_assess-0.5]) \
                           - len(x_space[x_space == outcome_to_assess+0.5])
                    C_x_y -= len(_x_space[_x_space == outcome_to_assess-0.5]) \
                            - len(_x_space[_x_space == outcome_to_assess+0.5])

                C_y = self.size_of_dataset - Cy

            X = Cx / self.size_of_dataset
            _X = 1 - X
            Y = Cy / self.size_of_dataset
            _Y = 1 - Y
            XY = Cxy / self.size_of_dataset
            _XY = C_xy / self.size_of_dataset
            X_Y = Cx_y / self.size_of_dataset
            _X_Y = C_x_y / self.size_of_dataset

            self.patterns.append({
                "outcome_to_assess": outcome_to_assess,
                "outcome_intervals": intervals,
                "columns": patterns[i]["columns"],
                "lines": patterns[i]["lines"],
                "nr_cols": nr_cols,
                "column_values": column_values,
                "noise": noise,
                "type": type,
                "x_space": x_space,
                "_x_space": _x_space,
                "x_data": x_data,
                "Cx": Cx,
                "C_x": C_x,
                "Cy": Cy,
                "C_y": C_y,
                "Cxy": Cxy,
                "C_xy": C_xy,
                "Cx_y": Cx_y,
                "C_x_y": C_x_y,
                "X": X,
                "_X": _X,
                "Y": Y,
                "_Y": _Y,
                "XY": XY,
                "_XY": _XY,
                "X_Y": X_Y,
                "_X_Y": _X_Y
            })

    def runAll(self):
        dict = {}
        for i in range(len(self.patterns)):
            information_gain = self.information_gain(i)
            #fisher_score = self.fisher_score(i)
            chi_squared = self.chi_squared(i)
            gini_index = self.gini_index(i)
            diff_sup = self.diff_sup(i)
            bigger_sup = self.bigger_sup(i)
            confidence = self.confidence(i)
            all_confidence = self.all_confidence(i)
            lift = self.lift(i)
            standardisation_of_lift = self.standardisation_of_lift(i)
            star_standardisation_of_lift = self.star_standardisation_of_lift(i)
            collective_strength = self.collective_strength(i)
            cosine = self.cosine(i)
            interestingness = self.interestingness(i)
            comprehensibility = self.comprehensibility(i)
            completeness = self.completeness(i)
            added_value = self.added_value(i)
            casual_confidence = self.casual_confidence(i)
            casual_support = self.casual_support(i)
            certainty_factor = self.certainty_factor(i)
            conviction = self.conviction(i)
            coverage = self.coverage(i)
            descriptive_confirmed_confidence = self.descriptive_confirmed_confidence(i)
            difference_of_confidence = self.difference_of_confidence(i)
            example_counter_example = self.example_counter_example(i)
            imbalance_ratio = self.imbalance_ratio(i)
            fishers_exact_test_p_value = self.fishers_exact_test_p_value(i)
            hyper_confidence = self.hyper_confidence(i)
            hyper_lift = self.hyper_lift(i)
            laplace_corrected_confidence = self.laplace_corrected_confidence(i)
            importance = self.importance(i)
            jaccard_coefficient = self.jaccard_coefficient(i)
            j_measure = self.j_measure(i)
            kappa = self.kappa(i)
            klosgen = self.klosgen(i)
            kulczynski = self.kulczynski(i)
            kruskal_lambda = self.kruskal_lambda(i)
            least_contradiction = self.least_contradiction(i)
            lerman_similarity = self.lerman_similarity(i)
            piatetsky_shapiro = self.piatetsky_shapiro(i)
            max_confidence = self.max_confidence(i)
            odds_ratio = self.odds_ratio(i)
            phi_correlation_coefficient = self.phi_correlation_coefficient(i)
            ralambondrainy_measure = self.ralambondrainy_measure(i)
            rld = self.rld(i)
            relative_risk = self.relative_risk(i)
            rule_power_factor = self.rule_power_factor(i)
            sebag = self.sebag(i)
            yule_q = self.yule_q(i)
            yule_y = self.yule_y(i)
            Wsup_pattern = self.Wsup_pattern(i) if "column_values" in list(self.patterns[i].keys()) else "Not enough information to calculate"
            Wsup_rule = self.Wsup_rule(i) if "column_values" in list(self.patterns[i].keys()) else "Not enough information to calculate"
            Wconf = self.Wconf(i) if "column_values" in list(self.patterns[i].keys()) else "Not enough information to calculate"
            WLift = self.WLift(i) if "column_values" in list(self.patterns[i].keys()) else "Not enough information to calculate"
            Tsig = self.Tsig(i) if "column_values" in list(self.patterns[i].keys()) else "Not enough information to calculate"
            FleBiC_score = self.FleBiC_score(i) if "column_values" in list(self.patterns[i].keys()) else "Not enough information to calculate"

            dict[i] = {
                "Outcome selected for analysis": self.patterns[i]["outcome_to_assess"],
                "Information Gain": information_gain,
                #"Fisher's score": fisher_score,
                "Chi-squared": chi_squared,
                "Gini index": gini_index,
                "Difference in Support": diff_sup,
                "Bigger Support": bigger_sup,
                "Confidence": confidence,
                "All-Confidence": all_confidence,
                "Lift": lift,
                "Standardised Lift": standardisation_of_lift,
                "Standardised Lift (with correction)": star_standardisation_of_lift,
                "Collective Strength": collective_strength,
                "Cosine": cosine,
                "Interestingness": interestingness,
                "Comprehensibility": comprehensibility,
                "Completeness": completeness,
                "Added Value": added_value,
                "Casual Confidence": casual_confidence,
                "Casual Support": casual_support,
                "Certainty Factor": certainty_factor,
                "Conviction": conviction,
                "Coverage (Support)": coverage,
                "Descriptive Confirmed Confidence": descriptive_confirmed_confidence,
                "Difference of Proportions": difference_of_confidence,
                "Example and Counter Example": example_counter_example,
                "Imbalance Ratio": imbalance_ratio,
                "Fisher's Exact Test (p-value)": fishers_exact_test_p_value,
                "Hyper Confidence": hyper_confidence,
                "Hyper Lift": hyper_lift,
                "Laplace Corrected Confidence": laplace_corrected_confidence,
                "Importance": importance,
                "Jaccard Coefficient": jaccard_coefficient,
                "J-Measure": j_measure,
                "Kappa": kappa,
                "Klosgen": klosgen,
                "Kulczynski": kulczynski,
                "Goodman-Kruskal's Lambda": kruskal_lambda,
                "Least Contradiction": least_contradiction,
                "Lerman Similarity": lerman_similarity,
                "Piatetsky-Shapiro": piatetsky_shapiro,
                "Max Confidence": max_confidence,
                "Odds Ratio": odds_ratio,
                "Phi Correlation Coefficient": phi_correlation_coefficient,
                "Ralambondrainy": ralambondrainy_measure,
                "Relative Linkage Disequilibrium": rld,
                "Relative Risk": relative_risk,
                "Rule Power Factor": rule_power_factor,
                "Sebag-Schoenauer": sebag,
                "Yule Q": yule_q,
                "Yule Y": yule_y,
                "Weighted Support": Wsup_pattern,
                "Weighted Rule Support": Wsup_rule,
                "Weighted Confidence": Wconf,
                "Weighted Lift": WLift,
                "Statistical Significance": Tsig,
                "FleBiC Score": FleBiC_score
            }
        return dict

    # IG (C|X) = H(C) - H(C|X)
    def information_gain(self, i):
        # H(C)
        #temp_sum = 0
        #for value in self.unique_classes:
        #    class_x = (len(self.y_column[self.y_column == value]) + len(self.y_column[self.y_column == value-0.5]) + len(self.y_column[self.y_column == value+0.5])) / self.size_of_dataset
        #    temp_sum += class_x * math.log(class_x, 10)
        #Hc = -temp_sum
        Hc = - (self.patterns[i]["Y"]*math.log(self.patterns[i]["Y"], 10) + self.patterns[i]["_Y"]*math.log(self.patterns[i]["_Y"],10))
        # H(C|X)
        #temp_sum = 0
        #for value in self.unique_classes:
        #    Pcix = (len(self.patterns[i]["x_space"][self.patterns[i]["x_space"] == value]) + len(self.patterns[i]["x_space"][self.patterns[i]["x_space"] == value-0.5]) + len(self.patterns[i]["x_space"][self.patterns[i]["x_space"] == value+0.5])) / self.size_of_dataset #/Px
        #    if Pcix == 0:
        #        temp_sum += 0
        #    else:
        #        temp_sum += Pcix*math.log(Pcix, 10)

        #Hcx = -self.patterns[i]["X"] * temp_sum
        one = 0
        two = 0
        if self.patterns[i]["XY"] != 0:
            one = self.patterns[i]["XY"]*math.log(self.patterns[i]["XY"], 10)
        if self.patterns[i]["X_Y"] != 0:
            two = self.patterns[i]["X_Y"]*math.log(self.patterns[i]["X_Y"],10)
        Hcx = -self.patterns[i]["X"] * (one + two)
        #temp_sum = 0
        #for value in self.unique_classes:
        #    Pcix = len(self.patterns[i]["_x_space"][self.patterns[i]["_x_space"] == value]) / self.size_of_dataset #/ NotPx
        #    if Pcix == 0:
        #        temp_sum += 0
        #    else:
        #        temp_sum += Pcix * math.log(Pcix, 10)
        #Hcx += -self.patterns[i]["_X"] * temp_sum
        three = 0
        four = 0
        if self.patterns[i]["_XY"] != 0:
            three = self.patterns[i]["_XY"] * math.log(self.patterns[i]["_XY"], 10)
        if self.patterns[i]["_X_Y"] != 0:
            four = self.patterns[i]["_X_Y"] * math.log(self.patterns[i]["_X_Y"], 10)

        Hcx = -self.patterns[i]["_X"] * (three + four)

        # esta ultima parte ainda é capaz de mudar vamos ver
        teta = self.patterns[i]["X"]
        p = self.patterns[i]["Y"]

        if self.patterns[i]["X"] < self.patterns[i]["Y"]:
            Hlbcx = (self.patterns[i]["X"] - 1) * ((((self.patterns[i]["Y"]-self.patterns[i]["X"])/self.patterns[i]["_X"]) * math.log((self.patterns[i]["Y"]-self.patterns[i]["X"])/self.patterns[i]["_X"], 10)) + ((self.patterns[i]["_Y"]/self.patterns[i]["_X"])*math.log(self.patterns[i]["_Y"]/self.patterns[i]["_X"],10)))
        elif self.patterns[i]["X"] == self.patterns[i]["Y"]:
            Hlbcx = (self.patterns[i]["X"] - 1) * ((self.patterns[i]["_Y"]/self.patterns[i]["_X"])*math.log(self.patterns[i]["_Y"]/self.patterns[i]["_X"],10))
        else:
            q1 = 1 - (1-p)/teta
            q2 = p/teta
            q = q1 if q1 > q2 else q2
            one = - teta*q*math.log(q,10)
            two = - teta*(1-q)*math.log(1-q,10)
            three = (teta*q-p)*math.log((p-teta*q)/(1-teta),10) if teta*q-p > 0 else 0
            four = (teta*(1-q)-(1-p))*math.log(((1-p)-teta*(1-q))/(1-teta),10) if (teta*(1-q)-(1-p)) > 0 else 0
            Hlbcx = one + two + three + four
            #Hlbcx = - teta*q*math.log(q,10) - teta*(1-q)*math.log(1-q,10) + (teta*q-p)*math.log((p-teta*q)/(1-teta),10) + (teta*(1-q)-(1-p))*math.log(((1-p)-teta*(1-q))/1-teta,10)

        return (Hc - Hcx)/(Hc - Hlbcx)

    '''
    def fisher_score(self, i):
        #if self.patterns[i]["X"] <= self.patterns[i]["Y"]:
        fr = self.patterns[i]["X"] * ((self.patterns[i]["Y"]-self.patterns[i]["XY"])**2) / ((self.patterns[i]["Y"]*self.patterns[i]["_Y"]*self.patterns[i]["_X"])-(self.patterns[i]["X"]*((self.patterns[i]["Y"]-self.patterns[i]["XY"])**2)))
        #fr_ub = (self.patterns[i]["X"] * self.patterns[i]["_Y"]) / (self.patterns[i]["Y"] - self.patterns[i]["X"])
        #fr_ub = (self.patterns[i]["X"]*(self.patterns[i]["Y"]**2))/(self.patterns[i]["Y"]*self.patterns[i]["_Y"]*self.patterns[i]["_X"]-(self.patterns[i]["X"]*(self.patterns[i]["Y"]**2)))
        #fr_ub = (self.patterns[i]["X"]*((self.patterns[i]["Y"]-1)**2))/(self.patterns[i]["Y"]*self.patterns[i]["_Y"]*self.patterns[i]["_X"]-(self.patterns[i]["X"]*((self.patterns[i]["Y"]-1)**2)))

        #else:
            #fr = self.patterns[i]["Y"] * ((self.patterns[i]["X"] - self.patterns[i]["XY"]) ** 2) / (
                        #(self.patterns[i]["X"] * self.patterns[i]["_X"] * self.patterns[i]["_Y"]) - (
                        #    self.patterns[i]["Y"] * ((self.patterns[i]["X"] - self.patterns[i]["XY"]) ** 2)))

            #fr_ub = (self.patterns[i]["Y"]*self.patterns[i]["_X"])/(self.patterns[i]["X"]-self.patterns[i]["Y"])

        return fr#/fr_ub
    '''

    def chi_squared(self, i):
        one=((self.patterns[i]["Cxy"]-(self.patterns[i]["Cx"]*self.patterns[i]["Cy"]/self.size_of_dataset))**2)/(self.patterns[i]["Cx"]*self.patterns[i]["Cy"]/self.size_of_dataset)
        two=((self.patterns[i]["C_xy"]-(self.patterns[i]["C_x"]*self.patterns[i]["Cy"]/self.size_of_dataset))**2)/(self.patterns[i]["C_x"]*self.patterns[i]["Cy"]/self.size_of_dataset)
        three=((self.patterns[i]["Cx_y"]-(self.patterns[i]["Cx"]*self.patterns[i]["C_y"]/self.size_of_dataset))**2)/(self.patterns[i]["Cx"]*self.patterns[i]["C_y"]/self.size_of_dataset)
        four=((self.patterns[i]["C_x_y"]-(self.patterns[i]["C_x"]*self.patterns[i]["C_y"]/self.size_of_dataset))**2)/(self.patterns[i]["C_x"]*self.patterns[i]["C_y"]/self.size_of_dataset)
        return one + two + three + four
        #return self.size_of_dataset * (((self.patterns[i]["XY"]*self.patterns[i]["_X_Y"])-(self.patterns[i]["X_Y"]*self.patterns[i]["_XY"]))/math.sqrt(self.patterns[i]["X"]*self.patterns[i]["Y"]*self.patterns[i]["_X"]*self.patterns[i]["_Y"]))

    def gini_index(self, i):
        return (self.patterns[i]["X"] * (((self.patterns[i]["XY"]/self.patterns[i]["X"])**2)+((self.patterns[i]["X_Y"]/self.patterns[i]["X"])**2)))\
               + (self.patterns[i]["_X"] * (((self.patterns[i]["_XY"]/self.patterns[i]["_X"])**2)+((self.patterns[i]["_X_Y"]/self.patterns[i]["_X"])**2)))\
               - (self.patterns[i]["Y"]**2) - (self.patterns[i]["_Y"]**2)

    def diff_sup(self, i):
        return abs((self.patterns[i]["XY"]/self.patterns[i]["Y"]) - (self.patterns[i]["X_Y"]/self.patterns[i]["_Y"]))

    def bigger_sup(self, i):
        return max((self.patterns[i]["XY"]/self.patterns[i]["Y"]), (self.patterns[i]["X_Y"]/self.patterns[i]["_Y"]))

    def confidence(self, i):
        return self.patterns[i]["XY"] / self.patterns[i]["X"]

    def all_confidence(self, i):
        return self.patterns[i]["XY"] / max(self.patterns[i]["X"], self.patterns[i]["Y"])

    def lift(self, i):
        return self.patterns[i]["XY"] / (self.patterns[i]["X"] * self.patterns[i]["Y"])

    def standardisation_of_lift(self, i):
        omega = max(self.patterns[i]["X"] + self.patterns[i]["Y"] - 1, 1/self.size_of_dataset)
        v = 1 / max(self.patterns[i]["X"], self.patterns[i]["Y"])
        return (self.lift(i)-omega)/(v-omega)

    def star_standardisation_of_lift(self, i):
        v = 1 / max(self.patterns[i]["X"], self.patterns[i]["Y"])
        s = v / self.size_of_dataset
        omega = max(
            max((self.patterns[i]["X"] + self.patterns[i]["Y"] - 1), 1/self.size_of_dataset)/(self.patterns[i]["X"] * self.patterns[i]["Y"]),
            (4*s/((1+s)**2)),
            (s/(self.patterns[i]["X"] * self.patterns[i]["Y"])),
            self.confidence(i)/self.patterns[i]["Y"]
        )
        if v - omega == 0:
            return math.inf
        else:
            return (self.lift(i)-omega)/(v-omega)

    def collective_strength(self, i):
        return (self.patterns[i]["XY"] + self.patterns[i]["_X_Y"] / self.patterns[i]["_X"]) / (self.patterns[i]["X"] * self.patterns[i]["Y"] + self.patterns[i]["_X"] * self.patterns[i]["_Y"])

    def cosine(self, i):
        return self.patterns[i]["XY"] / math.sqrt(self.patterns[i]["X"] * self.patterns[i]["Y"])

    def interestingness(self, i):
        return (self.patterns[i]["XY"] / self.patterns[i]["X"]) * (self.patterns[i]["XY"] / self.patterns[i]["Y"]) * (1 - (self.patterns[i]["XY"]/self.size_of_dataset))

    def comprehensibility(self, i):
        return np.log(1+1)/np.log(1+self.patterns[i]["nr_cols"]+1)

    def completeness(self, i):
        return self.patterns[i]["XY"] / self.patterns[i]["Y"]

    def added_value(self, i):
        return self.confidence(i) - (self.patterns[i]["Y"])

    def casual_confidence(self, i):
        return 0.5 * ((self.patterns[i]["XY"]/self.patterns[i]["X"]) + (self.patterns[i]["XY"]/self.patterns[i]["_X"]))

    def casual_support(self, i):
        return self.patterns[i]["XY"] + self.patterns[i]["_X_Y"]

    def certainty_factor(self, i):
        return ((self.patterns[i]["XY"] / self.patterns[i]["X"]) - self.patterns[i]["Y"])/self.patterns[i]["_Y"]

    def conviction(self, i):
        if self.patterns[i]["X_Y"] == 0:
            return math.inf
        else:
            return self.patterns[i]["X"] * self.patterns[i]["_Y"] / self.patterns[i]["X_Y"]

    def coverage(self, i):
        return self.patterns[i]["X"]

    def descriptive_confirmed_confidence(self, i):
        return (self.patterns[i]["XY"]/self.patterns[i]["X"]) - (self.patterns[i]["X_Y"]/self.patterns[i]["X"])

    def difference_of_confidence(self, i):
        return (self.patterns[i]["XY"] / self.patterns[i]["X"]) - (self.patterns[i]["_XY"] / self.patterns[i]["_X"])

    #Varia entre [-1,1]
    def example_counter_example(self, i):
        if self.patterns[i]["XY"] == 0:
            return "No intersection between subspace and outcome"
        return (self.patterns[i]["XY"] - self.patterns[i]["X_Y"]) / self.patterns[i]["XY"]

    def imbalance_ratio(self, i):
        if self.patterns[i]["XY"] == 0:
            return "No intersection between subspace and outcome"
        return abs((self.patterns[i]["XY"]/self.patterns[i]["X"])-(self.patterns[i]["XY"]/self.patterns[i]["Y"]))/((self.patterns[i]["XY"]/self.patterns[i]["X"])+(self.patterns[i]["XY"]/self.patterns[i]["Y"])-((self.patterns[i]["XY"]/self.patterns[i]["X"])*(self.patterns[i]["XY"]/self.patterns[i]["Y"])))

    def fishers_exact_test_p_value(self, i):
        # P( Cxy >= supXY)
        comb3 = math.factorial(self.size_of_dataset) // (math.factorial(self.patterns[i]["Cx"]) * math.factorial(self.size_of_dataset - self.patterns[i]["Cx"]))
        sum_Pcxy = 0
        for counter in range(0, self.patterns[i]["Cxy"]):
            comb1 = math.factorial(self.patterns[i]["Cy"])//(math.factorial(counter)*math.factorial(self.patterns[i]["Cy"]-counter))
            comb2_aux = (self.size_of_dataset-self.patterns[i]["Cy"])-(self.patterns[i]["Cx"]-counter)
            if comb2_aux < 0:
                comb2_aux = 0
            comb2 = math.factorial(self.size_of_dataset-self.patterns[i]["Cy"])//(math.factorial(self.patterns[i]["Cx"]-counter)*math.factorial(comb2_aux))
            sum_Pcxy += ((comb1*comb2)/comb3)
        return 1 - sum_Pcxy

    def hyper_confidence(self, i):
        return 1 - self.fishers_exact_test_p_value(i)

    def hyper_lift(self, i):
        [M, n, N] = [self.size_of_dataset, self.patterns[i]["Cy"], self.patterns[i]["Cx"]]
        ppf95 = hypergeom.ppf(0.95, M, n, N)
        return self.patterns[i]["Cxy"]/ppf95

    def laplace_corrected_confidence(self, i):
        return (self.patterns[i]["Cxy"]+1)/(self.patterns[i]["Cx"]+(len(self.unique_classes)))

    def importance(self, i):
        return math.log(((self.patterns[i]["Cxy"]+1)/(self.patterns[i]["Cx"]+len(self.unique_classes))) / ((self.patterns[i]["Cx_y"]+1)/(self.patterns[i]["Cx"]+len(self.unique_classes))), 10)

    def jaccard_coefficient(self, i):
        return self.patterns[i]["XY"]/(self.patterns[i]["X"]+self.patterns[i]["Y"]-self.patterns[i]["XY"])

    def j_measure(self, i):
        a = (self.patterns[i]["XY"]/self.patterns[i]["X"])/self.patterns[i]["Y"]
        if a == 0:
            a = 0
        else:
            a = self.patterns[i]["XY"] * math.log((self.patterns[i]["XY"]/self.patterns[i]["X"])/self.patterns[i]["Y"], 10)
        b = (self.patterns[i]["X_Y"]/self.patterns[i]["X"])/self.patterns[i]["_Y"]
        if b == 0:
            b = 0
        else:
            b = self.patterns[i]["X_Y"] * math.log((self.patterns[i]["X_Y"] / self.patterns[i]["X"]) / self.patterns[i]["_Y"], 10)
        return a + b

    def kappa(self, i):
        return (self.patterns[i]["XY"] + self.patterns[i]["_X_Y"]-(self.patterns[i]["X"] * self.patterns[i]["Y"])-(self.patterns[i]["_X"]*self.patterns[i]["_Y"])) / (1-(self.patterns[i]["X"]*self.patterns[i]["Y"])-(self.patterns[i]["_X"]*self.patterns[i]["_Y"]))

    def klosgen(self, i):
        return math.sqrt(self.patterns[i]["XY"])*((self.patterns[i]["XY"]/self.patterns[i]["X"])-self.patterns[i]["Y"])

    def kulczynski(self, i):
        return 0.5 * ((self.patterns[i]["XY"] / self.patterns[i]["X"]) + (self.patterns[i]["XY"] / self.patterns[i]["Y"]))

    def kruskal_lambda(self, i):
        one = max(self.patterns[i]["Cxy"], self.patterns[i]["C_xy"]) + max(self.patterns[i]["Cx_y"], self.patterns[i]["C_x_y"])
        two = max(self.patterns[i]["Cxy"], self.patterns[i]["Cx_y"]) + max(self.patterns[i]["C_xy"], self.patterns[i]["C_x_y"])
        three = max(self.patterns[i]["Cy"], self.patterns[i]["C_y"])
        four = max(self.patterns[i]["Cx"], self.patterns[i]["C_x"])
        return (one + two - three - four) / (2*self.size_of_dataset - three - four)

    def least_contradiction(self, i):
        return (self.patterns[i]["XY"] - self.patterns[i]["X_Y"]) / self.patterns[i]["Y"]

    #Resultado estranho
    def lerman_similarity(self, i):
        return (self.patterns[i]["Cxy"] - ((self.patterns[i]["Cx"] * self.patterns[i]["Cy"]) / self.size_of_dataset)) / math.sqrt((self.patterns[i]["Cx"] * self.patterns[i]["Cy"]) / self.size_of_dataset)
        #return math.sqrt(self.size_of_dataset) * (self.patterns[i]["XY"] - (self.patterns[i]["X"]*self.patterns[i]["Y"])) / math.sqrt(self.patterns[i]["X"] * self.patterns[i]["Y"])

    def piatetsky_shapiro(self, i):
        return self.patterns[i]["XY"] - (self.patterns[i]["X"] * self.patterns[i]["Y"])

    def max_confidence(self, i):
        return max(self.patterns[i]["XY"] / self.patterns[i]["X"], self.patterns[i]["XY"] / self.patterns[i]["Y"])

    def odds_ratio(self, i):
        if self.patterns[i]["X_Y"] == 0 or self.patterns[i]["_XY"] == 0:
            return math.inf
        else:
            return (self.patterns[i]["XY"] * self.patterns[i]["_X_Y"]) / (self.patterns[i]["X_Y"] * self.patterns[i]["_XY"])

    def phi_correlation_coefficient(self, i):
        return math.sqrt(self.chi_squared(i)/self.size_of_dataset)

    def ralambondrainy_measure(self, i):
        return self.patterns[i]["X_Y"]

    def rld(self, i):
        rld = 0
        d = (self.patterns[i]["Cxy"]*self.patterns[i]["C_x_y"])-(self.patterns[i]["Cx_y"]*self.patterns[i]["C_xy"])

        if d > 0:
            if self.patterns[i]["C_xy"] < self.patterns[i]["Cx_y"]:
                rld = d / (d+(self.patterns[i]["C_xy"] / self.size_of_dataset))
            else:
                rld = d / (d+(self.patterns[i]["Cx_y"] / self.size_of_dataset))
        else:
            if self.patterns[i]["Cxy"] < self.patterns[i]["C_x_y"]:
                rld = d / (d-(self.patterns[i]["Cxy"] / self.size_of_dataset))
            else:
                rld = d / (d-(self.patterns[i]["C_x_y"] / self.size_of_dataset))

        return rld

    def relative_risk(self, i):
        return (self.patterns[i]["XY"]/self.patterns[i]["X"])/(self.patterns[i]["_XY"]/self.patterns[i]["_X"])

    def rule_power_factor(self, i):
        return (self.patterns[i]["XY"]**2)/self.patterns[i]["X"]

    def sebag(self, i):
        if self.patterns[i]["X_Y"] == 0:
            return math.inf
        else:
            return self.patterns[i]["XY"]/self.patterns[i]["X_Y"]

    def yule_q(self, i):
        return (self.patterns[i]["XY"]*self.patterns[i]["_X_Y"] - self.patterns[i]["X_Y"]*self.patterns[i]["_XY"]) / (self.patterns[i]["XY"]*self.patterns[i]["_X_Y"] + self.patterns[i]["X_Y"]*self.patterns[i]["_XY"])

    def yule_y(self, i):
        return (math.sqrt(self.patterns[i]["XY"] * self.patterns[i]["_X_Y"]) - math.sqrt(self.patterns[i]["X_Y"] * self.patterns[i]["_XY"])) / (math.sqrt(self.patterns[i]["XY"] * self.patterns[i]["_X_Y"]) + math.sqrt(self.patterns[i]["X_Y"] * self.patterns[i]["_XY"]))

    def Wsup_pattern(self, i):
        counter = 0
        col_pos = 0
        for column in self.patterns[i]["columns"]:
            for row in self.patterns[i]["lines"]:
                column_value = self.patterns[i]["column_values"][col_pos]
                if pd.isna(self.data.at[row, column]) or not is_number(self.data.at[row, column]):
                    continue
                elif column_value - self.patterns[i]["noise"][col_pos] <= \
                        float(self.data.at[row, column]) <= \
                        column_value + self.patterns[i]["noise"][col_pos]:
                    counter += 1
                elif self.border_values:
                    if float(self.data.at[row, column]) == float(column_value) + 0.5 or\
                            float(self.data.at[row, column]) == float(column_value) - 0.5:
                        counter += 1
            col_pos += 1
        return counter / (len(self.data.columns) * len(list(self.data.index.values)))

    def Wsup_rule(self, i):
        counter = 0
        col_pos = 0
        for column in self.patterns[i]["columns"]:
            for row in self.patterns[i]["lines"]:
                skip = True
                if self.outcome_type == "Numerical":
                    intervals = self.patterns[i]["outcome_intervals"]
                    if intervals[0] <= float(self.y_column[row]) <= intervals[1]:
                        skip = False
                else:
                    if float(self.y_column[row]) == self.patterns[i]["outcome_to_assess"]:
                        skip = False
                if skip:
                    continue

                column_value = self.patterns[i]["column_values"][col_pos]
                if pd.isna(self.data.at[row, column]) or not is_number(self.data.at[row, column]):
                    continue
                elif column_value - self.patterns[i]["noise"][col_pos] <= \
                        float(self.data.at[row, column]) <= \
                        column_value + self.patterns[i]["noise"][col_pos]:
                    counter += 1
                elif self.border_values:
                    if float(self.data.at[row, column]) == float(column_value) + 0.5 or \
                            float(self.data.at[row, column]) == float(column_value) - 0.5:
                        counter += 1
            col_pos += 1
        return counter / (len(self.data.columns) * len(list(self.data.index.values)))

    def Wconf(self, i):
        return self.Wsup_rule(i) / self.Wsup_pattern(i)

    def WLift(self, i):
        return self.Wsup_rule(i) / (self.Wsup_pattern(i) * self.patterns[i]["Y"])

    # Constant
    def Tsig(self, i):
        p = 1.0
        # Constant
        if self.patterns[i]["type"] == "Constant":
            col_pos = 0
            for column in self.patterns[i]["columns"]:
                column_value = self.patterns[i]["column_values"][col_pos]
                counts = self.data[column]
                counter = 0
                for item in counts:
                    if self.border_values and (float(column_value) == item or float(column_value)+0.5 == item or float(column_value)-0.5 == item):
                        counter += 1
                    elif float(column_value) - self.patterns[i]["noise"][col_pos] <= item <= float(column_value) + self.patterns[i]["noise"][col_pos]:
                        counter += 1

                p = p * (counter/self.size_of_dataset)
                col_pos += 1
            return 1 - binomial_cumulative(self.patterns[i]["Cx"], p, self.size_of_dataset)
        elif self.patterns[i]["type"] == "Additive" or self.patterns[i]["type"] == "Multiplicative":
            for column in self.patterns[i]["columns"]:
                uniques_col = self.patterns[i]["x_data"][column].unique()
                counts = self.data[column]
                counter = 0
                for item in counts:
                    if self.border_values:
                        for unique_value in uniques_col:
                            if float(unique_value) == item or float(unique_value) + 0.5 == item or float(
                                    unique_value) - 0.5 == item:
                                counter += 1
                    elif not self.border_values:
                        for unique_value in uniques_col:
                            if float(unique_value) == item:
                                counter += 1

                p = p * (counter / self.size_of_dataset)
            return 1 - binomial_cumulative(self.patterns[i]["Cx"], p, self.size_of_dataset)
        elif self.patterns[i]["type"] == "Order-Preserving":
            p = float(Decimal(1.0) / Decimal(math.factorial(self.patterns[i]["nr_cols"])))
            nr_rows_bic = self.patterns[i]["Cx"]
            #Contar todos os valores que não são missings no dataset inteiro = A
            counter = 0
            for column in self.data.columns:
                for row in self.data[column]:
                    if is_number(row):
                        counter += 1
            #Contar o numero de linhas que têm valores (não missing) superior ao numero de colunas do padrão = B
            row_counter = 0
            for row in range(self.size_of_dataset):
                aux_counter = 0
                for values in self.data.iloc[row,:]:
                    if is_number(values):
                        aux_counter += 1
                if aux_counter > self.patterns[i]["nr_cols"]:
                    row_counter += 1
            #Calcular percentagem de missings presentes no dataset =  A / (total_nr_cols * total_nr_rows) = C
            percentage_missings = counter // (len(self.data.columns)*self.size_of_dataset)
            #Calcular  B * C = row_counter*percentage_missings
            return 1 - binomial_cumulative(self.patterns[i]["Cx"], p, row_counter*percentage_missings)

    def FleBiC_score(self, i):
        part1 = 0.6 * 0.7 * ((self.Wsup_rule(i)/self.Wsup_pattern(i)) * (self.patterns[i]["Y"]/self.size_of_dataset) + 0.3 * self.chi_squared(i))
        part2 = 0.3 * 0.5 * ((self.Wsup_rule(i)/self.size_of_dataset) * (self.patterns[i]["Y"]/self.size_of_dataset) + 0.5 * (self.patterns[i]["nr_cols"]/len(self.data.columns)))
        part3 = 0.1 * self.patterns[i]["percentage_of_noise"]
        return self.Tsig(i) * (part1 + part2 + part3)

    def handle_numerical_outcome(self, x_space):
        m1 = x_space.map(float).mean()
        m2 = self.y_column.map(float).mean()
        std1 = x_space.map(float).std()
        std2 = self.y_column.map(float).std()

        # Solve for a gaussian
        a= -1/(std1**2) + 1/(std2**2)
        b= 2*(-m2/(std2**2) + m1/(std1**2))
        c= (m2**2)/(std2**2) - (m1**2)/(std1**2) + np.log((std2**2)/(std1**2))
        intercep_points = np.roots([a, b, c])
        idxs = sorted(intercep_points)
        ##########
        # x-axis ranges from -5 and 5 with .001 steps
        x = np.arange(-1, 1, 0.001)
        # define multiple normal distributions
        plt.plot(x, norm.pdf(x, m1, std1), label='μ:'+str(round(m1,2))+ ', σ:' + str(round(std1,2)))
        plt.plot(x, norm.pdf(x, m2, std2), label='μ:'+str(round(m2,2))+ ', σ:' + str(round(std2,2)))
        plt.axvline(x=idxs[0],color="red", linestyle='--')
        plt.axvline(x=idxs[1],color="red", linestyle='--')
        plt.legend()
        plt.show()
        print(idxs)
        ############
        if len(idxs) == 1 or idxs[0] == idxs[1]:
            return [idxs[0] - std1*2, idxs[0] + std1*2]
        else:
            return [idxs[0], idxs[1]]


def binomial_cumulative(x, p, n):
    sum = Decimal(0)
    for i in range(x+1):
        comb1 = math.factorial(i)*math.factorial(n-i)
        comb2 = math.factorial(n)
        part1 = comb2 // comb1
        part2 = (p**i)
        part3 = ((1-p)**(n-i))
        sum = sum + (Decimal(part1) * Decimal(part2) * Decimal(part3))
    return float(sum)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

