import pandas as pd
import numpy as np
import math
from scipy.stats import hypergeom
from scipy.stats import norm
from prettytable import PrettyTable
from scipy.special import betainc
import matplotlib.pyplot as plt
import os
import scipy


class DISA:
    """
        A class to analyse the subspaces inputted for their analysis

        Parameters
        ----------
        data : pandas.Dataframe
        patterns : list
            [i] : dict
                "lines" : list (mandatory)
                "columns" : list (mandatory)
                "column_values": list (optional)
                "noise": list (optional)
                "type" : string (optional)

        outcome : dict
            "values": pandas.Series
            "outcome_value" : int
            "type": string
            "method": string
            "heuristic": boolean

        output_configurations : dict
            "print_table" : boolean
            "file_path_table" : string
            "show_plots" : boolean
            "file_path_plots" : string
            "print_numeric_intervals" : boolean
            "file_path_numeric_intervals" : string


        Class Attributes
        ----------------
        border_values : boolean
        data : pandas.Dataframe
        file_path_numeric_intervals : string
        file_path_plots : string
        file_path_table : string
        outcome_type : string
        patterns : dict
            Contains all the auxiliary information needed by the metrics
        print_numeric_intervals : boolean
        print_table : boolean
        show_plots : boolean
        size_of_dataset : int
        unique_classes : list
            If input is categorical contains the unique classes within the y_column
        y_column : pandas.Series
    """
    def __init__(self, data, patterns, outcome, output_configurations=None):
        # Setup default output configuration in case user didn't input it
        self.print_table = output_configurations["print_table"] if output_configurations is not None and "print_table" in output_configurations.keys() else True
        self.file_path_table = output_configurations["file_path_table"] if output_configurations is not None and "file_path_table" in output_configurations.keys() else None
        self.show_plots = output_configurations["show_plots"] if output_configurations is not None and "show_plots" in output_configurations.keys() else False
        self.file_path_plots = output_configurations["file_path_plots"] if output_configurations is not None and "file_path_plots" in output_configurations.keys() else None
        self.print_numeric_intervals = output_configurations["print_numeric_intervals"] if output_configurations is not None and "print_numeric_intervals" in output_configurations.keys() else False
        self.file_path_numeric_intervals = output_configurations["file_path_numeric_intervals"] if output_configurations is not None and "file_path_numeric_intervals" in output_configurations.keys() else None

        # Set up default parameters
        self.border_values = outcome["border_values"] if "border_values" in outcome.keys() else False
        self.data = data
        self.size_of_dataset = len(outcome["values"])
        self.y_column = outcome["values"]
        self.outcome_type = outcome["type"]
        self.heuristic = outcome["heuristic"] if "heuristic" in outcome.keys() else False

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
            if column_values is not None:
                col_values_counter = 0
                for value in column_values:
                    column_values[col_values_counter] = float(value)
                    col_values_counter += 1
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
                noise_aux = []
                for col in patterns[i]["columns"]:
                    noise_aux.append(0)
                noise = noise_aux

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
                intervals = self.handle_numerical_outcome(x_space, (outcome["method"] if "method" in outcome.keys() else "gaussian"), i)
                c1 = 0
                for value in outcome["values"]:
                    for interval in intervals:
                        if interval[0] <= float(value) <= interval[1]:
                            c1 += 1
                Cy = c1
                C_y = self.size_of_dataset - Cy
                c1 = 0
                for value in x_space:
                    for interval in intervals:
                        if interval[0] <= float(value) <= interval[1]:
                            c1 += 1
                Cxy = c1
                Cx_y = len(x_space) - Cxy
                c1 = 0
                for value in _x_space:
                    for interval in intervals:
                        if interval[0] <= float(value) <= interval[1]:
                            c1 += 1
                C_xy = c1
                C_x_y = len(_x_space) - C_xy
            else:
                if outcome_to_assess is None:
                    maxLift = 0
                    discriminative_unique_class = 0
                    for unique_class in self.unique_classes:
                        testY = len(outcome["values"][outcome["values"] == unique_class])
                        testXY = len(x_space[x_space == unique_class])
                        if testXY == 0:
                            continue
                        curr_lift = (testXY / (Cx * testY)) * self.size_of_dataset
                        if curr_lift > maxLift:
                            maxLift = curr_lift
                            discriminative_unique_class = unique_class
                    outcome_to_assess = discriminative_unique_class

                Cy = len(outcome["values"][outcome["values"] == outcome_to_assess])
                Cxy = len(x_space[x_space == outcome_to_assess])
                C_xy = len(_x_space[_x_space == outcome_to_assess])
                Cx_y = len(x_space) - len(x_space[x_space == outcome_to_assess])
                C_x_y = len(_x_space) - len(_x_space[_x_space == outcome_to_assess])
                if self.border_values:
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

    def assess_patterns(self):
        """
        Executes all the subspace metrics for the inputted patterns

        Parameters
        ----------
        print_table : boolean
            If true, prints a table containing the metric values
        Returns
        -------
        list
            [x] : dictionary :
                "Outcome selected for analysis", "Information Gain", "Chi-squared", "Gini index", "Difference in Support",
                "Bigger Support", "Confidence", "All-Confidence", "Lift", "Standardised Lift", "Standardised Lift (with correction)",
                "Collective Strength", "Cosine", "Interestingness", "Comprehensibility", "Completeness", "Added Value",
                "Casual Confidence", "Casual Support", "Certainty Factor", "Conviction", "Coverage (Support)",
                "Descriptive Confirmed Confidence", "Difference of Proportions", "Example and Counter Example",
                "Imbalance Ratio", "Fisher's Exact Test (p-value)", "Hyper Confidence", "Hyper Lift", "Laplace Corrected Confidence",
                "Importance", "Jaccard Coefficient", "J-Measure", "Kappa", "Klosgen", "Kulczynski", "Goodman-Kruskal's Lambda",
                "Least Contradiction", "Lerman Similarity", "Piatetsky-Shapiro", "Max Confidence", "Odds Ratio",
                "Phi Correlation Coefficient", "Ralambondrainy", "Relative Linkage Disequilibrium", "Relative Risk"
                "Rule Power Factor", "Sebag-Schoenauer", "Yule Q", "Yule Y", "Weighted Support", "Weighted Rule Support"
                "Weighted Confidence", "Weighted Lift", "Statistical Significance", "FleBiC Score"

            where "x" represents the position of a subspace, and the dictionary the corresponding metrics calculated for
            the subspace. More details about the metrics are given in the methods.
        """
        dict = []
        for i in range(len(self.patterns)):
            information_gain = self.information_gain(i)
            chi_squared = self.chi_squared(i)
            gini_index = self.gini_index(i)
            diff_sup = self.diff_sup(i)
            bigger_sup = self.bigger_sup(i)
            confidence = self.confidence(i)
            all_confidence = self.all_confidence(i)
            lift = self.lift(i)
            standardisation_of_lift = self.standardisation_of_lift(i)
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

            dict.append({
                "Outcome selected for analysis": self.patterns[i]["outcome_to_assess"],
                "Information Gain": information_gain,
                "Chi-squared": chi_squared,
                "Gini index": gini_index,
                "Difference in Support": diff_sup,
                "Bigger Support": bigger_sup,
                "Confidence": confidence,
                "All-Confidence": all_confidence,
                "Lift": lift,
                "Standardised Lift": standardisation_of_lift,
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
            })
        if self.file_path_numeric_intervals is not None:
            for i in range(len(self.patterns)):
                if os.path.isfile(self.file_path_numeric_intervals+str(i)+'.txt'):
                    file = open(self.file_path_numeric_intervals+str(i)+'.txt', "w")
                else:
                    file = open(self.file_path_numeric_intervals+str(i)+'.txt', "x")
                file.write("P"+str(i)+": "+str(self.patterns[i]["outcome_intervals"])+"\n")
                file.close()
        if self.print_numeric_intervals:
            for i in range(len(self.patterns)):
                print("P"+ str(i)+ ": "+ str(self.patterns[i]["outcome_intervals"]))
        if self.print_table:
            columns = ['Metric']
            for i in range(len(self.patterns)):
                columns.append('P'+str(i+1))
            t = PrettyTable(columns)
            for metric in list(dict[0].keys()):
                line = [metric]
                for x in range(len(self.patterns)):
                    if is_number(dict[x][metric]):
                        temp = round(dict[x][metric], 2)
                        if temp == 0.00:
                            temp = '{:0.2e}'.format(dict[x][metric])
                            if float(temp) == 0.00:
                                temp = round(dict[x][metric], 2)
                        line.append(str(temp))
                    else:
                        line.append(dict[x][metric])
                t.add_row(line)
            print(t)

        return dict

    def information_gain(self, i):
        """ Calculates information gain of the subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Information gain of subspace
        """
        one = self.patterns[i]["XY"]*math.log(self.patterns[i]["XY"]/(self.patterns[i]["X"]*self.patterns[i]["Y"]), 10) if self.patterns[i]["XY"] != 0 else 0
        two = self.patterns[i]["X_Y"]*math.log(self.patterns[i]["X_Y"]/(self.patterns[i]["X"]*self.patterns[i]["_Y"]), 10) if self.patterns[i]["X_Y"] != 0 else 0
        three = self.patterns[i]["_XY"]*math.log(self.patterns[i]["_XY"]/(self.patterns[i]["_X"]*self.patterns[i]["Y"]),10) if self.patterns[i]["_XY"] != 0 else 0
        four = self.patterns[i]["_X_Y"]*math.log(self.patterns[i]["_X_Y"]/(self.patterns[i]["_X"]*self.patterns[i]["_Y"]), 10) if self.patterns[i]["_X_Y"] != 0 else 0
        frac_up = one + two + three + four
        frac_down_one = - (self.patterns[i]["X"] * math.log(self.patterns[i]["X"],10) + self.patterns[i]["_X"] * math.log(self.patterns[i]["_X"], 10)) if self.patterns[i]["X"] != 0 and self.patterns[i]["_X"] != 0 else 0
        frac_down_two = - (self.patterns[i]["Y"] * math.log(self.patterns[i]["Y"],10) + self.patterns[i]["_Y"] * math.log(self.patterns[i]["_Y"], 10)) if self.patterns[i]["Y"] != 0 and self.patterns[i]["_Y"] != 0 else 0
        frac_down = min(frac_down_one, frac_down_two)
        if frac_down == 0:
            return 0
        return frac_up / frac_down

    def chi_squared(self, i):
        """ Calculates the Chi-squared test statistic given a subspace
        https://doi.org/10.1145/253260.253327
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Chi-squared test statistic of subspace
        """
        if self.patterns[i]["C_y"] == 0:
            return 0
        one=((self.patterns[i]["Cxy"]-(self.patterns[i]["Cx"]*self.patterns[i]["Cy"]/self.size_of_dataset))**2)/(self.patterns[i]["Cx"]*self.patterns[i]["Cy"]/self.size_of_dataset)
        two=((self.patterns[i]["C_xy"]-(self.patterns[i]["C_x"]*self.patterns[i]["Cy"]/self.size_of_dataset))**2)/(self.patterns[i]["C_x"]*self.patterns[i]["Cy"]/self.size_of_dataset)
        three=((self.patterns[i]["Cx_y"]-(self.patterns[i]["Cx"]*self.patterns[i]["C_y"]/self.size_of_dataset))**2)/(self.patterns[i]["Cx"]*self.patterns[i]["C_y"]/self.size_of_dataset)
        four=((self.patterns[i]["C_x_y"]-(self.patterns[i]["C_x"]*self.patterns[i]["C_y"]/self.size_of_dataset))**2)/(self.patterns[i]["C_x"]*self.patterns[i]["C_y"]/self.size_of_dataset)
        return one + two + three + four

    def gini_index(self, i):
        """ Calculates the gini index metric of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Gini index of subspace
        """
        return (self.patterns[i]["X"] * (((self.patterns[i]["XY"]/self.patterns[i]["X"])**2)+((self.patterns[i]["X_Y"]/self.patterns[i]["X"])**2)))\
               + (self.patterns[i]["_X"] * (((self.patterns[i]["_XY"]/self.patterns[i]["_X"])**2)+((self.patterns[i]["_X_Y"]/self.patterns[i]["_X"])**2)))\
               - (self.patterns[i]["Y"]**2) - (self.patterns[i]["_Y"]**2)

    def diff_sup(self, i):
        """ Calculates difference of support metric of a given subspace
        DOI 10.1109/TKDE.2010.241
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Difference in support of subspace
        """
        if self.patterns[i]["_Y"] == 0:
            return 0
        return abs((self.patterns[i]["XY"]/self.patterns[i]["Y"]) - (self.patterns[i]["X_Y"]/self.patterns[i]["_Y"]))

    def bigger_sup(self, i):
        """ Calculates bigger support metric of a given subspace
        DOI 10.1109/TKDE.2010.241
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Bigger support of subspace
        """
        if self.patterns[i]["_Y"] == 0:
            return 0
        return max((self.patterns[i]["XY"]/self.patterns[i]["Y"]), (self.patterns[i]["X_Y"]/self.patterns[i]["_Y"]))

    def confidence(self, i):
        """ Calculates the confidence of a given subspace
        DOI 10.1145/170036.170072
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Confidence of subspace
        """
        return self.patterns[i]["XY"] / self.patterns[i]["X"]

    def all_confidence(self, i):
        """ Calculates the all confidence metric of a given subspace
        DOI 10.1109/TKDE.2003.1161582
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            All confidence of subspace
        """
        return self.patterns[i]["XY"] / max(self.patterns[i]["X"], self.patterns[i]["Y"])

    def lift(self, i):
        """ Calculates the lift metric of a given subspace
        DOI 10.1145/170036.170072
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Lift of subspace
        """
        return self.patterns[i]["XY"] / (self.patterns[i]["X"] * self.patterns[i]["Y"])

    def standardisation_of_lift(self, i):
        """ Calculates the standardized version of lift metric of a given subspace
        https://doi.org/10.1016/j.csda.2008.03.013
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Standardized lift of subspace
        """
        omega = max(self.patterns[i]["X"] + self.patterns[i]["Y"] - 1, 1/self.size_of_dataset)/(self.patterns[i]["X"] * self.patterns[i]["Y"])
        v = 1 / max(self.patterns[i]["X"], self.patterns[i]["Y"])
        return (self.lift(i)-omega)/(v-omega)

    def collective_strength(self, i):
        """ Calculates the collective strength metric of a given subspace
        https://dl.acm.org/doi/pdf/10.1145/275487.275490
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Collective strength of subspace
        """
        return (self.patterns[i]["XY"] + self.patterns[i]["_X_Y"] / self.patterns[i]["_X"]) / (self.patterns[i]["X"] * self.patterns[i]["Y"] + self.patterns[i]["_X"] * self.patterns[i]["_Y"])

    def cosine(self, i):
        """ Calculates cosine metric of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Cosine of subspace
        """
        return self.patterns[i]["XY"] / math.sqrt(self.patterns[i]["X"] * self.patterns[i]["Y"])

    def interestingness(self, i):
        """ Calculates interestingness metric of a given subspace
        arXiv:1202.3215
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Interestingness of subspace
        """
        return (self.patterns[i]["XY"] / self.patterns[i]["X"]) * (self.patterns[i]["XY"] / self.patterns[i]["Y"]) * (1 - (self.patterns[i]["XY"]/self.size_of_dataset))

    def comprehensibility(self, i):
        """ Calculates the compregensibility metric of a given subspace
	    arXiv:1202.3215
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Interestingness of subspace
        """
        return np.log(1+1)/np.log(1+self.patterns[i]["nr_cols"]+1)

    def completeness(self, i):
        """ Calculates the completeness metric of a given
    	arXiv:1202.3215
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Completeness of subspace
        """
        return self.patterns[i]["XY"] / self.patterns[i]["Y"]

    def added_value(self, i):
        """ Calculates the added value metric of a subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Added value of subspace
        """
        return self.confidence(i) - (self.patterns[i]["Y"])

    def casual_confidence(self, i):
        """ Calculates casual confidence metric of a given subspace
        https://doi.org/10.1007/3-540-44673-7_1
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Casual confidence of subspace
        """
        return 0.5 * ((self.patterns[i]["XY"]/self.patterns[i]["X"]) + (self.patterns[i]["XY"]/self.patterns[i]["_X"]))

    def casual_support(self, i):
        """ Calculates the casual support metric of a given subspace
        https://doi.org/10.1007/3-540-44673-7_1
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Casual support of subspace
        """
        return self.patterns[i]["XY"] + self.patterns[i]["_X_Y"]

    def certainty_factor(self, i):
        """ Calculates the certainty factor metric of a given subspace
        DOI 10.3233/IDA-2002-6303
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Certainty factor metric of a given subspace
        """
        if self.patterns[i]["_Y"] == 0:
            return 0
        return ((self.patterns[i]["XY"] / self.patterns[i]["X"]) - self.patterns[i]["Y"])/self.patterns[i]["_Y"]

    def conviction(self, i):
        """ Calculates the conviction metric of a given subspace
        DOI 10.1145/170036.170072
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Conviction of subspace
        """
        if self.patterns[i]["X_Y"] == 0:
            return math.inf
        else:
            return self.patterns[i]["X"] * self.patterns[i]["_Y"] / self.patterns[i]["X_Y"]

    def coverage(self, i):
        """ Calculates the support metric of a given subspace
        10.1145/170036.170072
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Support of subspace
        """
        return self.patterns[i]["X"]

    def descriptive_confirmed_confidence(self, i):
        """ Calculates the descriptive confidence of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Descriptive confidence of subspace
        """
        return (self.patterns[i]["XY"]/self.patterns[i]["X"]) - (self.patterns[i]["X_Y"]/self.patterns[i]["X"])

    def difference_of_confidence(self, i):
        """ Calculates the difference of confidence metric of a subspace
        https://doi.org/10.1007/s001800100075
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Difference of confidence of subspace
        """
        return (self.patterns[i]["XY"] / self.patterns[i]["X"]) - (self.patterns[i]["_XY"] / self.patterns[i]["_X"])

    def example_counter_example(self, i):
        """ Calculates
        Generation of rules with certainty and confidence factors from incomplete and incoherent learning bases
        author : Sebag, M and Schoenauer, M
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Example and counter example metric of subspace
        """
        if self.patterns[i]["XY"] == 0:
            return "No intersection between subspace and outcome"
        return (self.patterns[i]["XY"] - self.patterns[i]["X_Y"]) / self.patterns[i]["XY"]

    def imbalance_ratio(self, i):
        """ Calculates the imbalance ratio metric of a given subspace
        https://doi.org/10.1007/s10618-009-0161-2
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Imbalance ratio of subspace
        """
        if self.patterns[i]["XY"] == 0:
            return "No intersection between subspace and outcome"
        return abs((self.patterns[i]["XY"]/self.patterns[i]["X"])-(self.patterns[i]["XY"]/self.patterns[i]["Y"]))/((self.patterns[i]["XY"]/self.patterns[i]["X"])+(self.patterns[i]["XY"]/self.patterns[i]["Y"])-((self.patterns[i]["XY"]/self.patterns[i]["X"])*(self.patterns[i]["XY"]/self.patterns[i]["Y"])))

    def fishers_exact_test_p_value(self, i):
        """ Calculates Fisher's test p-value of a given subspace
        DOI 10.3233/IDA-2007-11502
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            P-value of Fisher's test of subspace
        """
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
        """ Calculates the Hyper confidence metric of a given subspace
        DOI 10.3233/IDA-2007-11502
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Hyper confidence of subspace
        """
        return 1 - self.fishers_exact_test_p_value(i)

    def hyper_lift(self, i):
        """ Calculates the Hyper lift metric of a given subspace
        DOI 10.3233/IDA-2007-11502
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Hyper lift of subspace
        """
        [M, n, N] = [self.size_of_dataset, self.patterns[i]["Cy"], self.patterns[i]["Cx"]]
        ppf95 = hypergeom.ppf(0.95, M, n, N)
        return self.patterns[i]["Cxy"]/ppf95

    def laplace_corrected_confidence(self, i):
        """ Calculates the laplace corrected confidence of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Laplace corrected confidence
        """
        return (self.patterns[i]["Cxy"]+1)/(self.patterns[i]["Cx"]+(len(self.unique_classes)))

    def importance(self, i):
        """ Calculates the importance metric of a given subspace
        https://docs.microsoft.com/en-us/analysis-services/data-mining/microsoft-association-algorithm-technical-reference?view=asallproducts-allversions&viewFallbackFrom=sql-server-ver15
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Importance metric of subspace
        """
        return math.log(((self.patterns[i]["Cxy"]+1)/(self.patterns[i]["Cx"]+len(self.unique_classes))) / ((self.patterns[i]["Cx_y"]+1)/(self.patterns[i]["Cx"]+len(self.unique_classes))), 10)

    def jaccard_coefficient(self, i):
        """ Calculates the jaccard coefficient metric of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Jaccard coefficient of subspace
        """
        return self.patterns[i]["XY"]/(self.patterns[i]["X"]+self.patterns[i]["Y"]-self.patterns[i]["XY"])

    def j_measure(self, i):
        """ Calculates the J-Measure (scaled version of cross entropy) of a given subspace
        NII Article ID (NAID) 10011699020
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            J-Measure of subspace
        """
        if self.patterns[i]["_Y"] == 0:
            return 0
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
        """ Calculates the kappa metric for a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Kappa of subspace
        """
        return (self.patterns[i]["XY"] + self.patterns[i]["_X_Y"]-(self.patterns[i]["X"] * self.patterns[i]["Y"])-(self.patterns[i]["_X"]*self.patterns[i]["_Y"])) / (1-(self.patterns[i]["X"]*self.patterns[i]["Y"])-(self.patterns[i]["_X"]*self.patterns[i]["_Y"]))

    def klosgen(self, i):
        """ Calculates the klosgen metric for a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Klosgen metric of subspace
        """
        return math.sqrt(self.patterns[i]["XY"])*((self.patterns[i]["XY"]/self.patterns[i]["X"])-self.patterns[i]["Y"])

    def kulczynski(self, i):
        """ Calculates the kulczynski metric of a given subspace
        DOI https://doi.org/10.1007/s10618-009-0161-2
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Kulczynski metric of subspace
        """
        return 0.5 * ((self.patterns[i]["XY"] / self.patterns[i]["X"]) + (self.patterns[i]["XY"] / self.patterns[i]["Y"]))

    def kruskal_lambda(self, i):
        """ Calculates the goodman-kruskal lambda metric for a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Goodman-kruskal lambda of subspace
        """
        return ((1-self.patterns[i]["XY"])-(1-self.patterns[i]["Y"]))/(1-self.patterns[i]["XY"])

    def least_contradiction(self, i):
        """ Calculates the least contradiction metric of a given subspace
        (2004) Extraction de pepites de connaissances dans les donnees: Une nouvelle approche et une etude de sensibilite au bruit. In Mesures de Qualite pour la fouille de donnees. Revue des Nouvelles Technologies de l’Information, RNTI
        author : Aze, J. and Y. Kodratoff
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Least contradiction of subspace
        """
        return (self.patterns[i]["XY"] - self.patterns[i]["X_Y"]) / self.patterns[i]["Y"]

    def lerman_similarity(self, i):
        """ Calculates the lerman similarity metric of a given subspace
        (1981) Classification et analyse ordinale des données.
        Author : Lerman, Israel-César.
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Lerman similarity of subspace
        """
        return (self.patterns[i]["Cxy"] - ((self.patterns[i]["Cx"] * self.patterns[i]["Cy"]) / self.size_of_dataset)) / math.sqrt((self.patterns[i]["Cx"] * self.patterns[i]["Cy"]) / self.size_of_dataset)

    def piatetsky_shapiro(self, i):
        """ Calculates the shapiro metric of a given subspace
        NII Article ID (NAID) 10000000985
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Shapiro metric of subspace
        """
        return self.patterns[i]["XY"] - (self.patterns[i]["X"] * self.patterns[i]["Y"])

    def max_confidence(self, i):
        """ Calculates the maximum confidence metric of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Max Confidence of subspace
        """
        return max(self.patterns[i]["XY"] / self.patterns[i]["X"], self.patterns[i]["XY"] / self.patterns[i]["Y"])

    def odds_ratio(self, i):
        """ Calculates the odds ratio metric of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Odds ratio of subspace
        """
        if self.patterns[i]["X_Y"] == 0 or self.patterns[i]["_XY"] == 0:
            return math.inf
        else:
            return (self.patterns[i]["XY"] * self.patterns[i]["_X_Y"]) / (self.patterns[i]["X_Y"] * self.patterns[i]["_XY"])

    def phi_correlation_coefficient(self, i):
        """ Calculates the phi correlation coefficient metric of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Phi correlation coefficient of subspace
        """
        return math.sqrt(self.chi_squared(i)/self.size_of_dataset)

    def ralambondrainy_measure(self, i):
        """ Calculates the support of the counter examples of a given subspace
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Ralambondrainy metric of subspace
        """
        return self.patterns[i]["X_Y"]

    def rld(self, i):
        """ Calculates the Relative Linkage Disequilibrium (RLD) of a given subspace
        https://doi.org/10.1007/978-3-540-70720-2_15
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            RLD of subspace
        """
        if self.patterns[i]["_Y"] == 0:
            return 0
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
        """ Calculates the relative risk of a given subspace
        https://doi.org/10.1148/radiol.2301031028
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Relative risk of subspace
        """
        if self.patterns[i]["_XY"] == 0:
            return math.inf
        return (self.patterns[i]["XY"]/self.patterns[i]["X"])/(self.patterns[i]["_XY"]/self.patterns[i]["_X"])

    def rule_power_factor(self, i):
        """ Calculates the rule power factor of a given subspace
        https://doi.org/10.1016/j.procs.2016.07.175
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Rule power factor of subspace
        """
        return (self.patterns[i]["XY"]**2)/self.patterns[i]["X"]

    def sebag(self, i):
        """ Calculates the sebag metric of a given subspace
        Generation of rules with certainty and confidence factors from incomplete and incoherent learning bases
        author : Sebag, M and Schoenauer, M
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Sebag metric of subspace
        """
        if self.patterns[i]["X_Y"] == 0:
            return math.inf
        else:
            return self.patterns[i]["XY"]/self.patterns[i]["X_Y"]

    def yule_q(self, i):
        """ Calculates the yule's Q metric of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Yule's Q of subspace
        """
        if self.patterns[i]["_Y"] == 0:
            return 0
        return (self.patterns[i]["XY"]*self.patterns[i]["_X_Y"] - self.patterns[i]["X_Y"]*self.patterns[i]["_XY"]) / (self.patterns[i]["XY"]*self.patterns[i]["_X_Y"] + self.patterns[i]["X_Y"]*self.patterns[i]["_XY"])

    def yule_y(self, i):
        """ Calculates the yule's Y of a given subspace
        https://doi.org/10.1016/S0306-4379(03)00072-3
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Yule's Y of subspace
        """
        if self.patterns[i]["_Y"] == 0:
            return 0
        return (math.sqrt(self.patterns[i]["XY"] * self.patterns[i]["_X_Y"]) - math.sqrt(self.patterns[i]["X_Y"] * self.patterns[i]["_XY"])) / (math.sqrt(self.patterns[i]["XY"] * self.patterns[i]["_X_Y"]) + math.sqrt(self.patterns[i]["X_Y"] * self.patterns[i]["_XY"]))

    def quality_of_pattern(self, i):
        """ Calculates the amount of non-noisy elements of a given subspace
        https://doi.org/10.1016/j.patcog.2021.107900
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Percentage of non-noisy elements of subspace
        """
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
        return counter / (len(self.patterns[i]["columns"]) * len(self.patterns[i]["lines"]))

    def Wsup_pattern(self, i):
        """ Calculates weighted support (rows to be considered correct) of a given subspace
        https://doi.org/10.1016/j.patcog.2021.107900
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Weighted support of subspace
        """
        counter = 0
        for row in self.patterns[i]["lines"]:
            col_pos = -1
            nr_correct_row = 0
            for column in self.patterns[i]["columns"]:
                col_pos += 1
                column_value = self.patterns[i]["column_values"][col_pos]
                if pd.isna(self.data.at[row, column]) or not is_number(self.data.at[row, column]):
                    continue
                elif column_value - self.patterns[i]["noise"][col_pos] <= \
                        float(self.data.at[row, column]) <= \
                        column_value + self.patterns[i]["noise"][col_pos]:
                    nr_correct_row += 1
                elif self.border_values:
                    if float(self.data.at[row, column]) == float(column_value) + 0.5 or\
                            float(self.data.at[row, column]) == float(column_value) - 0.5:
                        nr_correct_row += 1
            if nr_correct_row/len(self.patterns[i]["columns"]) >= 0.5:
                counter += 1

        return counter / self.size_of_dataset

    def Wsup_rule(self, i):
        """ Calculates the weighted support of a given subspace for the selected outcome value
        https://doi.org/10.1016/j.patcog.2021.107900
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Weighted support of rule described by the subspace
        """
        counter = 0
        for row in self.patterns[i]["lines"]:
            nr_correct_row = 0
            col_pos = -1
            for column in self.patterns[i]["columns"]:
                col_pos += 1
                column_value = self.patterns[i]["column_values"][col_pos]
                skip = True
                if self.outcome_type == "Numerical":
                    intervals = self.patterns[i]["outcome_intervals"]
                    for interval in intervals:
                        if interval[0] <= float(self.y_column[row]) <= interval[1]:
                            skip = False
                else:
                    if float(self.y_column[row]) == self.patterns[i]["outcome_to_assess"]:
                        skip = False
                if skip:
                    continue

                if pd.isna(self.data.at[row, column]) or not is_number(self.data.at[row, column]):
                    continue
                elif column_value - self.patterns[i]["noise"][col_pos] <= \
                        float(self.data.at[row, column]) <= \
                        column_value + self.patterns[i]["noise"][col_pos]:
                    nr_correct_row += 1
                elif self.border_values:
                    if float(self.data.at[row, column]) == float(column_value) + 0.5 or \
                            float(self.data.at[row, column]) == float(column_value) - 0.5:
                        nr_correct_row += 1

            if nr_correct_row / len(self.patterns[i]["columns"]) >= 0.5:
                counter += 1

        return counter / self.size_of_dataset

    def Wconf(self, i):
        """ Calculates the weighted confidence of a given subspace
        https://doi.org/10.1016/j.patcog.2021.107900
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Weighted confidence of subspace
        """
        if self.Wsup_pattern(i) == 0:
            return math.inf
        else:
            return self.Wsup_rule(i) / self.Wsup_pattern(i)

    def WLift(self, i):
        """ Calculates weighted lift of a given subspace
        https://doi.org/10.1016/j.patcog.2021.107900
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Weighted lift of subspace
        """
        if self.Wsup_pattern(i) == 0:
            return math.inf
        else:
            return self.Wsup_rule(i) / (self.Wsup_pattern(i) * self.patterns[i]["Y"])

    def Tsig(self, i):
        """ Calculates the statistical significance of a given subspace
        https://doi.org/10.1007/s10618-017-0521-2
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            Statistical significance of a subspace
        """
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
                    elif not self.border_values and ((float(column_value) - self.patterns[i]["noise"][col_pos]) <= item <= (float(column_value) + self.patterns[i]["noise"][col_pos])):
                        counter += 1

                p = p * (counter/self.size_of_dataset)
                col_pos += 1
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
        elif self.patterns[i]["type"] == "Order-Preserving":
            p = 0 #float(Decimal(1.0) / Decimal(math.factorial(self.patterns[i]["nr_cols"])))
            counter = 0
            for column in self.data.columns:
                for row in self.data[column]:
                    if is_number(row):
                        counter += 1
            row_counter = 0
            for row in range(self.size_of_dataset):
                aux_counter = 0
                for values in self.data.iloc[row,:]:
                    if is_number(values):
                        aux_counter += 1
                if aux_counter > self.patterns[i]["nr_cols"]:
                    row_counter += 1
            percentage_missings = counter // (len(self.data.columns)*self.size_of_dataset)
        return betainc(self.patterns[i]["Cx"], self.size_of_dataset, p)

    def FleBiC_score(self, i):
        """ Calculates the score considered by FleBiC regarding subspace importance
        https://doi.org/10.1016/j.patcog.2021.107900
        Parameters
        ----------
        i : int
            Index of subspace.
        Returns
        -------
        metric : float
            FleBiC score of subspace
        """
        if self.Wsup_pattern(i) == 0:
            return math.inf
        part1 = 0.6 * 0.7 * ((self.Wsup_rule(i)/self.Wsup_pattern(i)) * (self.patterns[i]["Y"]/self.size_of_dataset) + 0.3 * self.chi_squared(i))
        part2 = 0.3 * 0.5 * ((self.Wsup_rule(i)/self.size_of_dataset) * (self.patterns[i]["Y"]/self.size_of_dataset) + 0.5 * (self.patterns[i]["nr_cols"]/len(self.data.columns)))
        part3 = 0.1 * self.quality_of_pattern(i)
        return self.Tsig(i) * (part1 + part2 + part3)

    def _plots(self, uniques, pattern, outcome, intervals, label_pattern, label_outcome, i):
        if self.file_path_plots is not None or self.show_plots:
            plt.plot(uniques, pattern, label=label_pattern)
            plt.plot(uniques, outcome, label=label_outcome)
            for interval in intervals:

                for val in interval:
                    plt.axvline(x=val, color="red", linestyle='--')
            #plt.ylim(top=0.5)
            #plt.ylim(bottom=-0.01)
            plt.legend()
        if self.file_path_plots is not None:
            plt.savefig(self.file_path_plots+str(i)+'.png', bbox_inches='tight')
            plt.close()
        if self.show_plots:
            plt.show()
            plt.close()

    def _min_max_numeric(self, outcome_p):
        return [[outcome_p.min(), outcome_p.max()]]

    def _average_var(self, outcome_p):
        m = outcome_p.mean()
        std = outcome_p.std()
        return [[m-std, m+std]]

    def _empirical(self, outcome_p, outcome, p_number, heuristic=False):
        # Get uniques
        x = []
        for val in outcome:
            if val not in x:
                x.append(val)
        x.sort()
        # Get unique counters
        outcome_p_counters = []
        outcome_counters = []
        for i in range(len(x)):
            outcome_p_counters.append(0)
            outcome_counters.append(0)
            for val in outcome_p:
                if val == x[i]:
                    outcome_p_counters[i] = outcome_p_counters[i] + 1
            for val in outcome:
                if val == x[i]:
                    outcome_counters[i] = outcome_counters[i] + 1

        #Transform to relative frequency
        for i in range(len(outcome_counters)):
            outcome_p_counters[i] = outcome_p_counters[i]/len(outcome_p)
            outcome_counters[i] = outcome_counters[i]/len(outcome)

        #Get intersection
        idx = np.argwhere(np.diff(np.sign(np.array(outcome_p_counters) - np.array(outcome_counters)))).flatten().tolist()

        intervals = []
        if outcome_p_counters[0] > outcome_counters[0]:
            intervals.append([-math.inf, x[idx[0]]])
            del idx[0]
        if outcome_p_counters[-1] > outcome_counters[-1]:
            intervals.append([x[idx[-1]], math.inf])
            del idx[-1]

        while len(idx) > 0:
            e1 = x[idx[0]]
            del idx[0]
            e2 = x[idx[0]]
            del idx[0]
            intervals.append([e1, e2])

        if self.heuristic:
            lift_of_intervals = []
            for interval in intervals:
                temp_outcome = outcome.copy()
                temp_pattern = outcome_p.copy()
                temp_outcome.apply(lambda x: 1 if interval[0] <= x <= interval[1] else 0)
                temp_pattern.apply(lambda x: 1 if interval[0] <= x <= interval[1] else 0)
                lift = (temp_pattern[temp_pattern == 1].size/temp_outcome.size)/((temp_outcome[temp_outcome == 1].size / temp_outcome.size) * (temp_pattern.size/temp_outcome.size))
                lift_of_intervals.append({"name":interval, "lift":lift})
            lift_of_intervals = sorted(lift_of_intervals, key=lambda x: x["lift"], reverse=True)

            while True:
                temp_outcome = outcome.copy()
                temp_pattern = outcome_p.copy()
                for val in lift_of_intervals:
                    temp_outcome.apply(lambda x: 1 if val["name"][0] <= x <= val["name"][1] else 0)
                    temp_pattern.apply(lambda x: 1 if val["name"][0] <= x <= val["name"][1] else 0)
                lift = (temp_pattern[temp_pattern == 1].size/temp_outcome.size)/((temp_outcome[temp_outcome == 1].size / temp_outcome.size) * (temp_pattern.size/temp_outcome.size))
                interval_comb = lift_of_intervals
                if len(lift_of_intervals) > 0 or lift < 1.3:
                    break
                del lift_of_intervals[-1]

            if len(lift_of_intervals) == 0 and lift < 1.3:
                return intervals
            else:
                intervals = []
                for val in interval_comb:
                    intervals.append(val["name"])
                return intervals

        if self.file_path_plots or self.show_plots:
            self._plots(x, outcome_p_counters, outcome_counters, intervals, "pattern", "outcome", p_number)
        return intervals

    def _gaussian(self, outcome_p, outcome, i):
        m1 = outcome_p.mean()
        m2 = outcome.mean()
        std1 = outcome_p.std()
        std2 = outcome.std()

        if std1 == 0:
            idxs = [m1,m1]
        else:
            # Solve for a gaussian
            a= -1/(std1**2) + 1/(std2**2)
            b= 2*(-m2/(std2**2) + m1/(std1**2))
            c= (m2**2)/(std2**2) - (m1**2)/(std1**2) + np.log((std2**2)/(std1**2))
            intercep_points = np.roots([a, b, c])
            idxs = sorted(intercep_points)

        if self.show_plots or self.file_path_plots:
            x = np.arange(-1, 1, 0.001)
            self._plots(x, norm.pdf(x, m1, std1), norm.pdf(x, m2, std2), [idxs], 'μ:' + str(round(m1, 2)) + ', σ:' + str(round(std1, 2)), 'μ:' + str(round(m2, 2)) + ', σ:' + str(round(std2, 2)), i)

        return [[idxs[0], idxs[1]]]


    def handle_numerical_outcome(self, x_space, method, i):
        """ Calculated the interception point between the gaussian curve of outcome variable and outcome variable described by the subspace

        Parameters
        ----------
        x_space : list
            Oucome variable described by the subspace.
        Returns
        -------
        metric : list
            [0] : first point of interception
            [1] : second point of interceptiion
        """
        intervals = []
        if method == "gaussian":
            intervals = self._gaussian(x_space.map(float), self.y_column.map(float), i)
        if method == "min_max":
            intervals = self._min_max_numeric(x_space.map(float))
        if method == "average":
            intervals = self._average_var(x_space.map(float))
        if method == "empirical":
            intervals = self._empirical(x_space.map(float), self.y_column.map(float), i)
        return intervals


def is_number(s):
    """ Checks if parameter passed is a number

    Returns
    -------
    boolean
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

