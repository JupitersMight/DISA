# DISA

DISA (**D**iscriminative and **I**nformative **S**ubspace **A**nalysis), a software package in Python (v3.7) capable of assessing patterns with numerical outputs by statistically testing the correlation gain of the subspace against the overall space
To illustrate the DISA properties, one example used was the [dataset](https://pubs.acs.org/doi/pdf/10.1021/acssynbio.9b00020) that monitors the concentration of key enzymes observed in two Design-Build-Test-Learn (DBTL) cycles of 1-dodecanol production in *Escherichia coli*.

## Please cite

Alexandre, L., Costa, R. S. & Henriques, R. (2022). DISA tool: Discriminative and informative subspace assessment with categorical and numerical outcomes. PLoS ONE, 17(10), e0276253 | doi: https://doi.org/10.1371/journal.pone.0276253

## Input parameters

data : pandas.Dataframe

patterns : A python list where each position is a dictionary with the pattern properties:

    [i] : { 
            "lines" : list (mandatory)
            "columns" : list (mandatory)
            "column_values": list (optional)
            "noise": list (optional)
            "type" : string (optional)
           }

Description of parameters:
- "lines" refers to the observations of the pattern
- "columns" refers to the variables of the pattern
- "column_values" refers to the pattern coherence on columns
- "noise" refers to the noise allowed in each column
- "type" refers to the type of coherence (by default we assume constant coherence)

outcome : dict
    
    {
        "values": pandas.Series (mandatory)
        "outcome_value" : int (optional)
        "type": string (optional)
        "method": string (optional)
        "heuristic": boolean (optional)
    }
    
Description of parameters:
- "values" the outcome variable
- "outcome_value" : if the outcome variable is categorical the user can force DISA to analyse according to a specific category, we assume the category is represented by a discrete value. (by default it will select the best category per pattern)
- "type": if the user wishes to analyse a continuous outcome variable this field should take the value "Numerical" (by default it is assumed to be categorical)
- "method": if in the "type" parameter the user inputted "Numerical" then this field can be filled in with:
  - "min_max" - uses the minimum and the maximum values of each pattern to define the pattern-conditioned outcome intervals
  - "average" - uses the average +- standard deviation of each pattern to define the pattern-conditioned outcome intervals
  - "gaussian" (default) - assumes both the pattern-conditioned outcome and the outcome variable follow a normal distribution to define the pattern-conditioned outcome intervals
  - "empirical" - uses the empirical distribution of the pattern-conditioned outcome and outcome variables to define the pattern-conditioned outcome intervals 
- "heuristic": if in the "method" parameter the user inputted "empirical" then this field can be used to optimize the discriminative properties of each pattern

   
output_configurations : dict (optional)

    {
        "print_table" : boolean 
        "file_path_table" : string
        "show_plots" : boolean
        "file_path_plots" : string
        "print_numeric_intervals" : boolean
        "file_path_numeric_intervals" : string
     }

Description of parameters:
- "print_table" : if set to True will output a table of results (by default it is set to True)
- "file_path_table" : path to file to write the output table to (by default it is set to None)
- "show_plots" : if the "method" parameters in the outcome parameter was set to "gaussian" or "empirical" then this parameter if set to True will plot a figure (by default it is set to False)
- "file_path_plots" : if the "method" parameters in the outcome parameter was set to "gaussian" or "empirical" then the path to a folder can be set in this parameter to output a PNG of the figures (by default it is set to None)
- "print_numeric_intervals" : if the "type" parameter in the outcome parameter was set to "Numerical" then this parameter can be set to True to output the interval that the pattern discriminates (by default it is set to False)
- "file_path_numeric_intervals" : if the "type" parameter in the outcome parameter was set to "Numerical" then the path to a folder can be set in this parameter to output to a file the intervals that each pattern discriminates

## Dataset examples

Four examples on how to use DISA are provided in the folder "Example", the Echocardiogram, the Liver Disorders, the Breast Cancer Wisconsin (diagnostic), and Dodecanol datasets. Inside each of the datasets corresponding folder you will find a set of files and a folder. The python and jupyter notebook files provide the code to analyse patterns using DISA. The patterns are contained in the .txt files and the processed datasets in both the .csv and .arff files.

## Package dependencies

pandas - 1.4.3

numpy - 1.23.1

scipy - 1.8.1

prettytable - 3.3.0

matplotlib - 3.5.1

## Metrics

A list of all the implemented metrics in DISA and the corresponding DOI (some but not all of these metrics are futher explained in https://mhahsler.github.io/arules/docs/measures).

Information Gain: https://doi.org/10.1016/S0306-4379(03)00072-3 

Chi-squared: https://doi.org/10.1145/253260.253327

Gini index: https://doi.org/10.1016/S0306-4379(03)00072-3

Difference in Support: 10.1109/TKDE.2010.241

Bigger Support: 10.1109/TKDE.2010.241

Confidence: 10.1145/170036.170072

All-Confidence: 10.1109/TKDE.2003.1161582

Lift: 10.1145/170036.170072

Standardised Lift: https://doi.org/10.1016/j.csda.2008.03.013

Collective Strength: https://dl.acm.org/doi/pdf/10.1145/275487.275490

Cosine: https://doi.org/10.1016/S0306-4379(03)00072-3

Interestingness: arXiv:1202.3215

Comprehensibility: arXiv:1202.3215

Completeness: arXiv:1202.3215

Added Value: https://doi.org/10.1016/S0306-4379(03)00072-3

Casual Confidence: https://doi.org/10.1007/3-540-44673-7_1

Casual Support: https://doi.org/10.1007/3-540-44673-7_1

Certainty Factor: 10.3233/IDA-2002-6303

Conviction: 10.1145/170036.170072

Coverage (Support): 10.1145/170036.170072

Descriptive Confirmed Confidence: https://doi.org/10.1016/S0306-4379(03)00072-3

Difference of Proportions: https://doi.org/10.1007/s001800100075

Example and Counter Example: SEBAG, M.; SCHOENAUER, M. Generation of rules with certainty and confidence factors from incomplete and incoherent learning bases. In: Proc. of EKAW. 1988. p. 28.

Imbalance Ratio: https://doi.org/10.1007/s10618-009-0161-2

Fisher's Exact Test (p-value): 10.3233/IDA-2007-11502

Hyper Confidence: 10.3233/IDA-2007-11502

Hyper Lift: 10.3233/IDA-2007-11502

Laplace Corrected Confidence: https://doi.org/10.1016/S0306-4379(03)00072-3

Importance: https://docs.microsoft.com/en-us/analysis-services/data-mining/microsoft-association-algorithm-technical-reference?view=asallproducts-allversions&viewFallbackFrom=sql-server-ver15

Jaccard Coefficient: https://doi.org/10.1016/S0306-4379(03)00072-3

J-Measure: NII Article ID (NAID) 10011699020

Kappa: https://doi.org/10.1016/S0306-4379(03)00072-3

Klosgen: https://doi.org/10.1016/S0306-4379(03)00072-3

Kulczynski: https://doi.org/10.1007/s10618-009-0161-2

Goodman-Kruskal's Lambda: https://doi.org/10.1016/S0306-4379(03)00072-3

Least Contradiction: (2004) Extraction de pepites de connaissances dans les donnees: Une nouvelle approche et une etude de sensibilite au bruit. In Mesures de Qualite pour la fouille de donnees. Revue des Nouvelles Technologies de l’Information, RNTI author : Aze, J. and Y. Kodratoff

Lerman Similarity: (1981) Classification et analyse ordinale des données. Author : Lerman, Israel-César.

Piatetsky-Shapiro: NII Article ID (NAID) 10000000985

Max Confidence: https://doi.org/10.1016/S0306-4379(03)00072-3

Odds Ratio: https://doi.org/10.1016/S0306-4379(03)00072-3

Phi Correlation Coefficient: https://doi.org/10.1016/S0306-4379(03)00072-3

Ralambondrainy: DIATTA, Jean; RALAMBONDRAINY, Henri; TOTOHASINA, André. Towards a unifying probabilistic implicative normalized quality measure for association rules. In: Quality Measures in Data Mining. Springer, Berlin, Heidelberg, 2007. p. 237-250.

Relative Linkage Disequilibrium: https://doi.org/10.1007/978-3-540-70720-2_15

Relative Risk: https://doi.org/10.1148/radiol.2301031028

Rule Power Factor:https://doi.org/10.1016/j.procs.2016.07.175

Sebag-Schoenauer : SEBAG, M.; SCHOENAUER, M. Generation of rules with certainty and confidence factors from incomplete and incoherent learning bases. In: Proc. of EKAW. 1988. p. 28.

Yule Q: https://doi.org/10.1016/S0306-4379(03)00072-3

Yule Y: https://doi.org/10.1016/S0306-4379(03)00072-3

Weighted Support: https://doi.org/10.1016/j.patcog.2021.107900

Weighted Rule Support: https://doi.org/10.1016/j.patcog.2021.107900

Weighted Confidence: https://doi.org/10.1016/j.patcog.2021.107900

Weighted Lift: https://doi.org/10.1016/j.patcog.2021.107900

Statistical Significance: https://doi.org/10.1007/s10618-017-0521-2

FleBiC Score: https://doi.org/10.1016/j.patcog.2021.107900

## Authors

DISA was developed by:

L. Alexandre (leonardoalexandre@tecnico.ulisboa.pt), R.S. Costa (rs.costa@fct.unl.pt) and R. Henriques
