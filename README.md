
-> Overview 
This repository contains a comprehensive Exploratory Data Analysis of the Titanic passenger manifest. The goal is to identify patterns and factors (like class, age, and gender) that determined survival outcomes. 

-> Data Cleaning Steps 
* Imputation: Missing 'age' values are filled using the median; 'embarked' is filled with the mode. 
* Feature Engineering: Irrelevant or redundant columns such as `deck`, `embark_town`, and `alive` are removed to streamline analysis. 
* Integrity: Duplicate records are identified and removed. 

-> Dashboard Components 
The analysis outputs a multi-panel dashboard (`task02_output.png`) featuring: 
* Survival Rates: Broken down by Gender and Passenger Class. 
* Distributions: Histograms for Age (segmented by survival) and Fare (log scale). 
* Correlations: A heatmap showing the relationship between numerical variables like SibSp, Parch, and Fare. 
* Demographics: Port of embarkation distribution. 

## Requirements 
* Python 3.x 
* Pandas, Seaborn, Matplotlib
