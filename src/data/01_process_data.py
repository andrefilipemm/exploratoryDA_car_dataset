import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------------------
# 1. Define Objective
# --------------------------------------------------------------

"""
The objective of this scrip is to create 2 figures for a report
to evaluate
    - Relationship between a car's weight and its fuel efficiency (MPG)
    - How the type of a car affect its MSRP
"""

# --------------------------------------------------------------
# 2. Read raw data
# --------------------------------------------------------------

file_path = r"C:\Users\1ab89\Desktop\VSCode Workspaces\Data Science Projects\DS Project #1\data\raw\cars_data.csv"
raw_data = pd.read_csv(file_path)

# We visualize part of the data to get a sense of it
raw_data.head()
raw_data.shape
raw_data.dtypes

# --------------------------------------------------------------
# 3. Process data
# --------------------------------------------------------------

"""
Instruction (data cleaning)

Delete the dollar sign was well as commas from the 'MSRP' column.
"""

raw_data = raw_data.loc[raw_data['MSRP'] != '']
raw_data['MSRP'] = raw_data['MSRP'].replace('[\$,]', '', regex=True).astype(float)

"""
Instruction (data cleaning)

Find all the Null values in the dataset.
If there is any Null value in any column, then fill it with the mean value. of that column
"""

raw_data.isnull().sum()  # We found every column has missing values

# Fill missing values with mean value for each column
raw_data.fillna(raw_data.mean(), inplace=True)

# We realized there were still object type columns with Null values
# Select only the non-numeric columns
non_numeric_cols = raw_data.select_dtypes(exclude="number").columns.tolist()
# Fill missing values in non-numeric columns with an empty string
raw_data[non_numeric_cols] = raw_data[non_numeric_cols].fillna("")

"""
Identify outliers

We shall identify any outliers regarding
the columns ->  'EngineSize', 'Cylinders',
                'Horsepower', 'MPG_City', 'MPG_Highway', 
                'Weight', 'Wheelbase', 'Length'
"""

# Select the columns to work with
cols = [
    "EngineSize",
    "Cylinders",
    "Horsepower",
    "MPG_City",
    "MPG_Highway",
    "Weight",
    "Wheelbase",
    "Length",
]

# Calculate the z-scores for each value in the selected columns
z = np.abs(stats.zscore(raw_data[cols]))
# Find the row indices where the z-score is greater than 3 (i.e., potential outliers)
outlier_indices = np.where(z > 3)[0]
# Remove the potential outliers from the data
clean_data = raw_data.drop(raw_data.index[outlier_indices])

# --------------------------------------------------------------
# Export data
# --------------------------------------------------------------

clean_data.to_pickle(
    r"C:\Users\1ab89\Desktop\VSCode Workspaces\Data Science Projects\DS Project #1\data\interim\car_clean_data.pkl"
)