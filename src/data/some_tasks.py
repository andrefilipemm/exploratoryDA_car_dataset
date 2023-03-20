import pandas as pd

# We import our previous stored pikle
data = pd.read_pickle(
    r"C:\Users\1ab89\Desktop\VSCode Workspaces\Data Science Projects\DS Project #1\data\interim\car_clean_data.pkl"
)

"""Before hopping into the next section we shall solve various questions 
in order to visualize and build better insights about the data we're dealing with"""

## Value Counts
# Check what are the different types of "Make" there are in our dataset.
# How many ocurrences of each "Make" are there?

types = data["Make"]    # different types
types.value_counts()    # counts the types

## Filtering Task
# Show all the records where Origin is Asia or Europe

countries = data["Origin"] # different countries in Origin

records = data[countries.isin(["Asia", "Europe"])]  # all the records where Origin is Asia or Europe
records.shape   # the number of rows = how many records there are where Origin is Asia or Europe

## Instruction
# Remove all the records where Weight > 4000

weights = data["Weight"]    
overweights = data[weights > 4000]  # all the cars whose Weight > 4000
remove_overweights = data[~(weights > 4000)] # ~ removes all records where the condition between () is met

## Instruction
# Increase all the values of "MPG_City" column by 3

mpg_col = data["MPG_City"]                  # A lambda function in python is a small anonymous
mpg_col = mpg_col.apply(lambda x: x+3)      # function that can take any number of arguments and execute an expression.
                                            # lambda expressions are utilized to construct anonymous functions

cols = list(data.columns.values)