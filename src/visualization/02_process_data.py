import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import sys

# This allows us to import our own plot settings as a module using sys library
sys.path.append("..")
import utility.plot_settings

# We import our previous stored pikle
data = pd.read_pickle(
    r"C:\Users\1ab89\Desktop\VSCode Workspaces\Data Science Projects\DS Project #1\data\interim\car_clean_data.pkl"
)


# --------------------------------------------------------------
# 4. Apply business logic
# --------------------------------------------------------------

"""
Q1) Are there any trends or patterns in the data that can help the company
identify which types of cars are most popular among customers, and how can this
information be used to inform marketing and sales strategies?
"""

# Creation of a bar chart displaying the Frequency of Car Types
car_type_freq = data["Type"].value_counts()
plt.bar(car_type_freq.index, car_type_freq.values)
# Plot labels
plt.title("Frequency of Car Types")
plt.xlabel("Car Type")
plt.ylabel("Frequency")
# Char display
plt.show()

    # Based on the bar chart created, it seems that Sedan is the most frequent car type in the dataset.
    # This information can be used to inform the company's marketing and sales strategies by focusing
    # their advertising efforts on Sedans or developing new products that are similar to popular Sedan
    # models. Additionally, the company may want to investigate why Sedans are so popular and consider
    # factors such as price, features, and customer preferences when developing new products. Overall,
    # this type of analysis can provide valuable insights into customer preferences and help the company
    # make data-driven decisions.

"""
Q2) What is the relationship between a car's weight and its fuel efficiency, and how can
this information be used to improve the company's product offerings?
"""

## We shall atempt a Polynomial Regression Model

# Extract the weight and fuel efficiency columns
X = data["Weight"].values.reshape(-1, 1)
y = data["MPG_City"].values.reshape(-1, 1)

# Fit a polynomial function of degree 2 to the data
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Predict fuel efficiency for a range of weights
X_range = np.arange(X.min(), X.max(), 10).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred = poly_reg.predict(X_range_poly)

# Calculate the mean squared error of the model
y_pred_train = poly_reg.predict(X_poly)
mse = mean_squared_error(y, y_pred_train)
print("Mean Squared Error:", mse)

mse_value = 3.37  # Just for visualization purposes

# Plot the data and the fitted curve
plt.scatter(X, y, s=2.5)
plt.plot(X_range, y_pred, color="green")
plt.xlabel("Weight")
plt.ylabel("Fuel Efficiency")
plt.title("Polynomial Regression Model (2nd Degree)")
plt.legend(["Data Points", "Model with MSE value: 3.37"], loc="center right")
plt.show()

# Using the polynomial regression model of 2nd degree I found the relationship
# between a car's weight and its fuel efficiency it to be negative, that is
# the higher the weight the less fuel efficient it tends to be.

"""
Q3) How does the type of a car affect its MSRP?
Are there any specific types that tend to have higher or lower prices?
"""

# Define the formatter function to add dollar signs and commas
def dollar_format(x, pos):
    return "${:,.0f}".format(x)

# Create a boxplot of MSRP by car type
boxplot = sns.boxplot(
    x="Type",
    y="MSRP",
    data=data,
    order=data.groupby("Type")["MSRP"].median().sort_values().index,
)

# Apply the formatter function to the y-axis
boxplot.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_format))
boxplot.set_ylim(10000,)

    # Yes, there are specific types that tend to have higher or lower prices based
    # on the boxplots. The "Sports" type tends to have the highest prices as it ha
    # the widest and highest boxplot, with the highest median. The "Truck" type tends
    # to have the lowest prices as it has the narrowest and lowest boxplot, with the
    # lowest median. The "Sedan Type", "SUV", and "Wagon" types have boxplots with median
    # prices close to each other, suggesting that their prices are not significantly
    # different.
    # Overall, it seems that certain types of cars are generally associated 
    # with higher MSRP values than others.