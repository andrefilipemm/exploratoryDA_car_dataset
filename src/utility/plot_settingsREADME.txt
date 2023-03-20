import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# This turns our plot_settings file into a module
# --------------------------------------------------------------

import sys
sys.path.append("..")
import utility.plot_settings


# --------------------------------------------------------------
# Import data Example
# --------------------------------------------------------------

data = pd.read_csv(
    "data path",
    parse_dates=[0],
    index_col=[0],
)

# --------------------------------------------------------------
# Plotting data Example
# --------------------------------------------------------------

data["first column"].plot()

# Plot columns in a loop
for col in data.columns[:5]:
    data[col].plot()
    plt.show()