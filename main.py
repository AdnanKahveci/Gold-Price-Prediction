# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 02:21:02 2024

@author: kahve
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv("gold_price.csv", parse_dates=True, index_col='Date')

# Calculate returns and lagged returns
df['Return'] = df['USD (PM)'].pct_change() * 100
df['Lagged_Return'] = df.Return.shift()
df = df.dropna()

# Ensure the date range exists
# Filter the data by year using the .loc accessor
train = df.loc['2001':'2018']
test = df.loc['2019']

# Create train and test sets for dependent and independent variables
X_train = train["Lagged_Return"].to_frame()
y_train = train["Return"]
X_test = test["Lagged_Return"].to_frame()
y_test = test["Return"]

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Plot the results
out_of_sample_results = y_test.to_frame()
out_of_sample_results["Out-of-Sample Predictions"] = predictions
out_of_sample_results.plot(subplots=True, title='Gold prices, USD')
plt.show()
