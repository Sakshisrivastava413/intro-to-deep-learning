# we want to predict animal body weight given its brain weight

# importing dependencies
# to read our data sets
import pandas as pd
# machine learning library
from sklearn import linear_model
# to visualize our model and data
import matplotlib.pyplot as plt

# read data
# fwf function is use to read the animal data sets
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['brain']]
y_values = dataframe[['body']]

# train model on data
# initialize linear regression and store it in body regression variable
body_reg = linear_model.LinearRegression()
# fit the model in x-y value pair
body_reg.fit(x_values, y_values)

# visulize result
# plot the regression line
plt.scatter(x_values, y_values)
# for every x_value predict the y_value (intersecting one)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
