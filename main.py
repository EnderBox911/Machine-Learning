# Pandas is a data analysis python library
import pandas as pd
# Seaborn is a python graphing library
import seaborn as sns
# Plotting graphs
from matplotlib import pyplot as plt
# Machine Learning Library (Gotta do "$ python3 -m pip install scikit-learn" in shell first) - Train and make predictions with a linear model
from sklearn.linear_model import LinearRegression
# Dealing with errors in predictions
from sklearn.metrics import mean_absolute_error
# Scientific Computimatng Package - Allows you to work with matrices: inverse them, etc.
import numpy as np

"""from sklearn.datasets import load_iris
import matplotlib
from pandas.plotting import table

# loading the iris dataset
iris = load_iris()
  
# creating a 2 dimensional dataframe out of the given data
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],columns=iris['feature_names'] + ['target'])
  
# grouping data and calculating average
grouped_dataframe = iris_df.groupby('target').mean().round(1)
grouped_dataframe['species_name'] = ['setosa', 'versicolor', 'virginica']
  
# plotting data
ax = plt.subplot(211)
plt.title("Iris Dataset Average by Plant Type")
plt.ylabel('Centimeters (cm)')
  
ticks = [4, 8, 12, 16]
a = [x - 1 for x in ticks]
b = [x + 1 for x in ticks]
  
plt.xticks([])
  
#plt.bar(a, grouped_dataframe.loc[0].values.tolist()[:-1], width=1, label='setosa')
#plt.bar(ticks, grouped_dataframe.loc[1].values.tolist()[:-1], width=1, label='versicolor')
#plt.bar(b, grouped_dataframe.loc[2].values.tolist()[:-1], width=1, label='virginica')
  
plt.legend()
plt.figure(figsize=(12, 8))
table(ax, grouped_dataframe.drop(['species_name'], axis=1), loc='bottom')
plt.show()"""



# -- STEPS -- 



# https://www.youtube.com/watch?v=Hr06nSA-qww
# 1. Form hypothesis
# 2. Find data
# 3. Reshape data
# 4. Clean data
# 5. Error metric
# 6. Split data
# 7. Train model




# -- DATA COLLECTING --




# Get data from the file (data.txt: Raw data from https://github.com/dataquestio/project-walkthroughs/tree/master/beginner_ml)
teams = pd.read_csv("data.csv") 




# -- DATA RESHAPING --




# Get specific columns and take out other collumns
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Print out the table
print(teams)

# See if the column "medals" has any corelaton to the other collumns
print(teams.corr(numeric_only = True)["medals"])




# -- GRAPHS --


# Plot graph to show correlation between no. of athletes and medals
# fit_reg makes a REGRESSION line
# sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)

# Display graph
# plt.title("Athletes Correlation To Medals Won")
# plt.show()

# Result - HIGH CORRELATION



# Plot graph to show correlation between age and medals
# sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)

# Display graph
# plt.title("Age Correlation To Medals Won")
# plt.show()

# Result - LOW CORRELATION



# Plot a histogram to show how many countries fall into each bin for number of medals earn
# teams.plot.hist(y="medals")
# plt.title("Histogram")
# plt.show()

# Result - Lots of countries (2000) won 0-50 medals, few countries won a lot of medals -> DATA IS UNBALANCED




# -- DATA CLEANING --




# Finds any rows with missing values - This is because some countries didn't participate in the previous olympics = no previous medals won
print(teams[teams.isnull().any(axis=1)])

# Removes all the empty rows from data
teams = teams.dropna()
print(teams)




# -- DATA SPLITING --



# Training data uses data from PAST years and not the future 
train = teams[teams["year"] < 2012].copy()

# Evaluate how well/accurate the program is doing on the training data
test = teams[teams["year"] >= 2012].copy()

# Gets (rows, columns) of the table - Train should be around 80% of the data and test 20%
print(train.shape)
print(test.shape)




# -- TRAINING --




# Initialise Linear Regression class
reg = LinearRegression()

# Columns used to predict target
predictors = ['athletes', "prev_medals"]

# The target to predict
target = "medals"

# Train linear regression model to predict target
# Fits linear regression model
reg.fit(train[predictors], train[target])


# Gets the y-intercept and the coefficient - This stuff was used for the formula when building own model (Link to vid on making own model with formulas in plans.txt) -> Making own model explains the formulas more better and helps learnwhen to use linear regression
print(reg.intercept_, reg.coef_, "GUH")

# Make predictions without knowing what the answers are
predictions = reg.predict(test[predictors])
print(predictions)

# Result - Data are floats (doesn't make sense in this context) and negative



# Putting a predictions column in the table
test["predictions"] = predictions
print(test)

# Index the test data frame and find any rows where the predictions column is less than 0 and turns it to 0
test.loc[test["predictions"] < 0, "predictions"] = 0

# Round predictions
test["predictions"] = test["predictions"].round()

test.to_csv('predictions.csv')

pd.read_csv('predictions.csv')

pd.plotting.table(plt.axes, test, rowLabels=["team", "country", "year", "athletes", "age", "prev_medals", "medals"], colLabels=None)

#test.plot.table("team", "country", "year", "athletes", "age", "prev_medals", "medals")

#plt.show()

# -- ERROR METRIC --




# Finds the average of how far the predictions were off
error = mean_absolute_error(test["medals"], test["predictions"])
#print(error)


# Deeper info regarding a column
#std = standard deviation
info = teams.describe()['medals']
#print(info)
# Result - Mean Absolute Error should be below std -  If it isn't, something is wrong (e.g wrong/irrelevant predictors, error with model)


# Look at predictions for specific countries
country = test[test["team"] == "USA"]
#print(country)

country = test[test["team"] == "IND"]
#print(country)


# See errors country by country
# Finds the mean absolute error
errors = (test["medals"] - test["predictions"]).abs()
#print(errors)

# Create seperate group for each team and finds mean of the absolute error
error_by_team = errors.groupby(test["team"]).mean()
#print(error_by_team)

# Create seperate group for each team and finds mean of the amount of medals won
medals_by_team = test["medals"].groupby(test["team"]).mean()

# Finds ratio between errors
error_ratio = error_by_team / medals_by_team
#print(error_ratio)
# Result - Lots of NaN because it is dividing by 0 since a lot of countries have an avg of 0

# Get values that are NOT missing
error_ratio = error_ratio[~pd.isnull(error_ratio)]
#print(error_ratio)
# Result - Some values are infinite because error_by_team is 1 and medals_by_team is 0

# Get values that are not infinite
error_ratio = error_ratio[np.isfinite(error_ratio)]
#print(error_ratio)


# Create a histogram of the error ratio
error_ratio.plot.hist()
# plt.title("Error Ratio")
# plt.show()
# Result - Some values are more than half predicted medals


# Sort values for sort for the best results AKA. countries that win the most medals since more accurate
#print(error_ratio.sort_values())
# Results - Errors are low for countries that get lots of medals but errors are high for countries we don't have much data for or don't send as many athletes 
