# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


df = pd.read_csv('Task3-dataset-HIVRVG.csv') 
print(df.shape)
df.describe().transpose()

target_column = ["Alpha","Beta","Lambda","Lambda1","Lambda2"]
predictors = ["Participant Condition"]
predictors = list(set(list(predictors))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
print(predictors, "pred")
df.describe().transpose()

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)