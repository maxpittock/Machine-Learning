import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Task1.csv')
#    df.drop(columns=)

x_train = df['x']
y_train = df['y']



print("Training x data")
print(x_train, x_train.shape)
print("Training y data")
print(y_train, y_train.shape)

plt.clf()
plt.title("Line graph")
plt.plot(x_train, y_train, 'bo', color ="red")
plt.show()

Xtilde = np.column_stack((np.ones(x_train.shape), x_train))
print(Xtilde)

#Split data into 2 sets code
# variable for getting 70% of csv - for training set
#spliter = np.random.rand(len(df)) <= 0.7
# train = random 70% of csv file
#train = df[spliter]
# test = the 30% that wasnt used in the training variable
#test = df[~spliter]

#print("training", train)
#print ("testing", test)

def pol_regression(features_train, y_train, degree):
     # code comes here
    parameters = 1
    #return polynomial coefficients