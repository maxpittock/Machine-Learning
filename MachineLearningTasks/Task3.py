## Import Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

#intialise findspark
findspark.init()
# SparkSession class
spark = SparkSession.builder.getOrCreate()

#function to normalise the data
def data_normalisation(data):
    #min max data noramlisation
    data_normalized = (data - data.min())/(data.max() - data.min())
    return data_normalized


def statistics(data):
    #convert to spark for computatation
    sparkDF=spark.createDataFrame(data) 
    #Find all statsistics for each column with spark
    print("Statistics")
    statistics = sparkDF.summary().toPandas()
    #shows stats table
    print(statistics)

def plots(data):

    #plot pandas dataframe by status
    data.boxplot(column= "Alpha", by="Participant Condition")
    #plt the boxplot
    plt.show()

    #convert to spark so we can select certain columns
    sparkDF=spark.createDataFrame(data) 
    #sepearting dataset so that we can plot denisty based on certain columns
    density_plot_patient = sparkDF.select(["Participant Condition", "Beta"]).where(col("Participant Condition") == "Patient").toPandas()
    density_plot_control = sparkDF.select(["Participant Condition", "Beta"]).where(col("Participant Condition") == "Control").toPandas()
    #Plot lines on graphs (THIS CIRRENT PLOTS ON 2 DIFFERENT ONES)
    plt1 = density_plot_patient.plot.density()
    plt.legend(['Patient'])
    plt2 = density_plot_control.plot.density()
    plt.legend(['Control'])
    plt.show()

def prep_data(data):
    print(data, "hello")
    #convert to spark dataframe
    #select the features and class that we want for the machine learning tasks
    dataprep = data[["Alpha","Beta","Lambda","Lambda1","Lambda2", "Participant Condition"]]
    #create label encoder to convert Participant Condition column to string
    le = LabelEncoder()
    #changes patient to 1 and control to 0
    dataprep['Participant Condition'] = le.fit_transform(data['Participant Condition'])

    #drop the Participant Condition string column 
    #drop_string = data.drop(["Participant Condition"], 1) 
    
    print("drop it")
    print(dataprep)
    return dataprep

def data_split(data):
    #import the prepared data
    drop_string = prep_data(data)
    #drop_string.show()
    #split the datsets by x and y
    #df = drop_string.toPandas()
    #get data as numpy arrays in own datasets
    X_df = drop_string[["Alpha","Beta","Lambda","Lambda1","Lambda2"]].to_numpy()
    Y_df = drop_string[['Participant Condition']].to_numpy()
    
    #split dataframes into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X_df, Y_df, test_size=0.10)

    print("xtrain")
    print(X_train.shape)
    print("xtest")
    print(X_test.shape)
    print("ytrain")
    print(y_train.shape)
    print("ytest")
    print(y_test.shape)

    return X_train, X_test, y_train, y_test


def neural_network(data):

    X_train, X_test, y_train, y_test = data_split(data)

    mlp = MLPClassifier(solver='adam', activation="logistic", max_iter=100, hidden_layer_sizes=(500,500,), verbose=50)
    mlp.out_activation_ = "logistic"
    x = mlp.fit(X_train,y_train)

    #training predictions
    predict_train = x.predict(X_train)
    #testing predicts
    predict_test = x.predict(X_test)
    #import accuracy libarys
    from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
    #Get error rate for training
    print("training error")
    print(accuracy_score(y_train, predict_train))
    # get error rate for testing
    print("testing error")
    print(accuracy_score(y_test, predict_test))

def tenfold(data):
    df = data
    #folded dataset
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    #amount of folds
    k = 10
    #k fold algorithm
    kf = KFold(n_splits=k, random_state=None)
    #logistic regressor 
    mlp = MLPClassifier(solver='adam', activation="logistic", max_iter=20, hidden_layer_sizes=(500,500,), verbose=10)

    #list for storing accuracy score
    acc_score = []

    #Loop through the folded dataset
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]

        #fit the model
        mlp.fit(X_train,y_train)
        #create model predictions 
        pred_values = mlp.predict(X_test)
        
        #get accuracy of classifer on each iteration
        acc = accuracy_score(pred_values , y_test)
        #append to list
        acc_score.append(acc)
    
    #accuracy avg total
    avg_acc_score = sum(acc_score)/k
 
    #print accuracy metrics to console.
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))


    
def main():
    path = os.path.join('Task3-dataset-HIVRVG.csv')

    df = pd.read_csv(path)
    print("df")
    print(df)

        #create temp to store participant column
    temp = df['Participant Condition']
    #drop the string column so we can normalise (this will bea added back after)
    dropped_df = df.drop('Participant Condition',1)

    #function call
    data_normalized = data_normalisation(dropped_df)
    #add removed column back
    data_normalized['Participant Condition'] = temp #Add back the participant condition
    #print to check dataframe
    print("data", data_normalized)

    statistics(df)
    statistics(data_normalized)
    plots(data_normalized)
    neural_network(data_normalized)
    tenfold(data_normalized)

if __name__ == '__main__':
    main()

