## Import Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

path = os.path.join('Task2-dataset-dog_breeds.csv')

df = pd.read_csv(path)

def compute_euclidean_distance(vec_1, vec_2):
     # your code comes here

    #find the square root of vector1 - vector2 pos 0
    first_index = np.square(vec_2[0] - vec_1[0])
    #find the square root of vector1 - vector2 pos 1
    second_index = np.square(vec_2[1] - vec_1[1])
    #find the square root of vector1 - vector2 pos 2
    third_index = np.square(vec_2[2] - vec_1[2])
    #find the square root of vector1 - vector2 pos 3
    fourth_index = np.square(vec_2[3] - vec_1[3])
    # Get the sum of the vectors squared total
    total = first_index + second_index + third_index + fourth_index
    #square the total summed numbers to get the distance between vectors.
    distance = np.sqrt(total)
    
    #return distance for later use
    return distance

#df.reset_index(drop=True, inplace=True)
def initialise_centroids(dataset, k):
    #create a matrix of values from the dataset
    Data = dataset.values
    #shuffle the matrix values to get random centroid
    np.random.shuffle(Data)
    #create matrix of ones to store centroids
    Cent = np.ones((k, 4))
    #create loop for appending k random rows to list based on defined k amount
    for c in range(k):
        #replace the data in cent matrix to the centroid data
        Cent[c] = Data[c]
    #return centorid
    return Cent 


#get centroids positions
centroid = initialise_centroids(df, 3)

#get data as matrix
df_matrix = df.values
print(df_matrix, "data")

#Create list to store distance indexes
cluster_groups = []

#loop through data within the created data matrix
for index in df_matrix:  
    
    print("index", index)
    centroid_len = len(centroid)
    centroid_distance_matrix = np.ones(centroid_len)
    
    #loop through centroids
    for c in range(centroid_len):

        #compute 3 eucliden distance for each index in the dataset
        centroid_distance = compute_euclidean_distance(index, centroid[c])
        #Store the k distances for the index in a list
        centroid_distance_matrix[c] = centroid_distance
 
    print(centroid_distance_matrix, "distance")
    #find which distance is minimum out of the 3 (this will be the euclidiean distance)
    min_distance = np.amin(centroid_distance_matrix)
    #find the index of the smallest number. This wil;l be the cluster that it belongs to
    min_distance_index = np.argmin(centroid_distance_matrix)
    print(min_distance, min_distance_index, "min_distance")
    cluster_groups.append(min_distance_index)

cluster_groups = np.array(cluster_groups) # This turns the cluster groups into a numpy array


#print(minindex, "list of index's")
df_with_clusters = df.copy()
df_with_clusters["Cluster"] = cluster_groups
print("data", df_with_clusters)

    