from re import X
from turtle import distance
import pandas as pd
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import os

#from sympy import false

#function for computing distance between vectors
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

def cluster_sort(dataset, k, centroid):
   
    #centroid = initialise_centroids(dataset, k)
    #get data as matrix
    df_matrix = dataset.values
    #print(df_matrix, "data")
    count= 0
    #print(df_matrix.shape, "shaper")
    list_totals = []
    #Create list to store distance indexes
    cluster_groups = []
    for index in df_matrix:  
        #counter
        row_index = 0 
        #counter 
        centroid_index = 0

        centroid_len = len(centroid)
        dataset_len = len(df_matrix)

        #create a matrix to store distances
        centroid_distance_matrix = np.ones(centroid_len)

        #loop through centroids
        for c in range(centroid_len):

            #compute 3 eucliden distance for each index in the dataset
            centroid_distance = compute_euclidean_distance(index, centroid[c])
            #Store the k distances for the index in a list
            centroid_distance_matrix[centroid_index] = centroid_distance
            centroid_index = centroid_index + 1 

            total = centroid_distance_matrix.sum()
        list_totals.append(total)
    
        min_distance_index = np.argmin(centroid_distance_matrix)
        #print(min_distance, min_distance_index, "min_distance")
        row_index = row_index + 1
        #put min index distance into array
        cluster_groups.append(min_distance_index)

    #This turns the cluster groups into a numpy array
    cluster_groups = np.array(cluster_groups)
    #print(minindex, "list of index's")
    #copy dataframe
    df_with_clusters = dataset.copy()
    #add cluster column to dataframe and add data    
    df_with_clusters["Cluster"] = cluster_groups

    #total distances
    distance_total = sum(list_totals)

    return df_with_clusters, distance_total

def kmeans(dataset, k):

    #get centroids positions
    centroid = initialise_centroids(dataset, k)
    #assign the data to the cloest centroid
    df_with_clusters, distance_total = cluster_sort(dataset, k, centroid)
    #get the mean of the centroids
    New_centroids = mean_cluster(df_with_clusters, k, centroid)
    
    #Error: 
    error_list = []
    iteration_list = []
    current_iteration = 1
    iteration_list.append(current_iteration)
    error_list.append(distance_total)

    #make vriable to store start centroids
    base_centroids = New_centroids
    #bool value to check data is not the same
    matching = False

    #while matching isnt true
    while matching != True:
        print("cluster sort")
        #run cluster sort to assign to cloest centroid
        df_with_clusters, distance_total= cluster_sort(dataset, k, centroid)
        # get mean of the new data

        adjusted_centroids = mean_cluster(df_with_clusters, k, centroid)

        error_list.append(distance_total)
        current_iteration += 1
        iteration_list.append(current_iteration)

        print(df_with_clusters)

        #print the centroids
        print(centroid, "these are centroids")
        #print the adjusted centorids after mean has been found
        print(adjusted_centroids, "adjusted_centroids")

        #If the centroids and the adjusted centorids are the same then
        if np.array_equal(centroid, adjusted_centroids):
            #stop the loop with matchng = true
            matching = True
            print("Match")
            #print the final centroid values to console 
            print(adjusted_centroids, "AD centorids")
            #make centroid the same as adjuatsed
            centroid = adjusted_centroids
        else:
            #make centroid the same as adjuatsed
            centroid = adjusted_centroids
    #return updated centroids and dataframe


    plt.plot(iteration_list, error_list)
    plt.title('Error')
    plt.show()

    return centroid, df_with_clusters


def mean_cluster(data, k, centroid):

    
    #get centorid length
    centroid_len = len(centroid)
    print(centroid_len, "length of centorid")

    #create counter
    count = 0
    #create data matrix of centroid shape to store mean centroids
    New_centroids = np.ones(centroid.shape)
    #loop through k amount
    for c in range(centroid_len):
        
        #seperate each cluster to own dataset
        Cluster_select = data[data['Cluster'] == c]

        #drop the string column so we can do numpy computations
        dropped = Cluster_select.drop('Cluster', 1)
       
        #Get mean of each column in dataframe
        Cluster_mean = dropped.mean()

        #get the mean values in a numpy array
        x = Cluster_mean.values

        #Add the means to each row in the constructed matrix
        New_centroids[count] = x

        #add to the count variable
        count = count + 1

    #retunr new centroids for later use.
    return New_centroids

def main():
    #set path
    path = os.path.join('Task2-dataset-dog_breeds.csv')
    #read dataset
    df = pd.read_csv(path)
    #df.reset_index(drop=True, inplace=True)


    centroid, df_with_clusters = kmeans(df, 2)

    # Plot the graph between height and tail length 
    cluster_1 = df_with_clusters[df_with_clusters['Cluster'] == 0]
    cluster_2 = df_with_clusters[df_with_clusters['Cluster'] == 1]
    #cluster_3 = df_with_clusters[df_with_clusters['Cluster'] == 2]

    plt.scatter(
        cluster_1['height'],
        cluster_1['tail length'],
        color='pink'
    )

    plt.scatter(
        cluster_2['height'],
        cluster_2['tail length'],
        color='green'
    )
    

    plt.scatter(
        centroid[0, 0],
        centroid[0, 1],
        color='black'
    )

    plt.scatter(
        centroid[1, 0],
        centroid[1, 1],
        color='black'
    )


    
    plt.title('K-Means clustering on Height and tail length')
    plt.show()

    plt.scatter(
        cluster_1['height'],
        cluster_1['leg length'],
        color='pink'
    )

    plt.scatter(
        cluster_2['height'],
        cluster_2['leg length'],
        color='green'
    )
 

    plt.scatter(
        centroid[0, 0],
        centroid[0, 1],
        color='black'
    )

    plt.scatter(
        centroid[1, 0],
        centroid[1, 1],
        color='black'
    )

    plt.title('K-Means clustering on Height and leg length')
    plt.show()



if __name__ == '__main__':
    main()

