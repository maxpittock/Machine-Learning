#%%

## Import Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

path = os.path.join('Task2-dataset-dog_breeds.csv')

df = pd.read_csv(path)

def compute_euclidean_distance(vec_1, vec_2):
    #  # your code comes here

    # #find the square and the sum of each vector
    # vec1_distance = np.sqrt(np.sum(np.square(vec_1)))
    # vec2_distance = np.sqrt(np.sum(np.square(vec_2)))

    # # minus the results of squaring the vectors by each other which gives you the euclidean distance.
    # distance = vec1_distance - vec2_distance 

    # x, y, z, g
    # x2, y2, z2, g2
    
    first_index = np.square(vec_2[0] - vec_1[0])
    second_index = np.square(vec_2[1] - vec_1[1])
    third_index = np.square(vec_2[2] - vec_1[2])
    fourth_index = np.square(vec_2[3] - vec_1[3])
    
    total = first_index + second_index + third_index + fourth_index
    distance = np.sqrt(total)
    
    
    return distance

def initialise_centroids(dataset, k):
    #create a matrix of values from the dataset
    Data = dataset.values

    #shuffle the matrix values to get random centroid
    np.random.shuffle(Data)
    
    Cent = np.ones((k, 4))
    
    #create loop for appending k random rows to list based on defined k amount
    for c in range(k):
        Cent[c] = Data[c]
    
    return Cent 

def kmeans(dataset, k):
    # Initialise centroids:
    centroids = initialise_centroids(dataset=dataset, k=3) # We use the dataset and k variables passed from the function.
    
    df_matrix = dataset.values
    
    # cluster_groups = np.ones(len(df_matrix))
    cluster_groups = [] # Max this is a python list to store the distances dw it becomes a numpy array later.
    for index in df_matrix:
        
        row_index = 0 # I then get a counter to store the current row
        
        distances = np.ones(len(centroids)) # I then create a np.ones to store the distances.
        centroid_index = 0 # This is also to count the current centroid (this is my work around enumerate)
        
        for centroid in centroids:
            distance = compute_euclidean_distance(index, centroid) # We then calculate the distance
            distances[centroid_index] = distance # We store the distance in the distances array to later calcualte the minimum
            centroid_index = centroid_index + 1 # We increase the counter at the end of the for loop.
        
        
        min_index = np.argmin(distances) # This retreives the minimum index (Leave this alone only way it works).
        cluster_groups.append(min_index) # I then append that new distance to it.
        row_index = row_index + 1 # We then increase the row index for the next row.            
    
    cluster_groups = np.array(cluster_groups) # This turns the cluster groups into a numpy array
    
    # This is putting it into a dataframe for a new column.
    df_with_clusters = dataset.copy() # Copy the dataset
    df_with_clusters['Cluster'] = cluster_groups # Add the clusters into a new column.
    
    return centroids, df_with_clusters
   

centroids, df_with_clusters = kmeans(dataset=df, k=2)
3
# I AM PUTTING THIS HERE TO VISUALISE IT FOR YOU CHANGE COMMENTS AND VARIABLES
# Plot the graph between height and tail length 
cluster_1 = df_with_clusters[df_with_clusters['Cluster'] == 0]
cluster_2 = df_with_clusters[df_with_clusters['Cluster'] == 1]
cluster_3 = df_with_clusters[df_with_clusters['Cluster'] == 2]

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
    cluster_3['height'],
    cluster_3['tail length'],
    color='brown'
)

plt.scatter(
    centroids[0, 0],
    centroids[0, 1],
    color='black'
)

plt.scatter(
    centroids[1, 0],
    centroids[1, 1],
    color='black'
)

plt.scatter(
    centroids[2, 0],
    centroids[2, 1],
    color='black'
)
plt.title('wasteman')
plt.show()
# %%
