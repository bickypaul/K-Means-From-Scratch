import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]]) # numpy array of datapoints

fig1 = plt.figure()
fig1.suptitle('K-Means Clustering Algorithm', fontsize=12, fontweight='bold')
plt.scatter(X[:,0], X[:,1], s=150)
ax = fig1.add_subplot(111)
fig1.subplots_adjust(top=0.85)
ax.set_title('Data Points')
plt.show() #plotting the datapoints, not yet clustered.

colors = 10 * ['g','r','c','b','k']

'''
This part is the actual algorithm for flat K-means clustering algorithm and prediction of test datapoint
which tells which group they would possibly belong based on the least eucliedean distances from the optimum centroid.
Here k=2 (as a default) which means there will be two clusters or grouping of the datapoints. 
According to scikit-learn algorithm for K-means default max-iterations is 300, so I chose to keep it same here as well.
'''
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        

    # here this is fit/train algorithm where the optimal centroids are evualated, or the training model. 
    def fit(self, data):
        self.centroids = {}
        
        # arbitrary selecting two datapoint as centroid, in this case
        # first two datapoints in the data array.
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        for j in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
            
            '''
            finding the euclidean distances between each datapoint and the centroid and taking the
            index value of the least distance in the distances array and appending the corresponding datapoint 
            associated in that particular iteration to the classifications dictionary, index value as the label number,
            which here is 0 and 1.
            '''
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            
            prev_centroids = dict(self.centroids)
            
            '''
            Average of the datapoints that belong in the self.classifications dictionary to calcucate the new centroids 
            because as in K-means the centroid movestowards an optimal centroid further which there is no new movement or
            change in the magnitude of centroid.
            '''
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                # axis = 0 means columns average
            
            optimized = True
            


            '''
            Here in this part the percentage movement of centroids is controlled by the tolerance(tol=0.001) 
            with respect to the previous centroids, so as to determine the optimal centroid.
            '''
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                               
                if np.sum((current_centroid - original_centroid)/original_centroid * 100.0) > self.tol:
                    optimized = False
                    
            if optimized:
                break
                
    
    '''
    This is just the prediction part of the algorithm which the returns the label 
    based on the minimum distances of respective new unknown datapoints.
    '''
    def predict(self, data):
        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            return classification

clf = K_Means() 
clf.fit(X) # train

# visualization
fig = plt.figure()
fig.suptitle('K-Means Clustering Algorithm', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Clusters/Groups')

#plotting the optimal centroids 
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
               marker="o", color="k", s=150, linewidths=5)
    ax.annotate('Centroids', xy=(clf.centroids[centroid][0], clf.centroids[centroid][1]), xytext=(2.5, 5.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
    
    

# actual datapoints that were used to train the model  
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1],
                    marker="x", color=color, s=150, linewidths=5)
        

unknowns = np.array([[1,3],[5,6],[8,9],[2,3],[2,9]])

# plotting the predicted group or cluster the unknown datapoint will belong to.
for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)
       
plt.show()
        
    