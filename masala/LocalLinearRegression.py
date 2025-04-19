from os import wait
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import compress
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from .cyclicRegression import CyclicRegression
from copy import deepcopy
import os
import glob
import pickle as pck

class LocalLinearRegression():
    '''
    This class performs local linear regression on a given dataset. It takes in the data and a distance function
    which is used to calculate the neighbourhood of points to include in each points local linear regression.
    The current neighbourhood is set to 5% of the maximum distance.
    '''
    def __init__(self, x,y, dist_function):
        self.feature_type = dist_function
        print(self.feature_type)
        self.x = x
        self.y = y
        self.N = len(x)
        self.CR = CyclicRegression(boundary=max(self.x))
        if dist_function == 'Linear':
            self.dist_function = self.euclideanDefine()
        elif dist_function == 'Cyclic':
            self.dist_function = self.CR.cyclic_distance
        elif dist_function == 'Time':
            self.dist_function = self.timeDiff

    def euclideanDefine(self):
        ''' Defines a eculidea distance function and normalises the distance based on the maximum value in the dataset.'''
        maxX = max(self.x)
        maxY = max(self.y)
        minX = min(self.x)
        minY = min(self.y)
#        maxDistance = np.sqrt((maxX-minX)**2 + 0.25*(maxY-minY)**2)
        maxDistance = np.sqrt((maxX-minX)**2)
#        euclidean = lambda x1, x2, y1, y2: np.sqrt(((x1-x2))**2 + (0.25*(y1-y2))**2)/maxDistance
        euclidean = lambda x1,x2: abs(x1-x2)/maxDistance
        return euclidean

    def timeDiff(self, x1,x2):
        ''' A time difference function which takes in datetime objects and returns the difference in days.'''
        return abs(x1-x2).days

    def pointwiseDistance(self, X, Y):
        ''' Calculates the distances between every point and every other point based on the defined distance function.
        Returns a matrix of all the distances which are normalised to range between 0 and 1.'''
        xDs = []
        for x1, y1 in zip(X, Y):
            x1Ds = []
            for x2, y2 in zip(X, Y):
                x1Ds.append(self.dist_function(x1,x2))
            xDs.append(x1Ds)
        self.xDs = xDs
        self.meanDistance = np.mean(xDs)
        # Normalise Distances:
        normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)
        maxX, minX = np.max(self.xDs), np.min(self.xDs)
        self.xDs_norm = np.array(list(map(lambda x: normalise(maxX, minX, x), self.xDs))).reshape(self.N, self.N)


    def calculateLocalModels(self, neighbourhood_threshold=0.05):
        ''' Calculates the local linear model for each point. First calulates the neighbourhood of a point by
        checking which points distance is less than the defined 5% neighbourhood threshold. Then performs linear regression within this neighbourhood.
        Stores the parameters and error of each LR model.'''
        plotModelParameters = False

        w1 = []
        w2 = []
        w = []
        MSE = []

#        Calculate distances between all points so can reuse later.
        self.pointwiseDistance(self.x, self.y)
        self.neighbourhoods = []
        for i in tqdm(range(self.N)):
            check = [self.xDs_norm[i][j]<neighbourhood_threshold for j in range(self.N)]
            localxdata = list(compress(self.x, check))
            localydata = list(compress(self.y, check))

            self.neighbourhoods.append(len(localxdata))
            if i ==0:
                self.first_point_neighbours = [localxdata, localydata]
            X = np.array([localxdata, np.ones(len(localxdata))]).T
            if self.feature_type == 'Linear':
                wlocal = np.linalg.lstsq(X, localydata, rcond=1)[0]
                w1.append(wlocal[0])
                w2.append(wlocal[1])
                w.append(wlocal)
            elif self.feature_type == 'Cyclic':
                m, c = self.CR.cyclicRegression(localxdata, localydata)
                w1.append(m)
                w2.append(c)
                w.append([m, c])

#           LLR plotting for individual points
#            if i%10 ==0:
#                xrange = np.linspace(min(localxdata), max(localxdata), 100)
#                fig, axes = plt.subplots(1,1,figsize=(10,10))
#                axes.scatter(self.x, self.y, s=3)
#                axes.scatter(localxdata, localydata, s=5, c='red')
#                axes.scatter(self.x[i], self.y[i], s=100, c='green')
#                ys = [wlocal[0]*xrange[j]+wlocal[1] for j in range(100)]
#                axes.legend(['All Data', 'Local Data', 'Selected Point', 'Local Model'])
#                plt.plot(xrange, ys, c='green', linewidth=5)
#                plt.savefig('Figures/LocalModels/housingLinearRegression{}.png'.format(i))

#            plt.scatter(self.x, self.y, s= self.xDs_norm[i]*100)
#            plt.show()
#            for circular regression uncomment the following:

#            preds, m, c = self.CR.cyclicRegression(localxdata, localydata)
#
#            w1.append(m)
#            w2.append(c)
#            w.append([m, c])
#            MSE.append(mean_squared_error(localydata, preds))

        return w1, w2, w



    def compute_distance_matrix(self, w, distance_weights= {'x': 1, 'w': 1, 'neighbourhood': 1}, instance=None):
        ''' Computes a distance matrix between all points for a distance function which includes the raw distance values,
        the parameters and error of the Local Linear Regression models for the respective points.
        This will be used as the distance matrix for any clustering algorithms. '''

        wDs, mseDs, neighbourhoodDs = [], [], []

        euclidean = lambda l1, l2: sum((p-q)**2 for p, q in zip(l1, l2)) ** .5
#        print('Computing Distance Matrix...')
        if instance == None:
            D = np.zeros((self.N,self.N))
            D = []
            indexes = [[i, j] for i in tqdm(range(self.N)) for j in range(self.N)]
            wDs = list(map(lambda x: euclidean(w[x[0]], w[x[1]]), indexes))
            neighbourhoodDs = list(map(lambda x: abs(self.neighbourhoods[x[0]]-self.neighbourhoods[x[1]]), indexes))
#            [(wDs.append(euclidean(w[i], w[j])), neighbourhoodDs.append(abs(self.neighbourhoods[i]-self.neighbourhoods[j]))) for i in range(self.N) for j in range(self.N)]
            normalise = lambda maxX, minX, x: (x-minX)/(maxX-minX)

            maxWds, minWds = np.max(wDs), np.min(wDs)
            wDs_norm = np.array(list(map(lambda x: normalise(maxWds, minWds, x), wDs))).reshape(self.N, self.N)

            maxNeighbourhood, minNeighbourhood = np.max(self.neighbourhoods), np.min(self.neighbourhoods)
            neighbourhood_norm = np.array(list(map(lambda x: normalise(maxNeighbourhood, minNeighbourhood, x), neighbourhoodDs))).reshape(self.N, self.N)

            def distance_calc(indices):
                i, j = indices
                distance = distance_weights['x']*self.xDs_norm[i,j] + distance_weights['w']*wDs_norm[i,j] +  distance_weights['neighbourhood']*neighbourhood_norm[i,j]
                return distance

            for i in tqdm(range(self.N)):
                D.append(list(map(distance_calc, [(i, j) for j in range(self.N)])))
#                for j in range(self.N):
#                    distance = distance_weights['x']*self.xDs_norm[i,j] + distance_weights['w']*wDs_norm[i,j] +  distance_weights['neighbourhood']*neighbourhood_norm[i,j]
#                    D[i,j] = distance

#        with open('saved/distance_matrix.pck', 'wb') as file:
#            pck.dump([D, self.xDs_norm], file)
        return D, self.xDs_norm



#    def KMedoidClustering(self, K, D):
#        km = kmedoids.KMedoids(n_clusters=K, init='random', random_state=0, method='pam')
#        # c = kmedoids.fasterpam(D,K)
#        c=km.fit(D)
#        clustered_data = []
#        for k in np.unique(c.labels_):
#            clustersx = []
#            clustersy = []
#            for i in range(len(c.labels_)):
#                if c.labels_[i] == k:
#                    clustersx.append(self.x[i])
#                    clustersy.append(self.y[i])
#
#            clustered_data.append([clustersx, clustersy])
#
#        orderedClusters = []
#        minimums = [min(clustered_data[j][0]) for j in range(len(clustered_data))]
#        minsCopy = minimums.copy()
#        for i in range(len(clustered_data)):
#            firstCluster = minimums.index(min(minsCopy))
#            minsCopy.remove(min(minsCopy))
#            orderedClusters.append(clustered_data[firstCluster])
#        return orderedClusters
