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
    def __init__(self, x,y, dist_function, feature_name):
        self.feature_type = dist_function
        print(self.feature_type)
        self.x = x
        self.y = y
        self.N = len(x)
        self.CR = CyclicRegression(boundary=max(self.x))
        self.feature_name = feature_name
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
        self.local_xs = []
        self.local_ys = []
        neighbourhood_threshold = 0.5*np.std(self.xDs_norm)
        for i in tqdm(range(self.N)):
            check = [self.xDs_norm[i][j]<neighbourhood_threshold for j in range(self.N)]
            localxdata = list(compress(self.x, check))
            self.local_xs.append(localxdata)
            localydata = list(compress(self.y, check))
            self.local_ys.append(localydata)

            self.neighbourhoods.append(len(localxdata))
            if i ==0:
                self.first_point_neighbours = [localxdata, localydata]
            X = np.array([localxdata, np.ones(len(localxdata))]).T
            if self.feature_type == 'Linear':
#                wlocal = np.linalg.lstsq(X, localydata, rcond=1)[0]
                model = LinearRegression()
#                model.fit(self.x.reshape(-1,1), self.y, sample_weight=weights)
                model.fit(X, localydata)
                wlocal = [model.coef_[0], model.intercept_]
                w1.append(wlocal[0])
                w2.append(wlocal[1])
                w.append(wlocal)
            elif self.feature_type == 'Cyclic':
                m, c = self.CR.cyclicRegression(localxdata, localydata)
                w1.append(m)
                w2.append(c)
                w.append([m, c])
        inch = 0.3997
        fig2, axes2 = plt.subplots(1,1, figsize=(18*inch, 12*inch))
        for i in range(len(self.x)):
            if i == 20:
                label = 'Local Linear Regression Model'

                axes2.plot(self.local_xs[i], [w1[i]*n_x + w2[i] for n_x in self.local_xs[i]], color='firebrick', linewidth=1, label=r'Local Linear Regression Model', zorder=4)
                axes2.scatter(self.x[i], self.y[i], s=20, color='firebrick', label=r'Instance $x_i$', zorder=5)
                axes2.scatter(self.local_xs[i], self.local_ys[i], color='lightsalmon', s=5, label=r'$N(x_i)$', zorder=3, alpha=0.5)
            else:
                label = '__no_legend__'
#                axes2.plot(self.local_xs[i], [w1[i]*n_x + w2[i] for n_x in self.local_xs[i]], color='firebrick', linewidth=1, label=label)
#            if i ==110:
#                fig, axes = plt.subplots(1,1, figsize=(20*inch, 8*inch))
#                axes.plot(self.local_xs[i], [w1[i]*n_x + w2[i] for n_x in self.local_xs[i]], color='red', linewidth=3)
#                axes.scatter(self.x,self.y, s=10)
#                axes.set_xlabel(r'$x$', fontsize=11)
#                axes.set_ylabel(r'$\hat{y}$')
#                fig.suptitle(r'Local Linear Regression Models', fontsize=11)
#                fig.savefig(f'Figures/LocalLinearRegression_{i}.png', bbox_inches='tight')
        axes2.scatter(self.x,self.y, s=5, color='lightsteelblue', label='Data Instance')
        axes2.set_xlabel(f'Median Income  '+r'$x$', fontsize=12)
        axes2.set_ylabel(r'Median House Value  $\hat{y}$', fontsize=12)
#        fig2.suptitle(r'Local Linear Regression Models', fontsize=11)
        axes2.legend(loc='upper center', fontsize=12, markerscale=1, ncol=2, bbox_to_anchor=(0.5, 1.2), fancybox=True)
        fig2.savefig(f'Figures/LocalLinearRegression_{self.feature_name}.png', bbox_inches='tight', dpi=300)

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
