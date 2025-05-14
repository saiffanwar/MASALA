from sqlite3 import Time
import numpy as np
import datetime
import pandas as pd
import math
from scipy.special import softmax
import pickle as pck
from pprint import pprint
import json


with open('feature_categories.json', 'r') as f:
    feature_dict = json.load(f)

euclideanFeatures = feature_dict["euclideanFeatures"]

#euclidean_features_2
cyclicFeatures = feature_dict["cyclicFeatures"]

# Cyclic features are calculated with distances going in a circle.
def cyclic(x1, x2, max_val):
#    x1, x2 = [x*len(possValues) for x in [x1,x2]]
    diff = abs(x1-x2)
    return min(max_val - diff, diff)

# Binary features are distanced as 0 or 1
def binary(instance, perturbation):
    return(int(instance != perturbation))


# Function to calculate the distance between 2 values for a single feature
def calcSingleDistance(instanceValue, perturbValue, feature, maxVal, possVal):
    if any(f in feature for f in euclideanFeatures):
        distance = abs(instanceValue - perturbValue)/maxVal
    if any(f in feature for f in cyclicFeatures):
        distance = cyclic(instanceValue, perturbValue, maxVal)
    return distance

# Function to calculate distances per feature for a full set of data. Returns a dictionary of distances with feature keys.
def calcAllDistances(instance, perturbations, features):
    # Seperated data by features.
    featureSeperatedData = [[p[f] for p in perturbations] for f in range(len(features))]
    # Calculates information needed for some distance metrics.
    maxVals = [max(val) for val in featureSeperatedData]
    possVals = [np.unique(val) for val in featureSeperatedData]
    # Initialise distance dictionary.
    distances = {}
    for f, feature in enumerate(features):
        # These features are calculated as a euclidean distance normalised by the maximum possible value.
        if any(f in feature for f in euclideanFeatures):
            maxVal = maxVals[f]
            def euclidean(num2):
                dist =  abs(instance[f]-num2)/maxVal
                if math.isnan(dist):
                    return 0
                else:
                    return dist
            distances[feature] = list(map(euclidean, [i[f] for i in perturbations]))
        # # These features are binary features.
        # elif feature in ['Bank Holiday', 'Weekend', 'Monday', 'Morning', 'Afternoon', 'Evening', 'Night']:
        #     distances[feature] = [binary(instance[f], i[f]) for i in perturbations]
        # These features are cyclic features.
        if any(f in feature for f in cyclicFeatures):
            possVal = possVals[f]
            max_val = maxVals[f]
            distances[feature] = [cyclic(instance[f], i[f], max_val) for i in perturbations]
    return distances


def distanceToWeights(distances):
    weights = {}
    for feature in distances.keys():
        with open('distances.pck', 'wb') as file:
            pck.dump(distances, file)
        # Decreasing kernel width decreases RMSE because weightings are smaller so contributions are smaller.
        # kernel_width = 1
        # kernel = lambda distance : np.exp(-(distance ** 2)/(kernel_width ** 2))
        # weights[feature] = list(map(kernel,distances[feature]))

        maxVal = max(distances[feature])
        dist2weight = lambda dist : (maxVal - dist)
        weights[feature] = list(map(dist2weight, distances[feature]))

        normaliser = lambda weight : weight/sum(weights[feature])
        weights[feature] = list(map(normaliser, weights[feature]))
    return weights

def combinedFeatureWeighting(weights):
    allWeights = []
    for feature in weights.keys():
        allWeights.append(weights[feature])
    combinedWeights = [np.mean(i) for i in zip(*allWeights)]
    # If None, defaults to sqrt(number of columns) * 0.75
# np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    # kernel_width = 0.75
    # kernel = lambda weight : np.exp(-(weight ** 2)/(kernel_width ** 2))
    # combinedWeights = list(map(kernel,combinedWeights))
    return combinedWeights

def distanceToWeightsList(distances):
    maxVal = max(distances)
    dist2weight = lambda dist : (maxVal - dist)
    weights = list(map(dist2weight, distances))
    # normaliser = lambda weight : weight/sum(weights)
    # weights = list(map(normaliser, weights))
    return weights


def combinedFeatureDistances(distances):

    allDistances = []
    for feature in distances.keys():
        normaliser = lambda dist : dist/sum(distances[feature])
        # allDistances.append(list(map(normaliser, distances[feature])))
        allDistances.append(distances[feature])
    combinedDistances = [np.mean(i) for i in zip(*allDistances)]

    return combinedDistances


def pointwiseDistance(x1, x2, features):
    distances = []
    for i, f in enumerate(features):
        if f in euclideanFeatures:
            distances.append(abs(x1[i] - x2[i]))
        if f in cyclicFeatures:
            distances.append(cyclic(x1[i], x2[i], 1))

    return np.mean(distances)

