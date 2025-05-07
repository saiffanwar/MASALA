from .distances import calcSingleDistance

import numpy as np
import random
import copy

def smotePerturbations(x_test, y_pred, features, instance, instance_num, num_samples=200):
    # Get the list of features and the training data

#        x_test = x_test[:2000]
#        y_pred = y_pred[:2000]

    # Start the list of perturbations with the instance being perturbed as first item
    perturbedData = [instance]

    # Define some information needed for later:
    dict_x_test = {} # Seperate training data into lists of feature data
    for f, feature in enumerate(features):
        dict_x_test[feature] = [i[f] for i in x_test]
    discreteFeatures = features
    maxVals = [max(val) for val in [dict_x_test[f] for f in dict_x_test.keys()]] # Needed to normalise euclidean distances.
    possValues = [np.unique(val) for val in [dict_x_test[f] for f in dict_x_test.keys()]] # Needed to caculate cyclic distances.
    unique_vals = [np.unique(dict_x_test[f]) for f in features] # Needed to calculate cyclic distances.

    # Calculate distances and weights to assign probabilities when selecting point for interpolation.
    # Distance is calculated in each feature dimension.
    # Loop for number of perturbations required
#        while len(perturbedData) <= num_samples+1:
    for i in range(num_samples):
        perturbation = np.zeros(len(features))
        # Remaining features starts as list of all features to be reduced later.
        remainingFeatures = copy.deepcopy(features)

        interpolationAmount = random.uniform(0,1)
        selectedPoint = random.choices(x_test, k=1)[0]

        # While all features have not been perturbed
        while len(remainingFeatures) != 0:
            # Select a random feature which is yet to be perturbed
            selectedFeature = random.choice(remainingFeatures)
            selectedFeatureIndex = features.index(selectedFeature)

            # Get the value of that feature in the isntance
            instance_value = instance[selectedFeatureIndex]

            # Select a random training data sample based with probability based on the distance to the instance.
            # selectedValue = random.choices((dict_x_test[selectedFeature]), weights=weights, k=1)[0]
            selectedValue = selectedPoint[selectedFeatureIndex]
            # Select a random value between 0 and 1 and adjust the instance value accordingly.
            interpolationDifference = interpolationAmount*(selectedValue - instance_value)
            interpolatedValue = instance_value+interpolationDifference

            # For cyclic features, if interpolated value is in fact further, interpolate in the other direction.
            if calcSingleDistance(instance_value, selectedValue, selectedFeature, maxVals[selectedFeatureIndex], possValues[selectedFeatureIndex]) < calcSingleDistance(instance_value, interpolatedValue, selectedFeature, maxVals[selectedFeatureIndex], possValues[selectedFeatureIndex]):
                interpolatedValue = instance_value-interpolationAmount

            # If selected feature is discrete, adjust the interpolation so it takes the closest existing value.
            if selectedFeature in discreteFeatures:
                interpolatedValue = min(unique_vals[selectedFeatureIndex], key=lambda x:abs(x-interpolatedValue))
            # Mark the working feature as perturbed by removing from the list of features yet to be perturbed.
            remainingFeatures.remove(selectedFeature)
            perturbation[selectedFeatureIndex] = interpolatedValue

        # Once all features have been perturbed, add the perturbation to the list of perturbations.
        perturbedData.append(perturbation)

    return perturbedData
