from LinearClustering import LinearClustering
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly
import plotly.express as px
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from LocalLinearRegression import LocalLinearRegression
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pickle as pck
import random
from tqdm import tqdm
import time
import math
import os
from itertools import combinations
from mpl_toolkits import mplot3d

class LLCExplanation():

    def __init__(self, model, x_test, y_pred, features, discrete_features, dataset, sparsity_threshold=0.5, coverage_threshold=0.05, starting_k=5, neighbourhood_threshold=0.5, preload_explainer=True, experiment_id=1):
        self.model = model
        self.features = features
        self.dataset = dataset
        self.x_test = x_test
        self.y_pred = y_pred
        self.plotting=True
        self.sparsity_threshold = sparsity_threshold
        self.coverage_threshold = coverage_threshold
        self.starting_k = starting_k
        self.neighbourhood_threshold = neighbourhood_threshold
        self.ploting=False
        self.discrete_features = discrete_features
        self.experiment_id = experiment_id

        self.gather_ensembles()
#        if preload_explainer:
#            print('Loading', f'saved/feature_ensembles/{self.dataset}_feature_ensembles_full_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck')
#            with open(f'saved/feature_ensembles/adaptive_{self.dataset}_feature_ensembles_full_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'rb') as file:
##            with open(f'saved/{self.dataset}_feature_ensembles_full.pck', 'rb') as file:
#                self.feature_ensembles = pck.load(file)
#        else:
#            self.feature_ensembles = None


    def gather_ensembles(self,):

        self.feature_ensembles = {feature: [] for feature in self.features}

        for feature in self.features:
            with open(f'saved/feature_ensembles/{self.dataset}/{feature}/{self.experiment_id}_{self.coverage_threshold}_{self.neighbourhood_threshold}.pck', 'rb') as file:
                self.feature_ensembles[feature] = pck.load(file)

    def generate_cluster_ranges(self, ):
        cluster_ranges = {f: [] for f in self.features}
        instance_cluster_models = {f: [] for f in self.features}
        feature_ensembles = self.feature_ensembles
        for f in self.features:
            ensemble_data = feature_ensembles[f]
            clustered_data, linear_models = ensemble_data
            all_xs = []
            for x in [cluster[0] for cluster in clustered_data]:
                all_xs.extend(x)
            for cluster, linear_model in zip(clustered_data, linear_models):
                cluster_xs = cluster[0]
                cluster_range = [min(cluster_xs), max(cluster_xs)]
                cluster_ranges[f].append(cluster_range)
                instance_cluster_models[f] = linear_model
        return cluster_ranges, instance_cluster_models

    def find_ranges(self, x, ranges):
        for i, (min_val, max_val) in enumerate(ranges):
            if min_val <= x <= max_val:
                return i  # Return the index of the range
        return None

    def generate_cluster_assignments(self, cluster_ranges, x):
        cluster_assignments = {f:None for f in self.features}
#        cluster_assignments = []
        assigned = 0
        for i, f in enumerate(self.features):
            cluster_assignments[f] = self.find_ranges(x[i], cluster_ranges[f])
            if cluster_assignments[f] is not None:
                assigned += 1
        if assigned != len(self.features):
            print(cluster_ranges, x)
            print(cluster_assignments)
            print('Cluster assignment error')
#            print(cluster_assignments)
#            print(x)
#            [print(f, cluster_ranges[f]) for f in self.features]
            raise ValueError
        return cluster_assignments


    def find_cluster_matches(self, data_instance, cluster_ranges, instance_assignments, match_tolerance=0):
        match_threshold = len(self.features) - match_tolerance
        print('\n')
        cluster_matches = []
        local_x, local_y_pred = [], []
        matched_instances = []
        for i, x in enumerate(self.x_test):
            assignments = self.generate_cluster_assignments(cluster_ranges, x)
            num_matches = 0
            for f in instance_assignments.keys():
                if instance_assignments[f] == assignments[f]:
                    num_matches += 1
            cluster_matches.append(num_matches)

        while len(matched_instances) < 5:
            matched_instances = np.argwhere(np.array(cluster_matches) >= match_threshold).flatten()
#            matched_instances = self.x_test[matched_instances]
            match_threshold -= 1

        local_x = np.array([self.x_test[i] for i in matched_instances])
        local_y_pred = np.array([self.y_pred[i] for i in matched_instances])

        return cluster_matches, local_x, local_y_pred, matched_instances


    def generate_explanation(self, data_instance, instance_index, instance_prediction=None, ground_truth=None):

        exp_prediction = 0

        cluster_ranges, instance_cluster_models = self.generate_cluster_ranges()
        instance_cluster_assignments = self.generate_cluster_assignments(cluster_ranges, data_instance)

        cluster_matches, local_x, local_y_pred, matched_instances = self.find_cluster_matches(data_instance, cluster_ranges, instance_cluster_assignments)
        print(f'Number of local points: {len(matched_instances)}')
        norm_cluster_matches = [m/max(cluster_matches) for m in cluster_matches]

        weights = [1 for x in local_x]
        instance_explanation_model = LinearRegression()
        instance_explanation_model.fit(local_x, local_y_pred, sample_weight=weights)
        exp_local_y_pred = instance_explanation_model.predict(local_x)
        exp_instance_prediction = instance_explanation_model.predict([data_instance])[0]
#        print(f'Ground Truth: {ground_truth}')
#        print(f'Model Prediction: { instance_prediction }')
        print(f'LLC Explanation prediction: { exp_instance_prediction }')

        plotting_data = [data_instance, instance_index, local_x, weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models]
#        fig = self.plot_data(plotting_data)
        return exp_instance_prediction, plotting_data, matched_instances

    def evaluate_explanation(self, exp_model, instance, method, instance_index):
        ''' This function measures the robustness of an explanation by adding noise to different features to monitor
        the change in the model prediction. If the variance in the prediction decreases as less important features
        are permuted, then the explanation is robust'''
        feature_contributions = exp_model.coef_
        contribution_magnitudes = [abs(item) for item in feature_contributions]

        original_prediction = self.model.predict([instance])[0]

        ordered_contribution_size_index = np.argsort(contribution_magnitudes)
        deviations = []
        for f in ordered_contribution_size_index:
            feature_noise = np.random.normal(0, 0.001, 1)
            noise = np.zeros(len(instance))
            noise[f] = feature_noise

            noisey_instance = instance + noise
            for f in range(len(noisey_instance)):
                poss_values = np.unique(self.x_test[:,f])
                noisey_instance[f] = poss_values[np.argmin(np.abs(poss_values - noisey_instance[f]))]
            noisy_prediction = self.model.predict([noisey_instance])[0]
            prediction_deviation = abs(original_prediction - noisy_prediction)
            deviations.append(prediction_deviation)

#        deviations = [item/max(deviations) for item in deviations]
#        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
#        axes.plot([i for i in range(len(deviations))], deviations)
##        axes.set_yscale('log')
#        fig.savefig(f'Figures/Robustness{method}_{instance_index}.pdf')

        return deviations

    def new_evaluation(self, instance_index, exp_model, primary_instance, importance_threshold=5):
#        important_features = [f for f in self.features if abs(exp_model.coef_[self.features.index(f)]) > importance_threshold]
        important_features = self.features
        only_important_x = self.x_test[:,[self.features.index(f) for f in important_features]]


        # Weighted euclidean distance
        def weighted_euclidean_distance(data):
            x, y , weights = data
            squared_diff = (x - y) ** 2
            weighted_squared_diff = squared_diff * weights
            distance = np.sqrt(np.sum(weighted_squared_diff))
            return distance
        weights = [abs(c) for c in exp_model.coef_]
        distances = [weighted_euclidean_distance((primary_instance, x, weights)) for x in only_important_x]
#        print(distances)
        closest_instances = list(np.argsort(distances))


        def absolute_difference(data):
            x, y = data
            sum_of_diffs = 0
            for i in range(len(x)):
                sum_of_diffs += abs(x[i] - y[i])

            return sum_of_diffs
        distances = [absolute_difference((primary_instance, x)) for x in self.x_test]
        closest_instances = list(np.argsort(distances))



        return closest_instances
#        distances =

##
#        instance_cluster = labels[instance_index]
##        instance_cluster_x = only_important_x[labels == instance_cluster]
#        instance_cluster_x = np.argwhere(labels == instance_cluster).flatten()
#
#        return instance_cluster_x


    def plot_explanation(self, plotting_data=None, features_to_plot=None):
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Feature Contributions', 'Local Feature Significance'], horizontal_spacing=0.25)
        if features_to_plot != None:
            data_instance, instance_index, local_x, weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = plotting_data
            explained_feature_indices = [f for f in range(len(self.features)) if self.features[f] in features_to_plot]
            explained_features = [self.features[e] for e in explained_feature_indices]
            feature_contributions = instance_explanation_model.coef_[explained_feature_indices]
            local_feature_significance = [instance_cluster_models[f][0] for f in explained_features]

            colours = ['green' if x>= 0 else 'red' for x in feature_contributions]
            fig.add_trace(go.Bar(x=feature_contributions, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=local_feature_significance, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=2)
            fig.update_layout(height=200+20*len(features_to_plot), title=dict(text=f'Ground Truth {ground_truth} <br> Model Prediction: {round(instance_prediction,2)} <br> Explanation Prediction: {round(exp_instance_prediction, 2)} <br>', y=0.95))
        else:
            fig.update_layout(height=10)

        return fig

    def interactive_exp_plot(self, instance, instance_index,  instance_model_prediction, y_test_pred, instance_exp_prediction, exp, local_points, local_weights, model_local_points_prediction, exp_local_points_prediction, targetFeature, neighbours=None):


        explanation_error = mean_squared_error(model_local_points_prediction, exp_local_points_prediction)

        num_rows=int(np.ceil(len(self.features)/4))+1
        num_cols=4

        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[0.25, 0.25, 0.25, 0.25], row_heights =[0.33]+[0.16]*(num_rows-1),
                            specs = [
                                [{'colspan':2}, None, {'colspan':2}, None],
                                ]+[[{}, {}, {}, {}] for i in range(num_rows-1)], subplot_titles=['Explanation Prediction Convergence', 'Feature Significance']+self.features,
                            horizontal_spacing=0.05, vertical_spacing=0.05)

        colours = ['green' if x>= 0 else 'red' for x in exp]

        # Plot convergence of error as features are added
#        exp_convergence = []
#        for included_features in range(len(self.features)):
#            intercept = exp.intercept
#            for i in range(included_features):
#                intercept+= feature_contributions[i]*instance_x[i]
#
#            exp_convergence.append(intercept)
#
#        fig.add_trace(go.Scatter(x=[num for num in range(len(exp_convergence))], y=exp_convergence, mode='lines', marker = dict(color='orange', size=3), showlegend=False), row=1, col=1)


        # Plot explanation bar chart
        fig.add_trace(go.Bar(x=exp, y=self.features, marker_color=colours, orientation='h', showlegend=False), row=1, col=3)

        axes = [[row, col] for row in range(2,num_rows+1) for col in range(1,num_cols+1)]

#        for n in range(len(neighbours)):
#            neighbours[n] = [neighbours[n][i] for i in explained_feature_indices]
#
        for i in range(len(self.features)):
#        fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_exp_y, mode='markers', marker = dict(color='orange', size=3)), row=ax[0], col=ax[1])
            if i==0:
                showlegend=True
            else:
                showlegend=False
            fig.add_trace(go.Scatter(x=self.x_test[:,i],y=y_test_pred,
                                     mode='markers', marker = dict(color='lightgrey', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Test data'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=local_points[:,i],y=model_local_points_prediction,
                                     mode='markers', marker = dict(color=local_weights, colorscale='Oranges', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Model (f) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=local_points[:,i],y=exp_local_points_prediction,
                                     mode='markers', marker = dict(color=local_weights, colorscale='Greens', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Explanation (g) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance[i]],y=[instance_model_prediction],
                                     mode='markers', marker = dict(color='red', size=20),
                                     showlegend=showlegend, name='Instance being explained'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance[i]],y=[instance_exp_prediction],
                                     mode='markers', marker = dict(color='blue', size=10),
                                     showlegend=showlegend, name='Instance being explained'),
                          row=axes[i][0], col=axes[i][1])


        fig.update_layout(title=dict(text = f' Explanation for instance {instance} <br> Explanation Error = {explanation_error:.2f} <br> Model Instance Prediction {instance_model_prediction} <br> Explanation Instance Prediction {instance_exp_prediction}', y=0.99, x=0),
                          font=dict(size=14),
                          legend=dict(yanchor="top", y=1.1, xanchor="right"),
                          height=300*num_rows, )
#        if self.automated_locality == True and self.newMethod == True:
#            suffix = 'LLC'
#        elif self.automated_locality == False and self.newMethod == True:
#            suffix = '_CHILLI'
#        elif self.automated_locality == False and self.newMethod == False:
#            suffix = '_LIME'
#        else:
#            suffix = ''
        suffix = 'LLC'
        fig.write_html(f'Figures/{self.dataset}/Explanations/instance_{instance_index}_{suffix}_explanation.html', auto_open=False)
