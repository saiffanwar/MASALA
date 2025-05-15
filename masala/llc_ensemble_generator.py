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
from multiprocessing import Pool, TimeoutError

from .LinearClustering import LinearClustering
from .LocalLinearRegression import LocalLinearRegression

class LLCGenerator():

    def __init__(self, model, model_type, x_test, y_pred, features, target_name, discrete_features, dataset, sparsity_threshold=0.5, coverage_threshold=0.05, starting_k=5, neighbourhood_threshold=0.5, experiment_id=1, num_workers=1):
        self.model = model
        self.model_type = model_type
        self.features = features
        self.target_name = target_name
        self.dataset = dataset
        self.x_test = x_test
        self.y_pred = y_pred
        self.plotting=True
        self.sparsity_threshold = sparsity_threshold
        self.coverage_threshold = coverage_threshold
        self.starting_k = starting_k
        self.neighbourhood_threshold = neighbourhood_threshold
        self.discrete_features = discrete_features
        self.experiment_id = experiment_id
        self.num_workers = num_workers

        self.generate_ensembles()


    def generate_ensembles(self, preload_clustering=True):

            self.multiworker_clustering(num_workers=self.num_workers)
            self.feature_ensembles = {feature: [] for feature in self.features}

            for feature in self.features:
                with open(f'saved/feature_ensembles/{self.dataset}/{self.model_type}/{feature}/{self.experiment_id}_{self.sparsity_threshold}_{self.starting_k}.pck', 'rb') as file:
                    self.feature_ensembles[feature] = pck.load(file)


    def feature_space_clustering(self, feature_xs):
        # Create a kernel density estimation model
        kde = KernelDensity(bandwidth=0.25*max(feature_xs))  # You can adjust the bandwidth
        kde.fit(np.array(feature_xs).reshape(-1, 1))

        # Create a range of data points for evaluation
        x_eval = np.linspace(min(feature_xs), max(feature_xs), 1000)
        log_dens = kde.score_samples(x_eval.reshape(-1, 1))
        dens = np.exp(log_dens)

        # Find local maxima in the density curve
        peaks, _ = find_peaks(dens)
        cluster_centers = x_eval[peaks]

        # Assign data points to clusters based on proximity to cluster centers
        cluster_assignments = []
        for data_point in feature_xs:
            distances = np.abs(cluster_centers - data_point)
            nearest_cluster = np.argmin(distances)
            cluster_assignments.append(nearest_cluster)

        return cluster_assignments

    def plot_final_clustering(self, clustered_data, linear_params):
#    cost = self.calculate_clustering_cost(clustered_data)
        fig = go.Figure()
        for cluster in range(len(clustered_data)):
            colour = np.random.rand(3)
            colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))
            xs, ys = clustered_data[cluster]
            w, b = linear_params[cluster]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=4, color=colour)))
            cluster_range = np.linspace(min(xs), max(xs), 100)
            fig.add_trace(go.Scatter(x=cluster_range, y=w*cluster_range+b, mode='lines', line=dict(color=colour, width=5)))
#    fig.update_layout(title=f'K-Medoids clustering of LLR models into {len(clustered_data)} clusters \n  Clustering Cost: {cost:.2f}')
        return fig

    def plot_clustering_matplotlib(self, clustered_data, linear_params):
#    cost = self.calculate_clustering_cost(clustered_data)
        fig = go.Figure()
        for cluster in range(len(clustered_data)):
            colour = np.random.rand(3)
            colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))
            xs, ys = clustered_data[cluster]
            w, b = linear_params[cluster]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=4, color=colour)))
            cluster_range = np.linspace(min(xs), max(xs), 100)
            fig.add_trace(go.Scatter(x=cluster_range, y=w*cluster_range+b, mode='lines', line=dict(color=colour, width=5)))
#    fig.update_layout(title=f'K-Medoids clustering of LLR models into {len(clustered_data)} clusters \n  Clustering Cost: {cost:.2f}')
        return fig

    def linear_clustering(self, super_cluster_num, xdata, ydata, feature, distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}
    , K=20, feature_type='Linear'):
#        print('Performing Local Linear Regression')
        # Perform LocalLinear regression on fetched data
        LLR = LocalLinearRegression(xdata, ydata, dist_function=feature_type, feature_name=feature)
        print('#-- Performing Local Linear Regression --#')
        w1, w2, w = LLR.calculateLocalModels(self.neighbourhood_threshold)

        D, xDs= LLR.compute_distance_matrix(w, distance_weights=distance_weights)

#        self.global_density = len(xdata)/abs(min(xdata)-max(xdata))
#        self.sparsity_threshold = 0.1*self.global_density
#        self.coverage_threshold = self.global_density

        LC = LinearClustering(self.dataset, super_cluster_num, xdata, ydata, D, xDs, feature, self.target_name, K,
                              sparsity_threshold = self.sparsity_threshold,
                              coverage_threshold=self.coverage_threshold,
                              gaps_threshold=0.1,
                              feature_type=feature_type,
                              experiment_id=self.experiment_id,
                              )

        clustered_data, medoids, linear_params, clustering_cost, fig = LC.adapted_clustering()

        return clustered_data, medoids, linear_params, clustering_cost

    def cluster_single_feature(self, i):
        print('Starting Clustering for Feature {i}'.format(i=i))
        x_test = self.x_test
        y_pred = self.y_pred
        features = self.features
        distance_weights={'x': 1, 'w': 1, 'neighbourhood': 1}

        feature_ensembles = []
#            print(f'--------{features[i]} ({i} out of {len(features)})---------')
        all_feature_clusters = []
        all_feature_linear_params = []
        feature_xs = x_test[:, i]
        if features[i] not in self.discrete_features:
            cluster_assignments = self.feature_space_clustering(feature_xs)
            K=self.starting_k
        else:
            cluster_assignments = np.zeros(len(feature_xs))
            K=1


        for super_cluster in np.unique(cluster_assignments):
            super_clsuter = int(super_cluster)
            super_cluster_x_indices = np.array(np.argwhere(cluster_assignments == super_cluster)).flatten()
            super_cluster_xs = feature_xs[super_cluster_x_indices]
            super_cluster_y_pred = y_pred[super_cluster_x_indices]

            clustered_data, medoids, linear_params, clustering_cost = self.linear_clustering(super_cluster, super_cluster_xs, super_cluster_y_pred, features[i], distance_weights, K=K, feature_type='Linear')
            [all_feature_clusters.append(cluster) for cluster in clustered_data]
            [all_feature_linear_params.append(linear_param) for linear_param in linear_params]

        if self.plotting==True:
            K=len(all_feature_clusters)
            fig = self.plot_final_clustering(all_feature_clusters, all_feature_linear_params)
            fig.write_html(f'Figures/{self.dataset}/Clustering/OptimisedClusters/{features[i]}_final_{K}.html')
        lens = []
        for j in range(len(all_feature_clusters)):
            lens = np.sum([len(cluster[0]) for cluster in all_feature_clusters])

        feature_ensembles = [all_feature_clusters, all_feature_linear_params]

        if os.path.isdir(f'saved/feature_ensembles/{self.dataset}/{self.model_type}/{self.features[i]}'):
            pass
        else:
            os.makedirs(f'saved/feature_ensembles/{self.dataset}/{self.model_type}/{self.features[i]}')

        with open(f'saved/feature_ensembles/{self.dataset}/{self.model_type}/{self.features[i]}/{self.experiment_id}_{self.sparsity_threshold}_{self.starting_k}.pck', 'wb') as file:
            pck.dump(feature_ensembles, file)
#            if i == len(features)-1:

        return i, feature_ensembles

    def multiworker_clustering(self,num_workers):
#
#        def distribute_features_to_workers(N, w):
#            chunk_size = N // w
#            remainder = N % w
#            distributed_list = []
#            start = 0
#            for i in range(w):
#                end = start + chunk_size
#                if remainder > 0:
#                    end += 1
#                    remainder -= 1
#                distributed_list.append(list(range(start, end)))
#                start = end
#
#            return distributed_list

#        [ self.cluster_single_feature(i) for i in range(len(self.features) )]
        if num_workers == 1:
            for i in range(len(self.features)):
#            for i in [1]:
                self.cluster_single_feature(i)
        else:
            pool = Pool(num_workers)
            pool.map(self.cluster_single_feature, range(len(self.features)))
#


#
    def matplot_all_clustering(self,instance=None, features_to_plot=None, instances_to_show=[]):

        if features_to_plot == None:
            features_to_plot = self.features
        if len(features_to_plot) > 4:
            num_cols=3
        else:
            num_cols=len(features_to_plot)
        num_rows=int(np.ceil(len(features_to_plot)/num_cols))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 6))

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.1,
                            hspace=0.4)
        axes = fig.get_axes()

#        if len(instances_to_show) != 0:
#            matched_instance_colours = [px.colors.qualitative.Alphabet[i % 10] for i in range(len(instances_to_show))]
#            instances_to_show_matches = []

        for feature in features_to_plot:
            value = self.feature_ensembles[feature]
            feature_index = self.features.index(feature)
            i = features_to_plot.index(feature)
            clustered_data, linear_params = value
            showlegend=False
            instanceFound = True
            instance_clusters = []
            all_feature_points = []
            for cluster, params in zip(clustered_data, linear_params):
                colour = np.random.rand(3)
                axes[i].scatter(cluster[0], cluster[1], s=1, alpha=0.2, c=colour)
                axes[i].plot(cluster[0], [params[0]*x+params[1] for x in cluster[0]], c=colour)

#                fig.add_trace(go.Scatter(x=cluster[0],y=[params[0]*x+params[1] for x in cluster[0]],
#                                         mode='lines', marker = dict(size=3, opacity=0.9, color=colour),
#                                         showlegend=showlegend),
#                              row=axes[i][0], col=axes[i][1])

#                instances_to_show = None


            min_y = np.min(self.y_pred)
            max_y = np.max(self.y_pred)
            axes[i].set_xlabel(feature)
            if i in [0,3,6]:
                axes[i].set_ylabel('Predicted Air Temperature')
            else:
                axes[i].set_ylabel('')
                axes[i].set_yticklabels([''])

#            fig.update_xaxes(title='Normalised Feature Value', range=[min([min(self.x_test[:,feature_index])*1.1, -0.1]), max(self.x_test[:,i])*1.1], row=axes[i][0], col=axes[i][1])
#            fig.update_yaxes(title='Predicted RUL',range=[min_y*-1.1, max_y*1.1], row=axes[i][0], col=axes[i][1])
        fig.set_facecolor((245/255, 245/255, 241/255))
        fig.savefig(f'Figures/{self.dataset}/Clustering/{self.dataset}#_{self.experiment_id}_clustering_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pdf', bbox_inches='tight')


