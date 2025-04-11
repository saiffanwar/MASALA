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
from multiprocessing import Pool, TimeoutError

class LLCGenerator():

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

        self.gather_ensembles(preload_explainer)


    def gather_ensembles(self, preload_explainer=True):

        if  preload_explainer == True:
            self.feature_ensembles = {feature: [] for feature in self.features}

            for feature in self.features:
                with open(f'saved/feature_ensembles/{self.dataset}/{feature}/{self.experiment_id}_{self.coverage_threshold}_{self.neighbourhood_threshold}.pck', 'rb') as file:
                    self.feature_ensembles[feature] = pck.load(file)
        else:
            self.feature_ensembles = None


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
        LLR = LocalLinearRegression(xdata,ydata, dist_function=feature_type)
        w1, w2, w = LLR.calculateLocalModels(self.neighbourhood_threshold)

        D, xDs= LLR.compute_distance_matrix(w, distance_weights=distance_weights)
#        print('Doing K-medoids-clustering')
        # Define number of medoids and perform K medoid clustering.
#        self.sparsity_threshold = np.mean(xDs)
        print(np.mean(xDs))
        self.sparsity_threshold = LLR.meanDistance

        print('Sparsity Threshold:', self.sparsity_threshold)
        LC = LinearClustering(self.dataset, super_cluster_num, xdata, ydata, D, xDs, feature, K,
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
        tic = time.time()
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

        if os.path.isdir(f'saved/feature_ensembles/{self.dataset}/{self.features[i]}'):
            pass
        else:
            os.makedirs(f'saved/feature_ensembles/{self.dataset}/{self.features[i]}')

        with open(f'saved/feature_ensembles/{self.dataset}/{self.features[i]}/{self.experiment_id}_{self.coverage_threshold}_{self.neighbourhood_threshold}.pck', 'wb') as file:
            pck.dump(feature_ensembles, file)
#            if i == len(features)-1:
        toc = time.time()
        print(f'Time taken for one feature: {toc-tic}')

        return i, feature_ensembles


    def multi_layer_clustering(self,num_workers=4):
        print(f'{self.dataset}_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}')
#
##       Check if ensemble exists already and can edit that.
#        if os.path.isfile(f'saved/ensembles/{self.dataset}_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck'):
#            with open(f'saved/ensembles/{self.dataset}_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'rb') as file:
#                self.feature_ensembles = pck.load(file)
#        else:
#            self.feature_ensembles = {feature: [] for feature in self.features}

        # XDs, WDs, neighbourhoodDs

        def distribute_features_to_workers(N, w):
            chunk_size = N // w
            remainder = N % w
            distributed_list = []
            start = 0
            for i in range(w):
                end = start + chunk_size
                if remainder > 0:
                    end += 1
                    remainder -= 1
                distributed_list.append(list(range(start, end)))
                start = end

            return distributed_list

#        [ self.cluster_single_feature(i) for i in range(len(self.features) )]
        pool = Pool(num_workers)
        pool.map(self.cluster_single_feature, range(len(self.features)))
#        pool.map(self.cluster_single_feature, range(4,len(self.features)))
#        for i in range(2,len(self.features)):
#            self.cluster_single_feature(i)
#        results.append(pool.map(self.cluster_single_feature, range(len(self.features))))
#        [results.append(self.cluster_single_feature(i)) for i in range(4,5)]

#        for result in results:
#            for i, feature_ensemble in result:
#                self.feature_ensembles[self.features[i]] = feature_ensemble
##        for i, feature_list in enumerate(distributed_list):
##            print('Starting worker {i}'.format(i=i))
##            pool.apply_async(start_worker, args=(i, feature_list))
#        with open(f'saved/feature_ensembles/temp_{self.dataset}_feature_ensembles_full_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.pck', 'wb') as file:
#            pck.dump(self.feature_ensembles, file)
#
        return self.feature_ensembles

    def plot_all_clustering(self,instance=None, features_to_plot=None, instances_to_show=[]):
        if features_to_plot == None:
            features_to_plot = self.features
        if len(features_to_plot) > 4:
            num_cols=3
        else:
            num_cols=len(features_to_plot)
        num_rows=int(np.ceil(len(features_to_plot)/num_cols))

        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[1/num_cols for i in range(num_cols)],
                            row_heights =[1/num_rows for row in range(num_rows)],
                            specs = [[{} for c in range(num_cols)] for i in range(num_rows)], subplot_titles=features_to_plot,
                            horizontal_spacing=0.05, vertical_spacing=0.05)


        axes = [[row, col] for row in range(1,num_rows+1) for col in range(1,num_cols+1)]

        if len(instances_to_show) != 0:
            matched_instance_colours = [px.colors.qualitative.Alphabet[i % 10] for i in range(len(instances_to_show))]
            instances_to_show_matches = []

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
                colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))
                fig.add_trace(go.Scatter(x=cluster[0],y=cluster[1],
                                         mode='markers', marker = dict(size=3, opacity=0.2, color=colour),
                                         showlegend=showlegend),
                              row=axes[i][0], col=axes[i][1])

                fig.add_trace(go.Scatter(x=cluster[0],y=[params[0]*x+params[1] for x in cluster[0]],
                                         mode='lines', marker = dict(size=3, opacity=0.9, color=colour),
                                         showlegend=showlegend),
                              row=axes[i][0], col=axes[i][1])

#                instances_to_show = None
            if instance != None:
                fig.add_trace(go.Scatter(x=[self.x_test[instance][feature_index]],y=[self.y_pred[instance]],
                                         mode='markers', marker = dict(size=10, opacity=1, color='black'),
                                         showlegend=showlegend),
                              row=axes[i][0], col=axes[i][1])
            if len(instances_to_show) != 0:
                for inst in instances_to_show:
                    fig.add_trace(go.Scatter(x=[self.x_test[inst, feature_index]],y=[self.y_pred[inst]],
                                             mode='markers', marker = dict(size=5, opacity=1, color='black'),
                                             showlegend=showlegend, name=f'Instance {inst}'),
                                  row=axes[i][0], col=axes[i][1])


            min_y = np.min(self.y_pred)
            max_y = np.max(self.y_pred)
            fig.update_xaxes(title='Normalised Feature Value', range=[min([min(self.x_test[:,feature_index])*1.1, -0.1]), max(self.x_test[:,i])*1.1], row=axes[i][0], col=axes[i][1])
            fig.update_yaxes(title='Predicted RUL',range=[min_y*-1.1, max_y*1.1], row=axes[i][0], col=axes[i][1])
        if len(features_to_plot) == 1:
            height = 750
        elif len(features_to_plot) in [2,3]:
            height = 500
        else:
            height=350*num_rows
        fig.update_layout(
                          height=height)
        fig.write_html(f'Figures/{self.dataset}/Clustering/{self.dataset}#_{self.experiment_id}_clustering_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.html')

        return fig

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

    def plot_data(self, plotting_data=None, features_to_plot=None, instances_to_show=None):
        if features_to_plot == None:
            features_to_plot = self.features

        if len(features_to_plot) > 4:
            num_cols=4
        else:
            num_cols=len(features_to_plot)
        num_rows=int(np.ceil(len(features_to_plot)/4))
        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[1/num_cols for i in range(num_cols)], row_heights =[1/num_rows for row in range(num_rows)], specs = [[{} for c in range(num_cols)] for i in range(num_rows)], subplot_titles=features_to_plot, horizontal_spacing=0.1, vertical_spacing=0.05)

        axes = [[row, col] for row in range(1,num_rows+1) for col in range(1,num_cols+1)]

        for feature in features_to_plot:
            value = self.feature_ensembles[feature]
            i = features_to_plot.index(feature)
            feature_index = self.features.index(feature)
            if i == 0:
                showlegend=True
            else:
                showlegend=False
            colour = np.random.rand(3)
            colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))

            if plotting_data == None:
                fig.add_trace(go.Scatter(x=self.x_test[:,feature_index],y=self.y_pred,
                                         mode='markers', marker = dict(size=3, opacity=0.9, color='black'),
                                         showlegend=showlegend, name='Test Data', legendgroup='Test Data'),
                              row=axes[i][0], col=axes[i][1])
            else:
                data_instance, instance_index, local_x, local_x_weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation, instance_cluster_models = plotting_data

                fig.add_trace(go.Scatter(x=self.x_test[:,feature_index],y=self.y_pred,
                                         mode='markers', marker = dict(size=3, opacity=0.9, color='lightgrey'),
                                         showlegend=showlegend, name='Test Data', legendgroup='Test Data'),
                              row=axes[i][0], col=axes[i][1])

                fig.add_trace(go.Scatter(x=[data_instance[feature_index]],y=[instance_prediction],
                                         mode='markers', marker = dict(size=30, opacity=0.9, color='black'),
                                         showlegend=showlegend, name='Instance Model Prediction', legendgroup='Instance Model Prediction'),
                              row=axes[i][0], col=axes[i][1])
                fig.add_trace(go.Scatter(x=[data_instance[feature_index]],y=[exp_instance_prediction],
                                         mode='markers', marker = dict(size=30, opacity=0.9, color='orange'),
                                         showlegend=showlegend, name='Instance Explanation Prediction', legendgroup='Instance Explanation Prediction'),
                              row=axes[i][0], col=axes[i][1])
                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=local_y_pred,
                                         mode='markers', marker = dict(size=3, opacity=0.9, color='red'),
                                         showlegend=showlegend, name='Local Points Model Prediction', legendgroup='Local Points Model Prediction'),
                              row=axes[i][0], col=axes[i][1])
                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=exp_local_y_pred,
                                         mode='markers', marker = dict(size=3, opacity=0.9, color='blue'),
                                         showlegend=showlegend, name='Local Points Explanation Prediction', legendgroup='Local Points Explanation Prediction'),
                              row=axes[i][0], col=axes[i][1])
#                if instances_to_show != None or []:
#                    for inst in instances_to_show:
#                        if inst == instances_to_show[0]:
#                            showlegend=True
#                        else:
#                            showlegend=False
#                        fig.add_trace(go.Scatter(x=[self.x_test[inst,feature_index]],y=[self.y_pred[inst]],
#                                     mode='markers', marker = dict(size=20, opacity=0.9, color='green'),
#                                     showlegend=showlegend, name='Other Instances', legendgroup='Other Instances'),
#                                      row=axes[i][0], col=axes[i][1])


#                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=[instance_cluster_models[i][0]*x+instance_cluster_models[i][1] for x in local_x[:,i]],
#                                         mode='lines', marker = dict(size=3, opacity=0.9, color='black'),
#                                         showlegend=showlegend),
#                              row=axes[i][0], col=axes[i][1])
#                fig.add_trace(go.Scatter(x=local_x[:,feature_index],y=[instance_explanation.coef_[i]*x+instance_explanation.intercept_ for x in local_x[:,i]],
#                                         mode='lines', marker = dict(size=3, opacity=0.9, color='green'),
#                                         showlegend=showlegend),
#                              row=axes[i][0], col=axes[i][1])
                fig.layout.annotations[i].update(text=f'{feature} : {round(data_instance[feature_index], 3)}')
            fig.update_xaxes(title='Normalised Feature Value', range=[min([min(self.x_test[:,feature_index])*1.1, -0.1]), max(self.x_test[:,i])*1.1], row=axes[i][0], col=axes[i][1])
#            fig.update_yaxes(title='Predicted RUL',range=[min_y*-1.1, max_y*1.1], row=axes[i][0], col=axes[i][1])
        if len(features_to_plot) == 1:
            height = 750
        elif len(features_to_plot) in [2,3]:
            height = 600
        else:
            height=350*num_rows
        fig.update_layout(legend=dict(yanchor="top", xanchor="auto", orientation='h', y=-0.25),
                          height=height)
#        fig.write_html(f'Figures/{self.dataset}/perturbations_{instance_index}.html', auto_open=False)
        return fig

