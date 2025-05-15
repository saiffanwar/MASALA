from os import wait
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import compress
from pandas.core.base import NoNewAttributesMixin
from sklearn.metrics import mean_squared_error
import statistics as stat
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from .cyclicRegression import CyclicRegression
from copy import deepcopy
import os
import glob
import time
import warnings
from pprint import pprint
import plotly.graph_objects as go
import plotly


warnings.filterwarnings("ignore")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})





class LinearClustering():

    ''' This class clusters a dataset into regions of linearity. It is provided a distance matrix computed in the LocalLinearRegression class.
    This distance matrix is based on the local linear models of each point as well as the raw distance values. '''

    def __init__(self, dataset, super_cluster, x, y, D, xDs, feature, target_name, K,
                sparsity_threshold,
                coverage_threshold,
                gaps_threshold,
                feature_type='Linear',
                experiment_id=1,
                ):

        self.x = x
        self.y = y
        self.D = np.array(D)
        self.xDs = xDs
        self.feature_name = feature
        self.target_name = target_name
        self.clustering_cost = None
        self.clustered_data = None
        self.medoids = None
        self.K = K
        self.init_K = K
        self.sparsity_threshold = sparsity_threshold
#        self.similarity_threshold = similarity_threshold
        self.coverage_threshold = coverage_threshold
        self.gaps_threshold = 0.1*abs(max(self.x)-min(self.x))
#        self.gaps_threshold=0.1
        self.dataset = dataset
        self.super_cluster = super_cluster
        self.feature_type = feature_type
        self.plotting=True
        self.experiment_id = experiment_id
        self.plotting_dir = f'Figures/{self.dataset}/Clustering/OptimisedClusters/{self.feature_name}'

        # Clear folder for figures to show evolution of clustering.

        if os.path.isdir(self.plotting_dir):
            pass
        else:
            os.makedirs(self.plotting_dir)

        files = glob.glob(self.plotting_dir+'/*')
        for f in files:
            os.remove(f)

    def LR(self, x, y):
        if self.feature_type == 'Linear':
            x = np.array(x).reshape(-1,1)
            y = np.array(y)
            reg = LinearRegression().fit(x, y)
            return reg.coef_, reg.intercept_
        elif self.feature_type == 'Cyclic':
            CR = CyclicRegression(boundary=max(self.x))
            m, c = CR.cyclicRegression(x, y)
            return m, c

    def find_gaps(self, x=None):
        '''Find gaps in the data and do not cluster across these gaps'''
        if self.gaps_threshold == None:
            return []
        x = np.sort(x)
        if len(np.unique(x)) > 1:
            all_x_differences = [x[i+1]-x[i] for i in range(len(x)-1)]
            median_x_difference = stat.mean(all_x_differences)
            gap_
            gap_indices = [i for i in range(len(all_x_differences)) if all_x_differences[i] > self.gaps_threshold*median_x_difference]
            gap_ranges = [[x[i], x[i+1]] for i in gap_indices]
        else:
            gap_ranges = []

        return gap_ranges

    def pick_medoids(self, K):
        '''Pick K medoids from the data that are evenly spaced across the data.'''
        sorted_x = np.sort(self.x)
        medoids = np.linspace(0, len(self.x)-1, K)
        medoids = [list(self.x).index(sorted_x[int(med)]) for med in medoids]
        return medoids

    def find_medoids(self, clustered_data):
        medoids = []
        for cluster in clustered_data:
            if cluster == []:
                medoids.append(None)
            else:
                distMatrix = np.zeros((len(cluster), len(cluster)))
                for i in range(len(cluster)):
                    for j in range(len(cluster)):
                        distMatrix[i,j] = self.D[cluster[i], cluster[j]]
                medoid = np.argmin(np.sum(distMatrix, axis=1))
                medoids.append(cluster[medoid])
        return medoids

    def adapted_clustering(self,init_clusters=True, clustered_data=None, medoids=None, clustering_cost=None, linear_params=None):

        '''
        This is the main clustering algorithm. It will start with some arbitrary number of K to generate clusters.
        It takes a K-medoids based approach to generate clusters using the provided distance matrix. It then checks if there
        are any clusters contained within other clusters. If so, it merges the child cluster into the parent cluster.
        Then it checks if there are any neighbouring clusters which have a similar linearity and merges those that do.
        Then based on the new number of clusters after merging, it will recluster the data to find the optimal clustering.
        This is done recursively until there are no new merges therefore an optimal value of K is found.
        '''

        K = self.K
        num_iterations = 1000
        stopping_criteria = 10
        no_change = 0
        iter = 0
        self.colours = [np.random.rand(1,3) for i in range(K)]

        new_clustering_cost = None
#        init_clusters = True
#        if K == 1:
#            recluster = False
#        else:
        recluster = True
#        self.gaps = self.find_gaps(self.x)
        if not self.plotting:
            fig=None
#        while iter < num_iterations:
        while iter < num_iterations and recluster == True:
#            print(iter, recluster)

            if init_clusters == True:

                if self.medoids == None:
                    medoids = self.pick_medoids(K)
                else:
                    medoids = deepcopy(self.medoids)

                # Reset clustered data for new clustsering.
                clustered_data = [[med] for med in medoids]
                minimum_distances = []
                for index,i in enumerate(self.D):
                    if index not in medoids:
                        minimum_distances.append(min([i[j] for j in medoids]))
                        closest = np.argmin([i[j] for j in medoids])
                        clustered_data[closest].append(index)
                _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
                init_clusters = False
                old_clustering_cost = clustering_cost

            else:
                old_clustering_cost = clustering_cost
                # Recluster the data to find the optimal clustering.


                if recluster == True:
                    for c_num in range(K):
                        lowest_cost_medoid = medoids[c_num]
                        lowest_cost = clustering_cost
                        cluster = clustered_data[c_num]
                        possible_medoids = random.sample(cluster,1)
                        for med in possible_medoids:
                            new_medoids = deepcopy(medoids)
                            new_medoids[c_num] = med
                            new_clustered_data = self.gen_clustering(new_medoids)
                            _,new_linear_params, new_clustering_cost = self.calc_cluster_models(self.x, self.y, new_clustered_data)
                            if new_clustering_cost < lowest_cost:
                                lowest_cost = new_clustering_cost
                                lowest_cost_medoid = med
                                medoids = new_medoids
                                clustered_data = new_clustered_data

#                    clustered_data = self.gen_clustering(medoids)
                    _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
                    if clustering_cost == old_clustering_cost:
                        no_change += 1
                        if no_change > stopping_criteria:
                            recluster = False
                    else:
                        no_change = 0
                        old_clustering_cost = clustering_cost
#                    recluster = False
#                clustering_cost, linear_params, clustered_data, medoids, recluster = self.recluster(K, clustered_data, medoids, clustering_cost, linear_params)

            if len(clustered_data) == 1:
                if self.plotting:
                    fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
                    fig.savefig(self.plotting_dir+f'/final_{K}.pdf', bbox_inches='tight')
                clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)
                return clustered_datapoints, medoids, linear_params, clustering_cost, fig
            iter+=1


        if self.plotting:
            fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
            fig.savefig(self.plotting_dir+f'/before_verifying_{K}.pdf', bbox_inches='tight')

        pre_verification_clustering = clustered_data
        clustered_data, medoids, linear_params, clustering_cost, changes = self.verify_clustering(clustered_data)

        self.clustered_data = deepcopy(clustered_data)
        self.medoids = deepcopy(medoids)
        self.linear_params = deepcopy(linear_params)
        self.clustering_cost = deepcopy(clustering_cost)

        if self.plotting:
            fig = self.plotMedoids(self.clustered_data, self.medoids, self.linear_params, self.clustering_cost)
            fig.savefig(self.plotting_dir+f'/after_verifying_{K}_merged_to_{len(self.clustered_data)}.pdf', bbox_inches='tight')

        # If the number of clusters has changed, pick new medoids from the same clusters and optimise further.
        if len(self.clustered_data) < K:
            K = len(self.clustered_data)
            self.K = K
            return self.adapted_clustering(False, self.clustered_data, self.medoids,self.clustering_cost, self.linear_params)
        else:
            if self.plotting:
                fig = self.plot_final_clustering(self.clustered_data, self.linear_params)
                fig.write_html(self.plotting_dir+f'/final_{K}.html')
                fig = self.plotMedoids(self.clustered_data, self.medoids, self.linear_params, self.clustering_cost)
                fig.savefig(self.plotting_dir+f'/final_{K}.pdf', bbox_inches='tight')
            clustered_datapoints = self.cluster_indices_to_datapoints(self.clustered_data)
            return clustered_datapoints, self.medoids, self.linear_params, self.clustering_cost, fig

        #  Check if there is any clusters contained within other clusters. If so, merge the child cluster into the parent cluster and recalculate the           new cluster model parameters.


    def gen_clustering(self, medoids):
        new_clustered_data = [[new_med] for new_med in medoids]
        for index,i in enumerate(self.D):
            closest = np.argmin([i[j] for j in medoids])
            new_clustered_data[closest].append(index)
        return new_clustered_data


    def recluster(self, K, clustered_data, medoids, clustering_cost, linear_params):

        '''
        This is a utility function used in the adapted clustering algorithm. It will randomly select a cluster
        and then randomly select a new medoid for the cluster and then assigning all points the appropriate cluster
        based on the new medoid. Linear Regression models are fit to all of the new clusters. A cost is calculated
        based on the combined error of all the new LR models. If this cost is lower than the previous cost, the new
        clustering is favoured. This function is called repeatedly for a specified number of iterations.
        '''




#        random_cluster = random.choice(np.arange(0,K,1))
        random_cluster= np.argmax([len(cluster) for cluster in clustered_data])
        while len(clustered_data[random_cluster]) == 1:
            random_cluster = random.choice(np.arange(0,K,1))
        if len(clustered_data[random_cluster]) != 1:
            prev_medoid = medoids[random_cluster]
            clustered_data[random_cluster].remove(prev_medoid)

            new_medoid = clustered_data[random_cluster][0]
#            new_medoid = random.choice(clustered_data[random_cluster])
            new_medoids = deepcopy(medoids)
            new_medoids[random_cluster] = new_medoid
            clustered_data[random_cluster].append(prev_medoid)



        # Reset clustered data for new clustsering.
        new_clustered_data = [[new_med] for new_med in new_medoids]
        for index,i in enumerate(self.D):
            if index not in new_medoids:
                closest = np.argmin([i[j] for j in new_medoids])
                new_clustered_data[closest].append(index)

        _,new_linear_params, new_clustering_cost = self.calc_cluster_models(self.x, self.y, new_clustered_data)

        recluster = True
#        print(clustering_cost, new_clustering_cost)
        if new_clustering_cost < clustering_cost:
#            print(f'New clustering cost: {new_clustering_cost} < {clustering_cost} = Old clustering cost')
            clustering_cost = new_clustering_cost
            linear_params = new_linear_params
            clustered_data = new_clustered_data
            medoids = new_medoids
        elif new_clustering_cost == clustering_cost:
            recluster = False
#        fig = self.plotMedoids(clustered_data, medoids, linear_params, clustering_cost)
#        fig.savefig(f'Figures/Clustering/{K}/AdaptedClustering_{iter}.pdf')
        return clustering_cost, linear_params, clustered_data, medoids, recluster

    def verify_clustering(self, clustered_data):

        changes = []
        clustered_data = self.order_clusters(clustered_data)

        ''' Check if there are large gaps within the clusters data.'''
        check_cluster_gaps = True
        if check_cluster_gaps == True:
            if len(clustered_data) != 1:
#            pre_clustered_data = deepcopy(clustered_data)
#            clustered_data = self.check_cluster_gaps(clustered_data)
#            # Check if satisfying this constraint has changed the clustering.
#            if pre_clustered_data != clustered_data:
#                changes.append(True)
#            else:
#                changes.append(False)
                bad_cluster = True
                while bad_cluster != None:
                    bad_cluster, dividing_point, clustered_data = self.remove_gaps(clustered_data)
                    if bad_cluster != None:
                        clustered_data = self.split_cluster(bad_cluster, dividing_point, clustered_data)
#                    sparse_clusters, _ = self.check_cluster_sparsity(clustered_data)
#                    if sparse_clsuter == bad_cluster:
#                        continue

        _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
        if self.plotting:
            fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
            fig.savefig(self.plotting_dir+f'/after_separating_gaps_{time.time()}.pdf', bbox_inches='tight')


        ''' Check if there are overlaps between neighbouring clusters or if clusters
            are contained within other clusters. '''
        if len(clustered_data) != 1:
#            # Check if neighbouring clusters overlap.
#            pre_clustered_data = deepcopy(clustered_data)
#            clustered_data = self.check_cluster_overlap(clustered_data, linear_params, len(clustered_data))
#            # Check if satisfying this constraint has changed the clustering.
#            if pre_clustered_data != clustered_data:
#                changes.append(True)
#            else:
#                changes.append(False)
#        _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
            pre_clustered_data = deepcopy(clustered_data)
            clustered_data = self.check_cluster_overlap(clustered_data)
            if pre_clustered_data != clustered_data:
                changes.append(True)
            else:
                changes.append(False)
        if self.plotting == True:
            fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
            fig.savefig(self.plotting_dir+f'/after_checking_overlaps_{time.time()}.pdf', bbox_inches='tight')



        ''' Check the sparsity and coverage of the clusters. '''
        if len(clustered_data) != 1:
        # Check size of clusters. If 10x smaller than neighbouring clusters, merge them.
            pre_clustered_data = deepcopy(clustered_data)
            bad_clusters = True
            while bad_clusters != None:
                bad_clusters, clustered_data = self.check_cluster_sparsity(clustered_data)
                if bad_clusters != None:
                    clustered_data = self.merge_clusters(clustered_data, bad_clusters)
                    _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
                    if self.plotting == True:
                        fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
                        fig.savefig(self.plotting_dir+f'/sparsity_{time.time()}.pdf', bbox_inches='tight')
            # Check if satisfying this constraint has changed the clustering.
            if pre_clustered_data != clustered_data:
                changes.append(True)
            else:
                changes.append(False)
        _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
        if self.plotting == True:
            fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
            fig.savefig(self.plotting_dir+f'/after_checking_sparsity_{time.time()}.pdf', bbox_inches='tight')

        clustering_cost = self.calculate_clustering_cost(clustered_data)

        _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
        medoids = self.find_medoids(clustered_data)

        return clustered_data, medoids, linear_params, clustering_cost, changes

    '''--------------------------GAPS---------------------------'''

    def check_cluster_gaps(self, clustered_data):
        clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        new_clustering = []
        for cluster_num, cluster in enumerate(clustered_datapoints):
            cluster_xs = cluster[0]
            if len(cluster_xs) > 1:
                cluster_gaps = self.find_gaps(cluster_xs)
                if len(cluster_gaps) > 0:
                    separated_clusters = [[] for i in range(len(cluster_gaps)+1)]
                    remaining_xs = [i for i in range(len(cluster_xs))]
                    for i, gap in enumerate(cluster_gaps):
                        separated_clusters[i] = [x for x in remaining_xs if cluster_xs[x] <= gap[0]]
                        for x in separated_clusters[i]:
                            remaining_xs.remove(x)
                        separated_clusters[i+1] = [x for x in remaining_xs if x not in separated_clusters[i]]

                    for separated_cluster in separated_clusters:
                        new_clustering.append([clustered_data[cluster_num][x] for x in separated_cluster])
                else:
                    new_clustering.append(clustered_data[cluster_num])
            else:
                new_clustering.append(clustered_data[cluster_num])

        new_clustering = [i for i in new_clustering if i != []]
        return new_clustering

    def remove_gaps(self, clustered_data):

        cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        for i in range(len(clustered_data)):
            if len(clustered_data[i]) > 1:
                sorted_datapoints = sorted(cluster_datapoints[i][0])
#                cluster_differences = [sorted_datapoints[j+1]-sorted_datapoints[j] for j in range(len(sorted_datapoints)-1)]
                cluster_differences = np.diff(sorted_datapoints)
#                try:
#                    print(max(cluster_differences))
#                except:
#                    print(cluster_differences, clustered_data[i])
                # value of the dividing point when sorted
                if max(cluster_differences) > self.gaps_threshold:
                    dividing_point = sorted_datapoints[np.argmax(cluster_differences)]
                    return i, dividing_point, clustered_data

        return None, None, clustered_data


    def split_cluster(self, i, dividing_point, clustered_data):
        clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        cluster_datapoints = clustered_datapoints[i]
        new_cluster1, new_cluster2 = [], []
        for indexed_point in clustered_data[i]:
            x = self.x[indexed_point]
            if x <= dividing_point:
                new_cluster1.append(indexed_point)
            else:
                new_cluster2.append(indexed_point)
        clustered_data[i] = new_cluster1
        clustered_data.append(new_cluster2)
        clustered_data = self.order_clusters(clustered_data)
        return clustered_data



    def order_clusters(self, clustered_data):

        '''
        Utility function which orders the clusters based on the minimum x value of each cluster.
        '''
        cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)

        minimums = [min(cluster_datapoints[i][0]) for i in range(len(cluster_datapoints))]
        index_sorted = sorted(range(len(minimums)), key=lambda k: minimums[k])
        ordered_clusters = [clustered_data[i] for i in index_sorted]

        return ordered_clusters


    '''--------------------------OVERLAP---------------------------'''
    def separate_overlapping_clusters(self, clustered_datapoints, clustered_data, c1, c2):
        x1, y1, idx1 = clustered_datapoints[c1][0], clustered_datapoints[c1][1], clustered_data[c1]
        x2, y2, idx2 = clustered_datapoints[c2][0], clustered_datapoints[c2][1], clustered_data[c2]


        # Convert to numpy arrays for convenience
        x1, y1, idx1 = np.array(x1), np.array(y1), np.array(idx1)
        x2, y2, idx2 = np.array(x2), np.array(y2), np.array(idx2)

        # Get x-ranges
        min1, max1 = x1.min(), x1.max()
        min2, max2 = x2.min(), x2.max()

        # Case 1: x1 is completely inside x2
        if min1 >= min2 and max1 <= max2:
            # Merge c1 into c2
            clustered_data[c2] = np.concatenate([idx2, idx1])

            # Remove c1
            del clustered_data[c1]
            return True, clustered_data

        # Case 2: x2 is completely inside x1
        if min2 >= min1 and max2 <= max1:
            # Merge c2 into c1
            clustered_data[c1] = np.concatenate([idx1, idx2])

            # Remove c2
            del clustered_data[c2]
            return True, clustered_data

        # Case 3: Partial overlap — split at midpoint
        if (min1 < min2 < max1):
            overlap_start = min2
            overlap_end = max1

            midpoint = (overlap_start + overlap_end) / 2

            # Split each cluster into left and right parts
            left_cluster = []
            right_cluster = []
            for i in clustered_data[c1]:
                if self.x[i] <= midpoint:
                    left_cluster.append(i)
                else:
                    right_cluster.append(i)
            for i in clustered_data[c2]:
                if self.x[i] <= midpoint:
                    left_cluster.append(i)
                else:
                    right_cluster.append(i)
            # Update clusters
            clustered_data[c1] = left_cluster
            clustered_data[c2] = right_cluster

            return True, clustered_data
        else:
            return False, clustered_data

    def check_cluster_overlap(self, clustered_data):
        cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        bad_clusters = True
        while bad_clusters:
            changed_clusters = False
            for c_num in range(len(clustered_data)-1):
                changed_clusters, clustered_data = self.separate_overlapping_clusters(cluster_datapoints, clustered_data, c_num, c_num+1)
                clustered_data = [list(c) for c in clustered_data]
                if changed_clusters:
                    cluster_datapoints = self.cluster_indices_to_datapoints(clustered_data)
                    _, linear_params, clustering_cost = self.calc_cluster_models(self.x, self.y, clustered_data)
                    if self.plotting:
                        fig = self.plotMedoids(clustered_data, None, linear_params, clustering_cost)
                        fig.savefig(self.plotting_dir+f'/overlap_{time.time()}.pdf', bbox_inches='tight')
                    break
            if not changed_clusters:
                bad_clusters = False
        return clustered_data




    '''--------------------------SPARSITY & COVERAGE---------------------------'''


    def check_cluster_sparsity(self, clustered_data):
        data_x_range = abs(max(self.x)-min(self.x))
        clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)
        largest_cluster_size = max([len(cluster) for cluster in clustered_data])


#        def css_score(cluster_xs):
#            return len(cluster_xs)/abs(max(cluster_xs) - min(cluster_xs))
#
#        all_css_score = [css_score(clustered_datapoints[c][0]) for c in range(len(clustered_data))]

        new_clustering = deepcopy(clustered_data)
        for i in range(len(clustered_data)):

            sparsity = len(clustered_data[i])/largest_cluster_size
            cluster_max = max(clustered_datapoints[i][0])
            cluster_min = min(clustered_datapoints[i][0])
            coverage = abs(cluster_max - cluster_min)
#            if coverage == 0:
#                return i, clustered_data
            coverage = coverage/data_x_range
#
#            local_density = len(clustered_data[i])/coverage
#            sparsity = local_density
#            css = css_score(clustered_datapoints[i][0])/np.mean(all_css_score)
#            sparsity = css
#            print(f'Threshold: {self.sparsity_threshold}, Sparsity: {sparsity}')

#            coverage = local_density

            is_sparsity = False
            is_coverage = False
            if coverage < self.coverage_threshold:
                is_coverage = True
            if sparsity < self.sparsity_threshold:
                is_sparsity = True

            if is_sparsity or is_coverage:
#                if is_sparsity:
#                    print(f'Cluster {i} is sparse with sparsity {sparsity:.4f}')
#                if is_coverage:
#                    print(f'Cluster {i} has low coverage with coverage {coverage:.4f}')
                return i, clustered_data
        return None, clustered_data


    def merge_clusters(self, clustered_data, i):
        new_clustering = deepcopy(clustered_data)
        if i == 0:
            new_clustering[i+1] = new_clustering[i] + new_clustering[i+1]
            new_clustering.pop(i)

        elif i == len(new_clustering)-1:
            new_clustering[i-1] = new_clustering[i-1] + new_clustering[i]
            new_clustering.pop(i)
        else:
            tempNewClustering1 = deepcopy(new_clustering)
            tempNewClustering2 = deepcopy(new_clustering)

            tempNewClustering1[i-1] = new_clustering[i-1] + new_clustering[i]
            tempNewClustering1.pop(i)
            cost1 = self.calculate_clustering_cost(tempNewClustering1)

            tempNewClustering2[i+1] = new_clustering[i+1] + new_clustering[i]
            tempNewClustering2.pop(i)
            cost2 = self.calculate_clustering_cost(tempNewClustering2)

            if cost1 <= cost2:
                new_clustering = deepcopy(tempNewClustering1)
            else:
                new_clustering = deepcopy(tempNewClustering2)
        return new_clustering

#
#
#
#
#
#            if (sparsity < self.sparsity_threshold) or (coverage < self.coverage_threshold):
#                print(self.sparsity_threshold, self.coverage_threshold)
#                print(suffix, sparsity, coverage)
##
##                print([len(cluster) for cluster in new_clustering])
##                fig, axes = plt.subplots(1,3, figsize=(20*0.39,4*0.39))
##                axes[0].scatter(self.x[new_clustering[i]], self.y[new_clustering[i]], color='blue', s=1)
#                # If first or last cluster
#                if i == 0 or new_clustering[i-1] == []:
#                    # if last cluster:
#                    if i ==len(new_clustering)-1:
#                        # make j the cluster second to last
#                        j = i-1
#                        # If the second to last is empty then pick the one before
#                        while new_clustering[j] == []:
#                            j -= 1
#                        # If all until j=-1 are empty (newclustering[-1] which is the last element) then skip
#                        if j != -1:
#                            temp_clustering = new_clustering[j] + new_clustering[i]
#                            if len(self.find_gaps(self.x[temp_clustering])) == 0:
#                                new_clustering[j] = new_clustering[j] + new_clustering[i]
#                                new_clustering[i] = []
##                        axes[0].scatter(self.x[new_clustering[j]], self.y[new_clustering[j]], color='red', s=1)
#                    else:
##                        axes[0].scatter(self.x[new_clustering[i+1]], self.y[new_clustering[i+1]], color='red', s=1)
#                        temp_clustering = new_clustering[i+1] + new_clustering[i]
#                        if len(self.find_gaps(self.x[temp_clustering])) == 0:
#                            new_clustering[i+1] = new_clustering[i+1] + new_clustering[i]
#                            new_clustering[i] = []
#                elif i == len(new_clustering)-1 or new_clustering[i+1] == []:
#
##                    axes[0].scatter(self.x[new_clustering[i-1]], self.y[new_clustering[i-1]], color='green', s=1)
##                    axes[1].scatter(self.x[new_clustering[i-1]], self.y[new_clustering[i-1]], color='green', s=1)
#
#                    temp_clustering = new_clustering[i-1] + new_clustering[i]
#                    if len(self.find_gaps(self.x[temp_clustering])) == 0:
#                        new_clustering[i-1] = new_clustering[i-1] + new_clustering[i]
#                        new_clustering[i] = []
#                else:
#                    tempNewClustering1 = deepcopy(new_clustering)
#
##                    axes[0].scatter(self.x[new_clustering[i-1]], self.y[new_clustering[i-1]], color='red', s=1)
##                    axes[2].scatter(self.x[new_clustering[i-1]], self.y[new_clustering[i-1]], color='red', s=1)
#
#                    tempNewClustering1[i-1] = new_clustering[i-1] + new_clustering[i]
#                    clustering_1_gaps = self.find_gaps(self.x[tempNewClustering1[i-1]])
#                    tempNewClustering1[i] = []
#                    cost1 = self.calculate_clustering_cost(tempNewClustering1)
#
##                    axes[1].scatter(self.x[tempNewClustering1[i-1]], self.y[tempNewClustering1[i-1]], color='red', s=1)
##                    axes[1].set_title(f'Clustering Cost = {cost1:.4f}')
#
#                    tempNewClustering2 = deepcopy(new_clustering)
#
##                    axes[0].scatter(self.x[new_clustering[i+1]], self.y[new_clustering[i+1]], color='green', s=1)
##                    axes[1].scatter(self.x[new_clustering[i+1]], self.y[new_clustering[i+1]], color='green', s=1)
#
#                    tempNewClustering2[i+1] = new_clustering[i+1] + new_clustering[i]
#                    clustering_2_gaps = self.find_gaps(self.x[tempNewClustering2[i+1]])
#                    tempNewClustering2[i] = []
#                    cost2 = self.calculate_clustering_cost(tempNewClustering2)
#
##                    axes[2].scatter(self.x[tempNewClustering2[i+1]], self.y[tempNewClustering2[i+1]], color='green', s=1)
##                    axes[2].set_title(f'Clustering Cost = {cost2:.4f}')
#
#                    if cost1 <= cost2  == 0 and len(clustering_1_gaps) == 0:
#                        new_clustering = deepcopy(tempNewClustering1)
#                    elif len(clustering_2_gaps) == 0:
##                    else:
#                        new_clustering = deepcopy(tempNewClustering2)
#
#
##                    for ax in [0,1,2]:
##                        axes[ax].set_xlim(1.05*(min(self.x[new_clustering[i-1]])), 1.05*(max(self.x[new_clustering[i+1]])))
##
##                for ax in [0,1,2]:
###                    axes[ax].set_xlim(1.05*(min(self.x)), 1.05*(max(self.x)))
##                    axes[ax].set_ylim(1.05*(min(self.y)), 1.05*(max(self.y)))
##                    axes[ax].set_xlabel(r'$x$', fontsize=11)
##                    axes[ax].set_xticklabels(axes[ax].get_xticklabels(), fontsize=11)
##                axes[0].set_ylabel(r'$\hat{y}$', fontsize=11)
##                for ax in [1,2]:
##                    axes[ax].set_ylabel('')
##                    axes[ax].set_yticklabels('')
##
##                fig.savefig(f'Figures/Clustering/OptimisedClusters/{self.feature_name}/{suffix}_{i}_{time.time()}.pdf', bbox_inches='tight')
##        print([len(cluster) for cluster in new_clustering])
#        clustered_data = [cluster for cluster in new_clustering if cluster != []]
#
#        return clustered_data

    def cluster_indices_to_datapoints(self, clustered_indices):

        '''
        Utility function which converts the clusters containing indices of the x values to the raw x datapoint values.
        '''
        # If it just a single cluster whose datapoints are being fetched, then dont create nested lists.
        if isinstance(clustered_indices[0], (int, np.integer)):
            xs, ys = [], []
            for i in clustered_indices:
                xs.append(self.x[i])
                ys.append(self.y[i])
            return xs, ys
        # If it is a list of clusters, then create nested lists.
        else:
            datapoints = [[[],[]] for i in range(len(clustered_indices))]
            for cluster in range(len(clustered_indices)):
                for i, point_index in enumerate(clustered_indices[cluster]):
#                        print(cluster, point_index)
                    datapoints[cluster][0].append(float(self.x[point_index]))
                    datapoints[cluster][1].append(float(self.y[point_index]))
            return datapoints

    def calculate_clustering_cost(self, clustered_data):

        '''
        Utility function which calculates the cost of a clustering based on the combined error of the linear regression models
        for each cluster. The cost is the sum of the absolute errors of each cluster.
        '''
        clustered_data = [cluster for cluster in clustered_data if cluster != []]
        clustered_data = self.cluster_indices_to_datapoints(clustered_data)
        total_error = 0
        ys = []
        preds = []
        for cluster_num, cluster in enumerate(clustered_data):

            cluster_x = cluster[0]
            cluster_y = cluster[1]
            w,b = self.LR(cluster_x, cluster_y)
            cluster_pred = [w*x+b for x in cluster_x]
            cluster_error = mean_squared_error(cluster_y, cluster_pred, squared=False)
            [ys.append(y) for y in cluster_y]
            [preds.append(pred) for pred in cluster_pred]
        total_error = mean_squared_error(ys, preds,  squared=False)
        return total_error


    def calc_cluster_models(self, X, Y, clustered_data, distFunction=None):

        '''
        Utility function which fits a linear regression model to each cluster and returns the linear parameters.
        '''

        clustered_datapoints = self.cluster_indices_to_datapoints(clustered_data)

        linear_params = []
        clusterPreds = []
        avg_intersection_cost = 0
        for i in range(len(clustered_datapoints)):
#            For cyclic features uncomment the following:
#            preds, w, b = self.CR.cyclicRegression(clustered_datapoints[i][0], clustered_datapoints[i][1])
            if len(clustered_datapoints[i][0]) > 0 and len(clustered_datapoints[i][1]) > 0:
                w,b = self.LR(clustered_datapoints[i][0], clustered_datapoints[i][1])
                linear_params.append([float(w),b])
        clustered_data = [cluster for cluster in clustered_data if cluster != []]
        return clustered_data, linear_params, self.calculate_clustering_cost(clustered_data)

    def plotMedoids(self, clustered_data, medoids, linear_params, clustering_cost):
        clustered_data = self.cluster_indices_to_datapoints(clustered_data)
#    CR = CyclicRegression()
        inch_to_cm = 0.39
        colours = self.colours
        fig, axes = plt.subplots(1,1,figsize=(15*inch_to_cm,8*inch_to_cm))
        axes = fig.axes
        axes[0].set_xlabel(f'{self.feature_name}  $x$', fontsize=11)
        axes[0].set_ylabel(f'{self.target_name}  $\hat{{y}}$', fontsize=11)

#        colours = []
        for i in range(len(clustered_data)):
            w,b = linear_params[i]
            colour = np.random.rand(1,3)
#            colours.append(colour)
#            colour = colours[i]
            axes[0].scatter(clustered_data[i][0], clustered_data[i][1], s=1, marker='o', c=colour, label='_nolegend_')
            cluster_range = np.linspace(min(clustered_data[i][0]), max(clustered_data[i][0]), 100)
#            axes[0].vlines([min(clustered_data[i][0]), max(clustered_data[i][0])], -10, 40, color=colour, label='_nolegend_')
            axes[0].plot(cluster_range, w*cluster_range+b, linewidth=0.5, c=colour)
        if clustering_cost:
            axes[0].set_title(f'K-Medoids clustering with {len(clustered_data)} clusters \n  Clustering Cost: {clustering_cost:.2f}', fontsize=11)

#        try:
#            axes[0].legend([str(i) for i in range(len(clustered_data))], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(clustered_data)/2, fontsize=11)
#        except:
#            pass

#        plotly_Fig = CR.plotCircularData(clustered_data[i][0], clustered_data[i][1], preds[i], plotly_Fig, colour)
        return fig

    def plot_final_clustering(self, clustered_data, linear_params):
        cost = self.calculate_clustering_cost(clustered_data)
        fig = go.Figure()
        for cluster in range(len(clustered_data)):
            colour = np.random.rand(3)
            colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))
            xs = self.x[clustered_data[cluster]]
            ys = self.y[clustered_data[cluster]]
            w, b = linear_params[cluster]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=4, color=colour)))
            cluster_range = np.linspace(min(xs), max(xs), 100)
            fig.add_trace(go.Scatter(x=cluster_range, y=w*cluster_range+b, mode='lines', line=dict(color=colour, width=5)))
        fig.update_layout(title=f'K-Medoids clustering of LLR models into {len(clustered_data)} clusters \n  Clustering Cost: {cost:.2f}')
        return fig

