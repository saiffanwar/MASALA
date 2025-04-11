import midas_model_data as midas
import torch.optim as optim
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
import pickle as pck
import random
from tqdm import tqdm
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from itertools import product


from MidasDataProcessing import MidasDataProcessing
from midasDistances import combinedFeatureDistances, calcAllDistances, calcSingleDistance, pointwiseDistance
from chilli import CHILLI
from llc_explainer import LLCExplanation
from llc_ensemble_generator import LLCGenerator

is_cuda = torch.cuda.is_available()
is_cuda = False
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

parser = argparse.ArgumentParser(description='Run the MIDAS model and generate explanations')
parser.add_argument('-d', '--dataset', type=str, default='MIDAS', help='Which dataset to work on')
parser.add_argument('-l', '--load_model', type=bool, default=True, help='Load the model from file')
parser.add_argument('-m', '--mode', type=str, default='explain', help='Whether to generate ensembles or explanations.')
parser.add_argument('-e', '--exp_mode', type=str, default='random', help='Which instances to generate explanations for.')
parser.add_argument('--sparsity', type=float, default=0.05, help='The sparsity threshold to use for the LLC explainer.')
parser.add_argument('--coverage', type=float, default=0.05, help='The coverage threshold to use for the LLC explainer.')
parser.add_argument('--starting_k', type=int, default=10, help='The number of neighbours to use for the LLC explainer.')
parser.add_argument('--neighbourhood', type=float, default=0.05, help='The neighbourhood threshold to use for the LLC explainer.')
parser.add_argument('-p', '--primary_instance', type=int, default=None, help='The instance to generate explanations for')
parser.add_argument('-n', '--num_instances', type=int, default=20, help='The number of instances to generate explanations for')
parser.add_argument('--c_id', type=int, default=1, help='Clustering ID')
parser.add_argument('--e_id', type=int, default=1, help='Experiment ID')


args = parser.parse_args()

class RUL():

    def __init__(self, load_model=True):
        self.load_model = load_model

    def data_preprocessing(self,):

        self.data = pd.read_csv('Data/PHM08/PHM08.csv')

        self.features = self.data.drop(['RUL','cycle', 'id'], axis=1).columns.tolist()

        train_df = self.data[self.data['id'] <= 150]
        test_df = self.data[self.data['id'] > 150]

        scaler = StandardScaler()
#        x_train = scaler.fit_transform(x_train)
#        x_test = scaler.fit_transform(x_test)
        self.x_train = scaler.fit_transform(train_df.drop(['RUL','cycle', 'id'], axis=1).values)
        self.x_test = scaler.fit_transform(test_df.drop(['RUL', 'cycle','id'], axis=1).values)
        self.y_train = train_df['RUL'].values
        self.y_test = test_df['RUL'].values

        # ----------------------------------------------------------

        return self.x_train, self.x_test, self.y_train, self.y_test, self.features

#    def data_visualisation(self,):
#        self.data = pd.read_csv('Data/PHM08/PHM08.csv')
#        for col in self.data.columns:
##        for i in data['id'].unique():
#            fig, axes = plt.subplots(1, 1, figsize=(10, 4))
#            axes.scatter(col, 'RUL', data=self.data, alpha=0.5,s=1)
#            fig.savefig('Figures/PHM08/' + col + '.png')
#data_visualisation()

    def train(self):
        model = GradientBoostingRegressor(max_depth=10, n_estimators=500, random_state=42, verbose=True)
        model.fit(self.x_train, self.y_train)
        with open('saved/models/PHM08_model.pck', 'wb') as file:
            pck.dump(model, file)
        return model

    def evaluate(self, model, X_train, X_test, y_train, y_test):
        y_hat_train = model.predict(X_train)
        print('training RMSE: ',mean_squared_error(y_train, y_hat_train,squared=False))
        oad_model = False
        y_hat_test = model.predict(X_test)
        print('test RMSE: ',mean_squared_error(y_test, y_hat_test, squared=False))
        fig = midas.plotPredictions(self.features, X_test, y_test, y_hat_test)
        fig.savefig(f'Figures/PHM08/PHM08_Predictions.pdf', bbox_inches='tight')

        return y_hat_train, y_hat_test


def run_clustering(model, x_test,y_pred, dataset, features, discrete_features, sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold):

    print('Starting thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)
    GLE = LLCGenerator(model=model, x_test=x_test, y_pred=y_pred, features=features, discrete_features=discrete_features, dataset=dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=False, experiment_id=args.c_id)
    GLE.multi_layer_clustering()
    print('finishing thread with parameters: ',sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)

def main():
    dataset = args.dataset

    if dataset == 'MIDAS':
        midas_runner = MIDAS(load_model=args.load_model)
        x_train, x_test, y_train, y_test, features = midas_runner.data.X_train, midas_runner.data.X_test, midas_runner.data.y_train, midas_runner.data.y_test, midas_runner.data.trainingFeatures

        target_feature = 'heathrow air_temperature'
        discrete_features = ['heathrow cld_ttl_amt_id']
        model = midas_runner.train_midas_rnn()
        y_train_pred, y_test_pred = midas_runner.make_midas_predictions()

        sampling=False
        sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = 0.05, 0.05, 10, 0.05
#        x_test = x_train
#        y_test = y_train
#        y_test_pred = y_train_pred
        y_pred = y_test_pred

    elif dataset == 'PHM08':
        phm08_runner = RUL()
        x_train, x_test, y_train, y_test, features = phm08_runner.data_preprocessing()
        target_feature = 'RUL'
        discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
        with open(f'saved/models/PHM08_model.pck', 'rb') as file:
            model = pck.load(file)
#        model = phm08_runner.train()
        y_train_pred, y_test_pred = phm08_runner.evaluate(model, x_train, x_test, y_train, y_test)

        sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = 0.5, 0.05, 5, 0.5
        sampling = True

    categorical_features = [features.index(feature) for feature in discrete_features]
    if sampling:
        R = np.random.RandomState(42)
        random_samples = R.randint(2, len(x_test), 5000)

        x_train = x_train[random_samples]
        y_train = y_train[random_samples]
        x_test = x_test[random_samples]
        y_pred = y_test_pred[random_samples]
        y_test = y_test[random_samples]
        print(f'Training samples: {len(x_train)}')
        print(f'Test samples: {len(x_test)}')

    for f in range(len(features)):
        sorted_f = np.sort(x_train[:,f])
        distances= [sorted_f[i+1]-sorted_f[i] for i in range(len(sorted_f)-1)]


#    sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = args.sparsity, args.coverage, args.starting_k, args.neighbourhood

    if args.mode == 'ensembles':

        starting_k = 10
        sparsity_threshold, coverage_threshold, neighbourhood_threshold = (None, 0.05, 0.05)
        run_clustering(model, x_test, y_pred, dataset, features, discrete_features, sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold)

    elif args.mode=='explain':
        sparsity_threshold, coverage_threshold, neighbourhood_threshold = (None, 0.05, 0.05)
        random.seed(args.e_id*10)

        # ---- Same Instances ----
        if args.exp_mode == 'same':
            if args.primary_instance == None:
                instance = random.randint(0, len(x_test))
            else:
                instance = args.primary_instance
            instances = [instance for i in range(10)]
#            instances=[instance]

        #  ---- Random Instances ----
        elif args.exp_mode == 'random':
            instances = [random.randint(0, len(x_test)) for i in range(args.num_instances)]
#            instances  = [1118, 3552, 261, 3560, 3377, 405, 2417, 389, 2409, 2199, 1567, 3758, 2553, 2779, 3676, 396, 3394, 3373, 3167, 2880]
#                    LLCGen.plot_all_clustering(instance = instances[0], instances_to_show=[])

        print('Generating explanations for instances: ', instances)

        # ---- Explanation Generation ----

        if args.kernel_width == None:
            if args.dataset == 'MIDAS':
                kernel_width = 0.1
            elif args.dataset == 'PHM08':
                kernel_width = 0.1
        else:
            kernel_width = args.kernel_width

#        for sparsity_threshold, coverage_threshold, neighbourhood_threshold in combinations:

        lime_results = {'predictions':[], 'explanations':[]}
        chilli_results = {'predictions':[], 'explanations':[]}
        llc_results = {'predictions':[], 'explanations':[]}
        model_predictions = [y_test_pred[instance] for instance in instances]
        for instance in tqdm(instances):


            LLCGen = LLCGenerator(model, x_test, y_pred, features, discrete_features, dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold,  preload_explainer=True, experiment_id=args.c_id)
            GLE = LLCExplanation(model=model, x_test=x_test, y_pred=y_pred, features=features, discrete_features=discrete_features, dataset=dataset, sparsity_threshold=sparsity_threshold, coverage_threshold=coverage_threshold, starting_k=starting_k, neighbourhood_threshold=neighbourhood_threshold, preload_explainer=True, experiment_id=args.c_id)
#            LLCGen.plot_all_clustering(instance=instance)

            print('plotting')
            LLCGen.matplot_all_clustering()

            print(f'################# Instance  = {instance} ###################')
            # ------ BASE MODEL ------
            print(f'Ground Truth: {y_test[instance]}')
            print(f'Model Prediction: {y_test_pred[instance]}')


            # ---- LIME EXPLANATION -------
            if args.lime_exp == 'y':

                print('\n ----- LIME EXPLANATION -----')
                _,_, lime_prediction, lime_plotting_data, exp_model = chilli_explain(dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred,features, instance=instance, newMethod=False, kernel_width=kernel_width, categorical_features=categorical_features)

                instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, lime_exp = lime_plotting_data
                print(f'LIME Prediction: {lime_prediction}')
                print(f'LIME Error: {abs(model_instance_prediction-lime_prediction)}')

                lime_results['predictions'].append(lime_prediction)
                lime_results['explanations'].append(lime_exp.as_list())

            # ---- CHILLI EXPLANATION -------
            if args.chilli_exp == 'y':
                print('\n ----- CHILLI EXPLANATION -----')
                _,_, chilli_prediction, chilli_plotting_data, exp_model = chilli_explain(dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, instance=instance, newMethod=True, kernel_width=kernel_width, categorical_features=categorical_features)

                instance_data, instance_index, perturbations, model_perturbation_predictions, ground_truth, model_instance_prediction, exp_instance_prediction, exp_perturbation_predictions, chilli_exp = chilli_plotting_data
                print(f'CHILLI Prediction: {chilli_prediction}')
                print(f'CHILLI Error: {abs(model_instance_prediction-chilli_prediction)}')


                chilli_results['predictions'].append(chilli_prediction)
                chilli_results['explanations'].append(chilli_exp.as_list())

            # ---- LLC EXPLANATION -------
            print('\n ----- LLC EXPLANATION -----')
            llc_prediction, llc_plotting_data, matched_instances = GLE.generate_explanation(x_test[instance], instance, y_test_pred[instance], y_test[instance])

            data_instance, instance_index, local_x, local_x_weights, local_y_pred, ground_truth, instance_prediction, exp_instance_prediction, exp_local_y_pred, instance_explanation_model, instance_cluster_models = llc_plotting_data
            print(f'LLC Prediction: {llc_prediction}')
            print(f'LLC Error: {abs(instance_prediction-llc_prediction)}')


            llc_results['predictions'].append(llc_prediction)
            llc_results['explanations'].append(instance_explanation_model.coef_)

            GLE.interactive_exp_plot(data_instance, instance_index, instance_prediction, y_test_pred, exp_instance_prediction, instance_explanation_model.coef_, local_x, local_x_weights, local_y_pred, exp_local_y_pred, target_feature)

        average_error = [np.mean([abs(a-b) for a,b in zip(model_predictions, results['predictions'])]) for results in [lime_results, chilli_results, llc_results]]
        print(f'Average Error: {average_error}')
        with open(f'saved/results/{dataset}/{dataset}#_{args.e_id}_{args.primary_instance}_{len(instances)}_{args.exp_mode}_kw={kernel_width}.pck', 'wb') as f:
            pck.dump([lime_results, chilli_results, llc_results, instances, model_predictions], f)
#            with open(f'saved/results/{dataset}/{dataset}_{sparsity_threshold}_{coverage_threshold}_{neighbourhood_threshold}_{args.exp_mode}.pck', 'wb') as f:
#                pck.dump([llc_results, instances, model_predictions ], f)

#            except:
#                print('Error in instance: ',instance)
#                pass
if __name__ == '__main__':
    main()
