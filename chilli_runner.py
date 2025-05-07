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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from itertools import product


from MidasDataProcessing import MidasDataProcessing
from midasDistances import combinedFeatureDistances, calcAllDistances, calcSingleDistance, pointwiseDistance
import midas_model_data as midas

from phm08_model_data import RUL

from housing import CaliforniaHousingModel
from data_manipulator import attribution_percentage_score, shap_attribution_percentage_score, fidelity_via_feature_removal

import webtrisDataProcessing
from webtrismodels import webTRIS

from chilli import CHILLI, exp_sorter
from masala import masala

import warnings
warnings.filterwarnings("ignore")
#from llc_explainer import LLCExplanation
#from llc_ensemble_generator import LLCGenerator

is_cuda = torch.cuda.is_available()
is_cuda = False
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

parser = argparse.ArgumentParser(description='Run the MIDAS model and generate explanations')
parser.add_argument('-d', '--dataset', type=str, default='housing', help='Which dataset to work on')
parser.add_argument('--model_type', type=str, default='GBR', help='Which dataset to work on')
parser.add_argument('-l', '--load_model', type=str, default='y', help='Load the model from file')
parser.add_argument('-m', '--mode', type=str, default='explain', help='Whether to generate ensembles or explanations.')
parser.add_argument('-e', '--exp_mode', type=str, default='random', help='Which instances to generate explanations for.')
parser.add_argument('--sparsity', type=float, default=0.05, help='The sparsity threshold to use for the LLC explainer.')
parser.add_argument('--coverage', type=float, default=0.05, help='The coverage threshold to use for the LLC explainer.')
parser.add_argument('--starting_k', type=int, default=10, help='The number of neighbours to use for the LLC explainer.')
parser.add_argument('--neighbourhood', type=float, default=0.05, help='The neighbourhood threshold to use for the LLC explainer.')
parser.add_argument('-p', '--primary_instance', type=int, default=None, help='The instance to generate explanations for')
parser.add_argument('-n', '--num_instances', type=int, default=30, help='The number of instances to generate explanations for')
parser.add_argument('--c_id', type=int, default=1, help='Clustering ID')
parser.add_argument('--e_id', type=int, default=1, help='Experiment ID')
parser.add_argument('-kw', '--kernel_width', type=float, default=None, help='The kernel width to use for the explanations')
parser.add_argument('--plots', action=argparse.BooleanOptionalAction)
parser.add_argument('--results', action=argparse.BooleanOptionalAction)

parser.add_argument('--lime', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--chilli', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--masala', action=argparse.BooleanOptionalAction, default=True)



args = parser.parse_args()
if args.load_model == 'y':
    args.load_model = True
else:
    args.load_model = False

def main():
    dataset = args.dataset

    if dataset == 'MIDAS':
        midas_runner = midas.MIDAS(load_model=args.load_model, model_type=args.model_type)
        x_train, x_test, y_train, y_test, features = midas_runner.data.X_train, midas_runner.data.X_test, midas_runner.data.y_train, midas_runner.data.y_test, midas_runner.data.trainingFeatures

        target_feature = 'heathrow air_temperature'
        discrete_features = ['heathrow cld_ttl_amt_id']
        model = midas_runner.train_midas_model()
        y_train_pred, y_test_pred = midas_runner.make_midas_predictions(model)

        sampling=False
        sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = 0.05, 0.05, 10, 0.05
#        x_test = x_train
#        y_test = y_train
#        y_test_pred = y_train_pred
        y_pred = y_test_pred

        manipulated_features = ['heathrow wind_speed']
        manipulated_feature_idxs = [features.index(f) for f in manipulated_features]

    elif dataset == 'PHM08':
        phm08_runner = RUL(model_type=args.model_type, load_model=args.load_model)
        x_train, x_test, y_train, y_test, features = phm08_runner.datapreprocessing()
        target_feature = 'RUL'
        discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
        if args.load_model:
            with open(f'saved/models/PHM08_model.pck', 'rb') as file:
                model = pck.load(file)
        else:
            model = phm08_runner.train()
        y_train_pred, y_test_pred = phm08_runner.evaluate(model, x_train, x_test, y_train, y_test)
        model = model.predict

        sampling = True

    elif dataset == 'housing':
        model_runner = CaliforniaHousingModel(model_type=args.model_type)
#        model_runner.run_pipeline()
        model_runner.load_data()
        model_runner.preprocess()
        if args.load_model:
            model_runner.model = pck.load(open(f'saved/models/housing_{args.model_type}.pck', 'rb'))
        else:
            model_runner.train()
        model_runner.evaluate()

        x_train, x_test, y_train, y_test, features = model_runner.X_train, model_runner.X_test, model_runner.y_train, model_runner.y_test, model_runner.df.columns.tolist()
        features = features[:-1]
        target_feature = 'MedHouseVal'
        discrete_features = []
        model = model_runner.model
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
#        sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = 0.5, 0.05, 5, 0.5
        sampling = False
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        model = model_runner.model.predict

        manipulated_features = ['MedInc', 'HouseAge']
        manipulated_feature_idxs = [features.index(f) for f in manipulated_features]
        sampling=True

    elif dataset == 'webTRIS':
        webtris_data = webtrisDataProcessing.DataProcessing(siteName='M6/7570A')
        webtris_runner = webTRIS(webtris_data, model_type=args.model_type, load_model=args.load_model)
        if args.load_model:
            with open(f'saved/models/webTRIS_{args.model_type}.pck', 'rb') as file:
                model = pck.load(file)
        else:
            model = webtris_runner.train_model()

        features = webtris_runner.features


        x_train = np.array(webtris_runner.x_train)
        x_test = np.array(webtris_runner.x_test)
        y_train = np.array(webtris_runner.y_train)
        y_test = np.array(webtris_runner.y_test)
        y_test_pred = model(x_test)
        y_train_pred = model(x_train)
        discrete_features = ['Day', 'Time Interval']
        sampling = True



    categorical_features = [features.index(feature) for feature in discrete_features]


    if sampling:
        R = np.random.RandomState(42)
        random_samples = R.randint(2, len(x_test), min(len(x_test), 5000))

        x_train = x_train[random_samples]
        y_train = y_train[random_samples]
        x_test = x_test[random_samples]
        y_test_pred = y_test_pred[random_samples]
        y_test = y_test[random_samples]
        print(f'Training samples: {len(x_train)}')
        print(f'Test samples: {len(x_test)}')

    for f in range(len(features)):
        sorted_f = np.sort(x_train[:,f])
        distances= [sorted_f[i+1]-sorted_f[i] for i in range(len(sorted_f)-1)]

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

    print('Generating explanations for instances: ', instances)

    # ---- Explanation Generation ----

    if args.kernel_width == None:
        kernel_widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#        kernel_widths = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
        if args.dataset == 'MIDAS':
            kernel_width = None
        elif args.dataset == 'PHM08':
            kernel_width = 0.1
    else:
        kernel_width = args.kernel_width
        kernel_widths = [kernel_width]


    # MASALA does not need to be ran for all kernel_widths. Set to True once ran once.
    masala_ran = False

#    for kernel_width in kernel_widths:
    if args.chilli:
        CHILLIExplainer = CHILLI(dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, using_chilli=True)
        CHILLIExplainer.build_explainer(mode='regression', kernel_width=kernel_width, categorical_features=categorical_features)

    if args.lime:
        LIMEExplainer = CHILLI(dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, using_chilli=False)
        LIMEExplainer.build_explainer(mode='regression', kernel_width=kernel_width, categorical_features=categorical_features)

    if args.masala:
        MASALAExplainer = masala.MASALA(model, args.model_type, x_test, y_test, y_test_pred, dataset, features, discrete_features, sparsity_threshold=0.05, coverage_threshold=0.05, starting_k=5, neighbourhood_threshold=0.05)


    chilli_results = {'features': features,
                      'instance_indices': [],
                      'instance_data': [],
                      'exp_instance_prediction':[],
                      'explanations':[],
                      'perturbations': [],
                      'model_perturbation_predictions':[],
                      'exp_perturbation_predictions':[],
                      'model_instance_prediction': []}
    lime_results = {'features': features,
                    'instance_indices': [],
                    'instance_data': [],
                    'exp_instance_prediction':[],
                    'explanations':[],
                    'perturbations': [],
                    'model_perturbation_predictions':[],
                    'exp_perturbation_predictions':[],
                    'model_instance_prediction': []}
    masala_results = {'features': features,
                      'instance_indices': [],
                      'instance_data': [],
                      'exp_instance_prediction':[],
                      'explanations':[],
                      'model_instance_prediction': [],
                      'local_errors': []}

    model_instance_predictions = [y_test_pred[instance] for instance in instances]

    for instance in tqdm(instances):

#            kw_lime_results = []
#            kw_chilli_results = []
#            try:
        print(f'\n \n ################# Instance  = {instance} ###################')
        # ------ BASE MODEL ------
        print(f'Ground Truth: {y_test[instance]}')
        print(f'Model Prediction: {y_test_pred[instance]}')


        # ---- LIME EXPLANATION -------
        if args.lime:

            print('\n ----- LIME EXPLANATION -----')

            lime_exp, lime_perturbations, model_perturbation_predictions, exp_perturbation_predictions, model_instance_prediction, exp_instance_prediction = LIMEExplainer.make_explanation(model, instance=instance, num_samples=1000)
#                LIMEExplainer.interactive_perturbation_plot(instance, lime_exp, kernel_width, lime_perturbations, model_perturbation_predictions, exp_perturbation_predictions)

            if args.plots:
                LIMEExplainer.plot_perturbations(instance, lime_exp, kernel_width, lime_perturbations, model_perturbation_predictions, exp_perturbation_predictions)
#                    kw_lime_results.append([lime_exp, lime_perturbations, model_perturbation_predictions, exp_perturbation_predictions])

#            print(f'LIME Attr %: {lime_attr_per}')
            print(f'LIME Prediction: {exp_instance_prediction}')
            print(f'LIME Instance Error: {abs(model_instance_prediction-exp_instance_prediction)}')
            print(f'LIME Local Error: {mean_absolute_error(model_perturbation_predictions, exp_perturbation_predictions)}')

#            lime_results['Attr_%s'].append(lime_attr_per)
            lime_results['instance_indices'].append(instance)
            lime_results['instance_data'].append(x_test[instance])
            lime_results['exp_instance_prediction'].append(exp_instance_prediction)
            lime_results['model_instance_prediction'].append(model_instance_prediction)
            lime_results['explanations'].append(lime_exp)
            lime_results['perturbations'].append(lime_perturbations)
            lime_results['model_perturbation_predictions'].append(model_perturbation_predictions)
            lime_results['exp_perturbation_predictions'].append(exp_perturbation_predictions)

        # ---- CHILLI EXPLANATION -------
        if args.chilli:
            print('\n ----- CHILLI EXPLANATION -----')

            chilli_exp, chilli_perturbations, model_perturbation_predictions, exp_perturbation_predictions, model_instance_prediction, exp_instance_prediction = CHILLIExplainer.make_explanation(model, instance=instance, num_samples=1000)
            if args.plots:
                CHILLIExplainer.plot_perturbations(instance, chilli_exp, kernel_width, chilli_perturbations, model_perturbation_predictions, exp_perturbation_predictions)
#                    kw_chilli_results.append([chilli_exp, chilli_perturbations, model_perturbation_predictions, exp_perturbation_predictions])

            print(f'CHILLI Prediction: {exp_instance_prediction}')
            print(f'CHILLI Instance Error: {abs(model_instance_prediction-exp_instance_prediction)}')
            print(f'CHILLI Local Error: {mean_absolute_error(model_perturbation_predictions, exp_perturbation_predictions)}')

            chilli_results['instance_indices'].append(instance)
            chilli_results['instance_data'].append(x_test[instance])
            chilli_results['exp_instance_prediction'].append(exp_instance_prediction)
            chilli_results['model_instance_prediction'].append(model_instance_prediction)
            chilli_results['explanations'].append(chilli_exp)
            chilli_results['perturbations'].append(chilli_perturbations)
            chilli_results['model_perturbation_predictions'].append(model_perturbation_predictions)
            chilli_results['exp_perturbation_predictions'].append(exp_perturbation_predictions)

#                CHILLIExplainer.plot_perturbation_distribution(chilli_perturbations, lime_perturbations, instance)

        if args.masala:
            if not masala_ran:
                explanation, local_error = MASALAExplainer.explain_instance(instance=instance)

                masala_results['instance_indices'].append(instance)
                masala_results['instance_data'].append(x_test[instance])
                masala_results['exp_instance_prediction'].append(explanation.target_exp_y)
                masala_results['model_instance_prediction'].append(explanation.target_model_y)
                masala_results['explanations'].append(explanation)
                masala_results['local_errors'].append(local_error)

#            except:
#                instances.append(random.randint(0, len(x_test)))
#                print(f'Random instance: {instances[-1]}')
#                pass

#        with open('saved/results/temp_kw_results.pck', 'wb') as f:
#            pck.dump([kw_lime_results, kw_chilli_results], f)
#        if args.plots:
#            LIMEExplainer.compare_kw_perturbations(kernel_widths, kw_lime_results, instance)
#            CHILLIExplainer.compare_kw_perturbations(kernel_widths, kw_chilli_results, instance)
    print('MASALA Average Error: ', np.mean(masala_results['local_errors']))

    if args.masala:
        masala_ran = True
    if args.results:
        if args.lime:
            with open(f'saved/results/{dataset}/LIME_{dataset}_{args.model_type}#_{args.e_id}_{args.num_instances}_kw={kernel_width}.pck', 'wb') as f:
                pck.dump(lime_results, f)
        if args.chilli:
            with open(f'saved/results/{dataset}/CHILLI_{dataset}_{args.model_type}#_{args.e_id}_{args.num_instances}_kw={kernel_width}.pck', 'wb') as f:
                pck.dump(chilli_results, f)
        if args.masala:
            with open(f'saved/results/{dataset}/MASALA_{dataset}_{args.model_type}#_{args.e_id}_{args.num_instances}.pck', 'wb') as f:
                pck.dump(masala_results, f)


if __name__ == '__main__':
    main()
