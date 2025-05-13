import random
import pickle as pck
import numpy as np
import json

from MidasDataProcessing import MidasDataProcessing
from midasDistances import combinedFeatureDistances, calcAllDistances, calcSingleDistance, pointwiseDistance
import midas_model_data as midas

from phm08_model_data import RUL

from housing import CaliforniaHousingModel
from data_manipulator import attribution_percentage_score, shap_attribution_percentage_score, fidelity_via_feature_removal

import webtrisDataProcessing
from webtrismodels import webTRIS

class BaseModelData:
    def __init__(self, dataset, load_saved_model, model_type):
        self.dataset = dataset
        with open('cleaned_feature_names.json', 'r') as f:
            feature_dict = json.load(f)
        self.cleaned_feature_names = feature_dict[dataset]

        self.load_data_and_predictions(dataset, load_saved_model, model_type)

    def load_data_and_predictions(self, dataset, load_saved_model, model_type):

        if dataset == 'MIDAS':
            midas_runner = midas.MIDAS(cleaned_feature_names=self.cleaned_feature_names, load_model=load_saved_model, model_type=model_type)
            x_train, x_test, y_train, y_test, features = midas_runner.data.X_train, midas_runner.data.X_test, midas_runner.data.y_train, midas_runner.data.y_test, midas_runner.data.trainingFeatures

            target_feature = 'Heathrow Air Temperature'
            discrete_features = [self.cleaned_feature_names['heathrow cld_ttl_amt_id']]
            model = midas_runner.train_midas_model()
            y_train_pred, y_test_pred = midas_runner.make_midas_predictions(model)

            sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold = 0.05, 0.05, 10, 0.05
#        x_test = x_train
#        y_test = y_train
#        y_test_pred = y_train_pred
            y_pred = y_test_pred
            sampling=False


        elif dataset == 'PHM08':
            phm08_runner = RUL(model_type=model_type, load_model=load_saved_model)
            x_train, x_test, y_train, y_test, features = phm08_runner.datapreprocessing()
            target_feature = 'RUL'
            discrete_features = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
            if load_saved_model:
                with open(f'saved/models/PHM08_model.pck', 'rb') as file:
                    model = pck.load(file)
            else:
                model = phm08_runner.train()
            y_train_pred, y_test_pred = phm08_runner.evaluate(model, x_train, x_test, y_train, y_test)
            model = model.predict

            sampling = False

        elif dataset == 'housing':
            model_runner = CaliforniaHousingModel(model_type=model_type, cleaned_feature_names=self.cleaned_feature_names)
#        model_runner.run_pipeline()
            model_runner.load_data()
            model_runner.preprocess()
            if load_saved_model:
                model_runner.model = pck.load(open(f'saved/models/housing_{model_type}.pck', 'rb'))
            else:
                model_runner.train()
#            model_runner.evaluate()

            x_train, x_test, y_train, y_test, features = model_runner.X_train, model_runner.X_test, model_runner.y_train, model_runner.y_test, model_runner.df.columns.tolist()
            features = features[:-1]
            target_feature = 'Median House Value'
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


        elif dataset == 'webTRIS':
            webtris_data = webtrisDataProcessing.DataProcessing(siteName='M6/7570A')
            webtris_runner = webTRIS(webtris_data, feature_names=self.cleaned_feature_names, model_type=model_type, load_model=load_saved_model)
            print(f'Loading data for {model_type} model')
            model = webtris_runner.train_model()

            features = webtris_runner.features
            target_feature = 'Traffic Volume'


            x_train = np.array(webtris_runner.x_train)
            x_test = np.array(webtris_runner.x_test)
            y_train = np.array(webtris_runner.y_train)
            y_test = np.array(webtris_runner.y_test)
            y_test_pred = model(x_test)
            y_train_pred = model(x_train)
            discrete_features = ['Day of Week']
            sampling = False

        if sampling:
            x_train, y_train, x_test, y_test, y_test_pred = self.sample_data(x_train, y_train, x_test, y_test, y_test_pred)

        self.model = model

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_test_pred = y_test_pred

        self.features = features
        self.discrete_features = discrete_features
        self.target_feature = target_feature


    def sample_data(self, x_train, y_train, x_test, y_test, y_test_pred):
        R = np.random.RandomState(42)
        random_samples = R.randint(2, len(x_test), min(len(x_test), 500))

        x_train = x_train[random_samples]
        y_train = y_train[random_samples]
        x_test = x_test[random_samples]
        y_test_pred = y_test_pred[random_samples]
        y_test = y_test[random_samples]
        print(f'Training samples: {len(x_train)}')
        print(f'Test samples: {len(x_test)}')

        return x_train, y_train, x_test, y_test, y_test_pred

