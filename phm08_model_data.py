import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pickle as pck
from midas_model_data import plotPredictions
from data_manipulator import phm08_data_manipulator



class RUL():

    def __init__(self, model_type='GBR', load_model=True):
        self.load_model = load_model
        self.model_type = model_type

    def datapreprocessing(self,):

        self.data = pd.read_csv('PHM08/Data/PHM08.csv')
#        self.data = phm08_data_manipulator(self.data)

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

        if self.model_type == 'GBR':
            model = GradientBoostingRegressor(max_depth=10, n_estimators=500, random_state=42, verbose=True)
        elif self.model_type == 'SVR':
            model = SVR(kernel='rbf', verbose=True)
        elif self.model_type == 'RF':
            model = RandomForestRegressor(random_state=42, n_estimators=500, verbose=True)
        model.fit(self.x_train, self.y_train)
        with open(f'saved/models/PHM08_{self.model_type}.pck', 'wb') as file:
            pck.dump(model, file)
        return model

    def evaluate(self, model, X_train, X_test, y_train, y_test):
        y_hat_train = model.predict(X_train)
        print('training RMSE: ',mean_squared_error(y_train, y_hat_train,squared=False))
        oad_model = False
        y_hat_test = model.predict(X_test)
        print('test RMSE: ',mean_squared_error(y_test, y_hat_test, squared=False))
        fig = plotPredictions(self.features, X_test, y_test, y_hat_test)
        fig.savefig(f'PHM08/Figures/PHM08_{self.model_type}_Predictions.pdf', bbox_inches='tight')

        return y_hat_train, y_hat_test
