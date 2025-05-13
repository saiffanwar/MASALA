import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle as pck
from midas_model_data import plotPredictions as midasPlotPredictions
from midas_model_data import RNNModel, Optimization
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn

def plotPredictions(features, x_test, y_test, y_pred):
    # fig, axs = plt.subplots(2,3, figsize=(7,4))

#     fig = plt.figure(figsize=(5,6))
#     plt.tight_layout()
#     grid = plt.GridSpec(6, 2, hspace=0.6, wspace=0.1)
#     # axs1 = fig.add_subplot(grid[0:2, 0])
#     # hist1 = fig.add_subplot(grid[2,0])
#     # axs2 = fig.add_subplot(grid[0:2, 1], sharey=axs1)
#     # hist2 = fig.add_subplot(grid[2,1])
#     # axs3 = fig.add_subplot(grid[3:5, 0])
#     # hist3 = fig.add_subplot(grid[5,0])
#     # axs4 = fig.add_subplot(grid[3:5, 1], sharey=axs1)
#     # hist4 = fig.add_subplot(grid[5, 1])
#     # axs5 = fig.add_subplot(grid[6:8, 0])
#     # hist5 = fig.add_subplot(grid[8, 0])
#     # axs6 = fig.add_subplot(grid[6:8, 1], sharey=axs1)
#     # hist6 = fig.add_subplot(grid[8, 1])
#     axs1 = fig.add_subplot(grid[0:2, 0])
#     axs2 = fig.add_subplot(grid[0:2, 1], sharey=axs1)
#     axs3 = fig.add_subplot(grid[2:4, 0])
#     axs4 = fig.add_subplot(grid[2:4, 1], sharey=axs1)
#     axs5 = fig.add_subplot(grid[4:6, 0])
#     axs6 = fig.add_subplot(grid[4:6, 1], sharey=axs1)

#     for plot in [axs2, axs4, axs6]:
#    # for plot in [axs2, axs4, axs6, hist2, hist4, hist6]:
#         plt.setp(plot.get_yticklabels(), visible=False)
#         plt.setp(plot.set_ylabel(''), visible=False)

#     # for plot in [axs1, axs2, axs3, axs4, axs5, axs6, hist1, hist2, hist3, hist4]:
#     for plot in [axs1, axs2, axs3, axs4]:
#         plt.setp(plot.get_xticklabels(), visible=False)
#         plt.setp(plot.set_xlabel(''), visible=False)


    fig, axes = plt.subplots(1,6, figsize=(15,2.5), sharey=True)
    plt.subplots_adjust(wspace=0.05)
    # for i in range(3):
    axes = fig.get_axes()
    # plot = [axs1,axs2,axs3,axs4,axs5,axs6]
    # hist = [hist1,hist2,hist3,hist4,hist5,hist6]
    for i in range(len(axes)):
        axes[i].grid('x')
        # hist[i].grid('x')
        axes[i].scatter(np.array(x_test)[:,i],np.array(y_test), c="skyblue", s=1, alpha=0.9, zorder=1)
        instances = [2019, 122, 1837, 206, 2475, 223, 889, 2452, 532, 2082, 1007, 481, 1223, 645, 820, 2481, 1215, 893, 2495, 202, 246, 2139, 1140, 1047, 1489]
        axes[i].scatter(np.array(x_test)[:,i][instances],np.array(y_test)[instances], c="navy", s=20, alpha=0.9, zorder=2)

        axes[i].set_title(features[i], fontsize=12)
        if i ==0:
            axes[i].set_ylabel('Total Volume', fontsize=12)
        #     axes[i].set_yticklabels([])
        # else:
        # hist[i].set_ylabel('Count', fontsize=10)
    # legend = fig.legend(['Ground Truth', 'Prediction'], loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fancybox=True, shadow=False, fontsize = 12)
    legend = fig.legend(['Training Data', 'Random Instances'], loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=False, fontsize = 12, alignment='center')

    legend.legend_handles[0]._sizes = [30]
    legend.legend_handles[1]._sizes = [30]
    # legend.legend_handles[2]._sizes = [30]
    # shift = max([t.get_window_extent().width for t in legend.get_texts()])
    # # for t in legend.get_texts():
    # #     t.set_ha('center') # ha is alias for horizontalalignments
    # legend.legend_handles[1].set_position((shift,0))
    # legend.get_texts()[1].set_position((shift,0))
    # legend.legendHandles[0]._sizes = [30]
    # legend.legendHandles[1]._sizes = [30]
    # plt.subplots_adjust(left=0.1,
                    # bottom=0.1,
                    # right=0.9,
                    # top=0.9,
                    # wspace=0.4,
                    # hspace=0.4)
    # axs1.text(0.8,-2.75, 'Feature Value', transform=axs1.transAxes, fontsize=10)
    return fig

class MultivariateRegression():
    def __init__(self,dataprocessing):
        self.DataProcessing = dataprocessing
        self.y_pred = None

    def dataSplit(self, features):
        self.features = features

        df = self.DataProcessing.fetchData()
        x, y, _ = self.DataProcessing.augment(df)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # print('Computed Data: ', self.x_test)
        self.x_test_full = self.x_test
        self.allData = self.DataProcessing.allData

        self.allDataNormalised = self.DataProcessing.filter(self.allData, self.features, normalise=True)
        self.x_train = self.DataProcessing.filter(self.x_train, self.features, normalise=True)
        self.x_test = self.DataProcessing.filter(self.x_test, self.features, normalise=True)



    def linearTrain(self, featureNames):
        self.dataSplit(featureNames)
        model_ols = LinearRegression()
        model_ols.fit(self.x_train, self.y_train)

        return model_ols



    def predictor(self, features):
        if isinstance(features, int):
            featureNames = self.DataProcessing.independants[0:features]
        else:
            featureNames = features

        self.model = self.linearTrain(features)
        self.y_pred = self.model.predict(np.array(self.x_test))
        mse = mean_squared_error(np.array(self.y_test), self.y_pred, squared=False)
        print('RMSE Achieved by full model on training Data: ', mse)

        fig = plotPredictions(features, self.x_test, self.y_test, self.y_pred)
        # fig = plotPredictions(features, self.x_train, self.y_train, self.y_pred)

        fig.savefig('Figures/PredictionsMVR.pdf')
        plt.show()

    def eval(self,):
        features = ['Ordinal Date','Time Interval', 'Bank Holiday', 'Weekend', '0 - 520 cm', '521  - 660 cm', '661 - 1160 cm', '1160+ cm', 'Avg mph', 'Letters in Day', 'Monday', 'Morning', 'Afternoon', 'Evening', 'Night', 'Day']
        if not self.y_pred:
            self.predictor(features)

        mse = mean_squared_error(self.y_test, self.y_pred)
        print('Achieved MSE: ', mse)


class webTRIS():
    def __init__(self,dataprocessing, feature_names, model_type='SVR', load_model=False):
        self.load_model = load_model
        self.DataProcessing = dataprocessing
        self.y_pred = None
        self.model_type = model_type
        self.features = feature_names
        self.dataSplit()

    def dataSplit(self, ):

        df = self.DataProcessing.fetchData()
        df = df.rename(columns=self.features)
        self.features = list(self.features.values())
        x, y, _ = self.DataProcessing.augment(df)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # print('Computed Data: ', self.x_test)
        self.x_test_full = self.x_test
        self.allData = self.DataProcessing.allData
        self.allDataNormalised = self.DataProcessing.filter(self.allData, self.features, normalise=True)
        self.x_train = self.DataProcessing.filter(self.x_train, self.features, normalise=True)
        self.x_test = self.DataProcessing.filter(self.x_test, self.features, normalise=True)



    def train_model(self,):
        if self.model_type == 'RNN':
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            self.x_train = np.array(self.x_train)
            self.y_train = np.array(self.y_train)
            self.x_test = np.array(self.x_test)
            self.y_test = np.array(self.y_test)

            train_data = TensorDataset(torch.tensor(self.x_train), torch.tensor(self.y_train))
            self.train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
            test_data = TensorDataset(torch.tensor(self.x_test), torch.tensor(self.y_test))
            self.test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

            self.input_dim = len(self.features)
            self.output_dim = 1
            self.hidden_dim = 12
            self.layer_dim = 3
            self.batch_size = 64
            self.dropout = 0.1
            self.n_epochs = 500
            self.learning_rate = 1e-1
            self.weight_decay = 1e-6

            model_params = {'input_dim': self.input_dim,
                            'hidden_dim' : self.hidden_dim,
                            'layer_dim' : self.layer_dim,
                            'output_dim' : self.output_dim,
                            'dropout_prob' : self.dropout}

            self.model = RNNModel(self.input_dim, self.hidden_dim, self.layer_dim, self.output_dim, self.dropout)
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

            loss_fn = nn.L1Loss(reduction="mean")
            self.opt = Optimization(model=self.model.to(device), loss_fn=loss_fn, optimizer=optimizer)
#
            if self.load_model:
                self.model.load_state_dict(torch.load(f'saved/models/webTRIS_{self.model_type}.pck'))
            else:
                self.opt.train(self.train_dataloader, self.train_dataloader, batch_size=self.batch_size, n_epochs=self.n_epochs, n_features=self.input_dim)
#        print(self.load_model)
#
                torch.save(self.model.state_dict(), f'saved/models/webTRIS_{self.model_type}.pck')

            return self.opt.predictor_from_numpy

        else:
            if self.load_model:
                with open(f'saved/models/webTRIS_{self.model_type}.pck', 'rb') as file:
                    self.model = pck.load(file)

            else:
                if self.model_type == 'SVR':
                    self.model = SVR(kernel='poly', verbose=True)
                elif self.model_type == 'RF':
                    self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif self.model_type == 'GBR':
                    self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)

                np_x = np.array(self.x_train)
                np_y = np.array(self.y_train).ravel()
                self.model.fit(np_x, np_y)
                print('Saving model: ', self.model_type)
                with open(f'saved/models/webTRIS_{self.model_type}.pck', 'wb') as file:
                    pck.dump(self.model, file)

            self.y_test_pred = self.model.predict(np.array(self.x_test))
            self.y_train_pred = self.model.predict(np.array(self.x_train))

#            fig = midasPlotPredictions(self.features, self.x_test, self.y_test, self.y_test_pred)
#            fig.savefig(f'webTRIS/Figures/webTRIS_{self.model_type}.png')


            return self.model.predict

    def predictor(self, features):

        if isinstance(features, int):
            featureNames = self.DataProcessing.independants[0:features]
        else:
            featureNames = features

        #Try to estimate by plunging test data
        self.y_pred = self.model.predict(np.array(self.x_test))
        mse = mean_squared_error(np.array(self.y_test), self.y_pred, squared=False)
        print('RMSE Achieved by SVR model: ', mse)

#        fig = plotPredictions(features, self.x_test, self.y_test, self.y_pred)
#        fig.savefig('Figures/PredictionsSVR.png', dpi=600, bbox_inches='tight')
#
        return mse

    def eval(self,):
        features = ['Ordinal Date','Time Interval', 'Bank Holiday', 'Weekend', '0 - 520 cm', '521  - 660 cm', '661 - 1160 cm', '1160+ cm', 'Avg mph', 'Letters in Day', 'Monday', 'Morning', 'Afternoon', 'Evening', 'Night', 'Day']
        if not self.y_pred:
            mse = self.predictor(features)
