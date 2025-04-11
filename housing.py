import pandas as pd
import numpy as np
import pickle as pck
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from midas_model_data import plotPredictions
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from data_manipulator import housing_data_manipulator

class CaliforniaHousingModel:
    def __init__(self, model_type, test_size=0.2, random_state=42):
        # Default to RandomForest if no model is passed
#        self.model = model if model else RandomForestRegressor(random_state=random_state, n_estimators=100)
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        # Load and convert to pandas DataFrame
        data = fetch_california_housing()
        self.df = pd.DataFrame(data.data, columns=data.feature_names)
        self.df['MedHouseVal'] = data.target
#        self.df = housing_data_manipulator(self.df)
        print(self.df.head())
        # plot data

    def preprocess(self):
        # Basic train-test split
        X = self.df.drop(columns='MedHouseVal')
        y = self.df['MedHouseVal']

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(np.array(X))
#        y = self.scaler.fit_transform(np.array(y).reshape(-1, 1))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def train(self,):
        # Train the model
        if self.model_type == 'GBR':
            self.model = GradientBoostingRegressor(max_depth=10, n_estimators=500, random_state=42, verbose=True)
        elif self.model_type == 'SVR':
            self.model = SVR(kernel='rbf', verbose=True)
        elif self.model_type == 'RF':
            self.model = RandomForestRegressor(random_state=42, n_estimators=200, verbose=True)
        self.model.fit(self.X_train, self.y_train)
        with open(f'saved/models/housing_{self.model_type}.pck', 'wb') as file:
            pck.dump(self.model, file)

    def evaluate(self):
        # Predict and evaluate
        y_pred = self.model.predict(self.X_test)
        features = self.df.columns[:-1]
        fig =plotPredictions(features, self.X_test, self.y_test, y_pred)
        fig.savefig('housing/Figures/CaliforniaHousing.png')
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        return mse, r2

    def run_pipeline(self):
        print("Loading data...")
        self.load_data()
        print("Preprocessing...")
        self.preprocess()
        print("Training model...")
        self.train()
        print("Evaluating model...")
        return self.evaluate()

if __name__ == "__main__":
    model_runner = CaliforniaHousingModel()
    model_runner.run_pipeline()

