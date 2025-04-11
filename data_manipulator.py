#from chilli_example import MIDAS
import numpy as np
import random
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


def midas_data_manipulator(df):
    reassignment_prob = 0.8
    scaler = MinMaxScaler()
    cols = [col for col in df.columns if col != 'heathrow air_temperature'] + ['heathrow air_temperature']
    df = df[cols]#
#    df_scaled = scaler.fit_transform(df.to_numpy())
#    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
#    df = df_scaled

    X = df.to_numpy()
    y_orig = X[:, -1]  # Original target


    strong_feature = X[:, 0]  # The feature you want to dominate
    x=strong_feature
# Create a non-linear transformation

    amplitude = 1 + 0.3 * np.sin(x)
    frequency = ( 0.2 * x)

    y_strong = amplitude * np.sin(frequency * x)


# Add weaker linear effect from other features
# Combin    # Apply small non-linear transformations to all weak features
    weights = np.random.uniform(-0.05, 0.05, size=(len(X[0]) - 2))  # very small weights
    x_weak = X[:, 1:-1]
    x_weak_transformed = np.tanh(x_weak)  # or try np.sin, x**2, etc.
    y_weak = np.dot(x_weak_transformed, weights)
    y_new = y_strong + y_weak
    # Standardize the synthetic target
    y_new_std = (y_new - np.mean(y_new)) / np.std(y_new)
# Rescale to original target's mean and std
    y_modified = y_new_std * np.std(y_orig) + np.mean(y_orig)
    dominance = 0.6  # 0 to 1

    y_combined = dominance * y_strong + (1 - dominance) * y_weak
    y_combined_std = (y_combined - np.mean(y_combined)) / np.std(y_combined)
    y_modified = y_combined_std * np.std(y_orig) + np.mean(y_orig)
#
    X[:, -1] = y_modified
#    X[:, 1:-1] = x_weak_transformed
    df = pd.DataFrame(X, columns=df.columns)

#    df = scaler.inverse_transform(df_scaled.to_numpy())
#    df = pd.DataFrame(df, columns=df_scaled.columns)
    return df

def phm08_data_manipulator(df):
    reassignment_prob = 0.95
    print(df.columns)
    for row in df.iterrows():
           for f in df.columns:
                if f =='setting1':
                    if random.random() < reassignment_prob:
                        s = row[1][f]
                        rul = row[1]['RUL']
                        new_rul = rul + 50*s
                        df.at[row[0], 'RUL'] = new_rul
                else:
                    s = row[1][f] + random.uniform(-1,1)
                    df.at[row[0], f] = s
    return df

def housing_data_manipulator(df):
    reassignment_prob = 0
#    for i in range(len(df)):
#        if random.random() < reassignment_prob:
#        print('\n ------------')
#        print(df.iloc[i])
#            medInc = df.iloc[i]['MedInc']
#            houseAge = df.iloc[i]['HouseAge']
#
#            new_medInc = medInc*abs(math.sin(houseAge))
#            new_houseAge = houseAge*abs(math.cos(medInc))
#
#            medHouseVal = df.iloc[i]['MedHouseVal']
#            df.at[i, 'MedHouseVal'] = medHouseVal*abs(math.sin(new_medInc +new_houseAge))
#            df.at[i, 'MedInc'] = new_medInc
#            df.at[i, 'HouseAge'] = new_houseAge
#            for f in df.columns:
#                if f not in ['MedInc', 'HouseAge', 'MedHouseVal']:
#                    s = df.iloc[i][f]
#                    df.at[i, f] = s+s*random.uniform(0,1)
#    np_df = df.to_numpy()
#    strong_feature = np_df[:,0]
#    new_strong_feature = np.sin(strong_feature * np.pi) + 0.3 * strong_feature**2
#
#    remaining_features = np_df[:,1:-1]
#    weak_y = 0.1 * np.dot(remaining_features, np.random.uniform(-1, 1, size=(len(df.columns)-2)))
#    noise = np.random.normal(0, 0.5, size=len(df))
#
## Final target is dominated by y_strong
#    y = new_strong_feature + weak_y + noise
#
#    np_df[:,0] = new_strong_feature
##    np_df[:,1:-1] = remaining_features
#    np_df[:,-1] = y
#    df = pd.DataFrame(np_df, columns=df.columns)

    X = df.to_numpy()
    y_orig = X[:, -1]  # Original target


    strong_feature = X[:, 0]  # The feature you want to dominate
    x=strong_feature
# Create a non-linear transformation

    amplitude = 1 + 0.3 * np.sin(x)
    frequency = ( 0.2 * x)

    y_strong = amplitude * np.sin(frequency * x)


# Add weaker linear effect from other features
# Combin    # Apply small non-linear transformations to all weak features
    weights = np.random.uniform(-0.05, 0.05, size=(len(X[0]) - 2))  # very small weights
    x_weak = X[:, 1:-1]
    x_weak_transformed = np.tanh(x_weak)  # or try np.sin, x**2, etc.
    y_weak = np.dot(x_weak_transformed, weights)
    y_new = y_strong + y_weak
    # Standardize the synthetic target
    y_new_std = (y_new - np.mean(y_new)) / np.std(y_new)
# Rescale to original target's mean and std
    y_modified = y_new_std * np.std(y_orig) + np.mean(y_orig)
    dominance = 0.6  # 0 to 1

    y_combined = dominance * y_strong + (1 - dominance) * y_weak
    y_combined_std = (y_combined - np.mean(y_combined)) / np.std(y_combined)
    y_modified = y_combined_std * np.std(y_orig) + np.mean(y_orig)
#
#    X[:, -1] = y_modified
#    X[:, 1:-1] = x_weak_transformed
    df = pd.DataFrame(X, columns=df.columns)


    return df




def attribution_percentage_score(feature_contributions, important_feature_idxs):
    feature_contributions = [abs(f) for f in feature_contributions]
    attr_per = np.mean([feature_contributions[f]/sum(feature_contributions) for f in important_feature_idxs])
    return attr_per

def shap_attribution_percentage_score(shap_values, important_features):
    feature_names = shap_values[0].feature_names
    importances = shap_values[0].values
    feature_contributions = [abs(imp) for imp in importances]
    important_feature_idxs = [i for i, f in enumerate(feature_names) if f in important_features]
    attr_per = np.mean([feature_contributions[f]/sum(feature_contributions) for f in important_feature_idxs])
    return attr_per

def fidelity_via_feature_removal(explanation, instance, predictor, model_pred):
    errors = []
    importance_scores = np.array(explanation)
    magnitudes = [abs(f) for f in importance_scores]
    sorted_importances = importance_scores[np.argsort(magnitudes)]
    instance = instance[np.argsort(magnitudes)]
#    print(list(zip(sorted_importances, instance)))

#    for i in range(len(importance_scores)):
#        if mode == 'SHAP':
#            pred = np.sum(sorted_importances[:i]) + intercept
#        else:
#            pred = np.dot(instance[:i], sorted_importances[:i]) + intercept
    for i in range(len(importance_scores)):
        new_input = instance.copy()
        top_features = np.argsort(magnitudes)[i:]
        new_input[top_features] = random.random()
        m_i = new_input.reshape(1, -1)
        prediction = predictor(m_i)
        errors.append(abs(model_pred - prediction))
#        print(f'Error: {abs(model_pred - prediction)}')


#    fig, ax = plt.subplots()
#    ax.plot(range(len(importance_scores)), errors)
#    ax.set_xlabel('Number of Features Removed')
#    ax.set_ylabel('Error')
#    ax.set_title('Fidelity via Feature Removal')
#    plt.show()

    return errors




#x_train, x_test, y_train, y_test, features = midas_runner.data.X_train, midas_runner.data.X_test, midas_runner.data.y_train, midas_runner.data.y_test, midas_runner.data.trainingFeatures


