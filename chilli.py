import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from limeLocal import lime_tabular
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.utils.extmath import safe_sparse_dot
from pprint import pprint

#Use searborn
import seaborn as sns
#sns.set(style='whitegrid')

chilli_color = '#dc0000'
lime_color = '#10a500'


def exp_sorter(exp_list, features):

    explained_features = [i[0] for i in exp_list]
    for e in explained_features:

        for f in features:
            if f in e:
                if f in e.split(' ') or f in e.split('=') or f in e.split('>') or f in e.split('<'):
                    explained_features[explained_features.index(e)] = f

    feature_contributions = {f:[] for f in features}
    contributions = [e[1] for e in exp_list]

    explained_feature_indices = [features.index(i) for i in explained_features]
    for f in features:
        for num, e in enumerate(explained_features):
            if e == f:
#                    sorted_exp.append(e[1])
                feature_contributions[f].append(contributions[explained_features.index(e)])
    sorted_exp = [feature_contributions[f][0] for f in features]

    return sorted_exp


class CHILLI():

    def __init__(self, dataset, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features, using_chilli=True):
        self.dataset = dataset
        self.model = model
        # These should be scaled numpy arrays
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_pred = y_train_pred
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        self.features = features
        self.using_chilli = using_chilli


    def build_explainer(self, categorical_features=None, kernel_width=None, mode='regression'):
#        The explainer is built herem on the training data with the features and type of model specified.
#        y_hat_test = self.model.predict(self.x_test)
        self.explainer = lime_tabular.LimeTabularExplainer(self.x_train, test_data=self.x_test, test_labels=self.y_test, test_predictions=self.y_test_pred, feature_names=self.features, categorical_features=categorical_features, mode=mode, verbose=False, kernel_width=kernel_width)

    def make_explanation(self, predictor, instance, num_features=25, num_samples=1000):
        ground_truth = self.y_test[instance]
        instance_prediction = predictor(self.x_test[instance].reshape(1,-1))[0]
        exp, local_model, perturbations, model_perturbation_predictions, exp_perturbation_predictions = self.explainer.explain_instance(self.x_test[instance], instance_num=instance, predict_fn=predictor, num_features=num_features, num_samples=num_samples, using_chilli=self.using_chilli)
        self.local_model = local_model
        exp_instance_prediction = exp_perturbation_predictions[0]
        model_perturbation_predictions = np.array(model_perturbation_predictions).flatten()
        exp_perturbation_predictions = np.array(exp_perturbation_predictions).flatten()

#        print(f'Ground Truth: {ground_truth}')
#        print(f'Model Prediction: { instance_prediction }')

        return exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, instance_prediction, exp_instance_prediction
#
#

    def plot_perturbations(self, instance, exp, kernel_width, perturbations, model_perturbation_predictions, exp_perturbation_predictions):
        # Compute error
        exp_list = exp.as_list()

        feature_contributions = exp_sorter(exp_list, self.features)
        explained_features = self.features
        explained_features_x_test = self.x_test
#
        instance_x, instance_model_y, instance_exp_y = np.array(perturbations[0]), np.array(model_perturbation_predictions[0]), np.array(exp_perturbation_predictions[0])
        perturbations_x, perturbations_model_y, perturbations_exp_y = np.array(perturbations[1:]), np.array(model_perturbation_predictions[1:]), np.array(exp_perturbation_predictions[1:])
#        perturbations_model_y = [int(i) for i in perturbations_model_y]
        explanation_error = mean_absolute_error(perturbations_model_y, perturbations_exp_y)
        perturbation_weights = exp.weights[1:]
        feature_contributions = exp_sorter(exp_list, self.features)

# Layout
        num_features = len(self.features)
        num_cols = len(self.features)
        num_rows = int(np.ceil(num_features / num_cols)) + 1  # +1 for top row

#        fig = plt.figure(figsize=(15, 3 + 2 * num_rows))
#        gs = gridspec.GridSpec(num_rows, num_cols, height_ratios=[1] + [2] * (num_rows - 1), hspace=0.5, wspace=0.3)
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))

# --- Top row plots ---
# Explanation convergence (left side)

# Feature contribution bar chart (right side)
#        ax_top_right = fig.add_subplot(gs[0, :])
#        colors = ['green' if x >= 0 else 'red' for x in feature_contributions]
#        ax_top_right.barh(explained_features, feature_contributions, color=colors)
#        ax_top_right.set_title('Feature Significance')

# --- Feature scatter subplots ---
        for i, feature in enumerate(self.features):
            row = i // num_cols + 1
            col = i % num_cols
#            ax = fig.add_subplot(gs[row, col])
            ax = fig.get_axes()[i]

            ax.scatter(explained_features_x_test[:, i], self.y_test_pred, color='lightgrey', s=10)
            ax.scatter(perturbations_x[:, i], perturbations_model_y, c=perturbation_weights, cmap='Oranges', s=10)
            ax.scatter(perturbations_x[:, i], perturbations_exp_y, c=perturbation_weights, cmap='Greens', s=10)
            ax.scatter([instance_x[i]], [instance_model_y], color='red', s=50)
            ax.scatter([instance_x[i]], [exp.local_model.predict(instance_x.reshape(1, -1))[0]], color='blue', s=30)

            ax.set_title(feature)
            if row == 1 and col == 0:
                ax.legend(fontsize=8)

# --- Main title ---
#        fig.suptitle(
#            f'Explanation for instance {instance}\n'
#            f'Explanation Error = {explanation_error:.2f}\n'
#            f'Model Instance Prediction: {instance_model_y}\n'
#            f'Explanation Instance Prediction: {instance_exp_y}',
#            fontsize=16,
#            x=0.1, y=0.99, ha='left'
#        )
            if i ==0 or i == 4:
                ax.set_ylabel('Predicted Value')
            else:
                ax.set_yticks([])

#        plt.tight_layout(rect=[0, 0, 1, 0.95])

# --- Save figure ---
        fig.subplots_adjust(hspace=0.3)
        fig.legend(['Test Data', 'Model Local Predictions', 'Surrogate Local Predictions', 'Surrogate ', 'Model Instance Prediction', 'Surrogate Instance Prediction'], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
        suffix = '_CHILLI' if self.using_chilli else '_LIME'
        fig.savefig(f'{self.dataset}/Figures/Explanations/instance_{instance}{suffix}_kw={kernel_width}_explanation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_perturbation_distribution(self, chilli_perturbations, lime_perturbations, instance_idx):

        chilli_perturbations = np.array(chilli_perturbations)
        lime_perturbations = np.array(lime_perturbations)
        instance = self.x_test[instance_idx]


        fig, axes = plt.subplots(3, len(self.features), figsize=(15, 4), sharex='col')

        for i in range(3):
            for f in range(len(self.features)):
                ax = axes[i][f]

                # Set histogram data and color based on row
                if i == 0:
                    data = chilli_perturbations[:, f]
                    color = chilli_color
                elif i == 1:
                    data = lime_perturbations[:, f]
                    color = lime_color
                elif i == 2:
                    data = self.x_test[:, f]
                    color = 'blue'

                # Plot the main histogram
                ax.hist(data, bins=20, color=color, alpha=0.5)

                # Add instance value as a vertical line
                ax.axvline(instance[f], color='red', linewidth=2, label='Instance')

                # Add feature name as title for the top row
                if i == 0:
                    ax.set_title('Heathrow \n'+f'{self.features[f].split(' ')[-1]}')
        fig.legend(['CHILLI', 'LIME', 'Training Data', 'Instance'], loc='upper center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.05))
        fig.savefig(f'{self.dataset}/Figures/Explanations/instance_{instance_idx}_perturbation_distribution.png', dpi=300, bbox_inches='tight')

    def compare_kw_perturbations(self, kws, kw_results, instance,):
        kws = kws[:5]
        kw_results = kw_results[:5]
        fig, axes = plt.subplots(len(kws), len(self.features), figsize=(20, 2* len(kws)))

        for k, kw in enumerate(kws):
            exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions = kw_results[k]

            exp_list = exp.as_list()

            feature_contributions = exp_sorter(exp_list, self.features)
            explained_features = self.features
            explained_features_x_test = self.x_test
#
            instance_x, instance_model_y, instance_exp_y = np.array(perturbations[0]), np.array(model_perturbation_predictions[0]), np.array(exp_perturbation_predictions[0])
            perturbations_x, perturbations_model_y, perturbations_exp_y = np.array(perturbations[1:]), np.array(model_perturbation_predictions[1:]), np.array(exp_perturbation_predictions[1:])
#        perturbations_model_y = [int(i) for i in perturbations_model_y]
            explanation_error = mean_absolute_error(perturbations_model_y, perturbations_exp_y)
            perturbation_weights = exp.weights[1:]
            feature_contributions = exp_sorter(exp_list, self.features)
            for i, feature in enumerate(self.features):

#            ax = fig.add_subplot(gs[row, col])
                ax = axes[k][i]

                ax.scatter(explained_features_x_test[:, i], self.y_test_pred, color='lightgrey', s=3)
                ax.scatter(perturbations_x[:, i], perturbations_model_y, c=perturbation_weights, cmap='Oranges', s=3)
                ax.scatter(perturbations_x[:, i], perturbations_exp_y, c=perturbation_weights, cmap='Greens', s=3)
                ax.scatter([instance_x[i]], [instance_model_y], color='red', s=50)
                ax.scatter([instance_x[i]], [exp.local_model.predict(instance_x.reshape(1, -1))[0]], color='blue', s=30)

                if k == 0:
                    ax.set_title(feature, fontsize=8)
                if i == 0:
                    ax.set_ylabel(r'$\sigma$='+f'{kw}'+'\n\nPredicted Value')
                else:
                    ax.set_yticks([])
#                if k == 0 and col == 0:
#                    ax.legend(fontsize=8)

# --- Main title ---
#        fig.suptitle(
#            f'Explanation for instance {instance}\n'
#            f'Explanation Error = {explanation_error:.2f}\n'
#            f'Model Instance Prediction: {instance_model_y}\n'
#            f'Explanation Instance Prediction: {instance_exp_y}',
#            fontsize=16,
#            x=0.1, y=0.99, ha='left'
#        )
            if i ==0:
                ax.set_ylabel('Predicted Value')
            else:
                ax.set_yticks([])

# Layout
        fig.legend(['Test Data', 'Model Local Predictions', 'Surrogate Local Predictions', 'Surrogate ', 'Model Instance Prediction', 'Surrogate Instance Prediction'], loc='upper center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0.95))
        suffix = '_CHILLI' if self.using_chilli else '_LIME'
        fig.savefig(f'{self.dataset}/Figures/Explanations/instance_{instance}{suffix}_explanation.png', dpi=300, bbox_inches='tight')
        plt.close()


    def interactive_perturbation_plot(self, instance, exp, kernel_width, perturbations, model_perturbation_predictions, exp_perturbation_predictions):

        exp_list = exp.as_list()

        feature_contributions = exp_sorter(exp_list, self.features)
        explained_features = self.features
        explained_features_x_test = self.x_test
#
        instance_x, instance_model_y, instance_exp_y = np.array(perturbations[0]), np.array(model_perturbation_predictions[0]), np.array(exp_perturbation_predictions[0])
        perturbations_x, perturbations_model_y, perturbations_exp_y = np.array(perturbations[1:]), np.array(model_perturbation_predictions[1:]), np.array(exp_perturbation_predictions[1:])
#        perturbations_model_y = [int(i) for i in perturbations_model_y]
        explanation_error = mean_absolute_error(perturbations_model_y, perturbations_exp_y)
#        print(perturbations_model_y)
#        fig = plt.figure(figsize=(14,12))
#        plt.plot(perturbations_x[:,0], perturbations_model_y, 'o', color='green')
#        plt.plot(perturbations_x[:,0], perturbations_exp_y, 'o', color='orange')
#        plt.plot(instance_x[0], instance_model_y, 'o', color='red')
#        plt.show()


        perturbation_weights = exp.weights[1:]

        num_rows=int(np.ceil(len(explained_features)/4))+1
        num_cols=4

        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[0.25, 0.25, 0.25, 0.25], row_heights =[0.33]+[0.16]*(num_rows-1),
                            specs = [
                                [{'colspan':2}, None, {'colspan':2}, None],
                                ]+[[{}, {}, {}, {}] for i in range(num_rows-1)], subplot_titles=['Explanation Prediction Convergence', 'Feature Significance']+explained_features,
                            horizontal_spacing=0.05, vertical_spacing=0.05)

        colours = ['green' if x>= 0 else 'red' for x in feature_contributions]

        # Plot explanation bar chart
        fig.add_trace(go.Bar(x=feature_contributions, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=3)

        axes = [[row, col] for row in range(2,num_rows+1) for col in range(1,num_cols+1)]


#
        for i in range(len(self.features)):
#        fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_exp_y, mode='markers', marker = dict(color='orange', size=3)), row=ax[0], col=ax[1])
            if i==0:
                showlegend=True
            else:
                showlegend=False
            fig.add_trace(go.Scatter(x=explained_features_x_test[:,i],y=self.y_test_pred,
                                     mode='markers', marker = dict(color='lightgrey', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Test data'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_model_y,
                                     mode='markers', marker = dict(color=perturbation_weights, colorscale='Oranges', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Model (f) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_exp_y,
                                     mode='markers', marker = dict(color=perturbation_weights, colorscale='Greens', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Explanation (g) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance_x[i]],y=[instance_model_y],
                                     mode='markers', marker = dict(color='red', size=20),
                                     showlegend=showlegend, name='Instance being explained'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance_x[i]],y=[exp.local_model.predict(instance_x.reshape(1,-1))],
                                     mode='markers', marker = dict(color='blue', size=10, opacity=0.9),
                                     showlegend=showlegend, name='Explanation (g) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])


        fig.update_layout(title=dict(text = f' Explanation for instance {instance} <br> Explanation Error = {explanation_error:.2f} <br> Model Instance Prediction {instance_model_y} <br> Explanation Instance Prediction {instance_exp_y}', y=0.99, x=0),
                          font=dict(size=14),
                          legend=dict(yanchor="top", y=1.1, xanchor="right"),
                          height=300*num_rows, )
        if self.using_chilli == True:
            suffix = '_CHILLI'
        elif self.using_chilli == False:
            suffix = '_LIME'

        fig.write_html(f'{self.dataset}/Figures/Explanations/instance_{instance}{suffix}_kw={kernel_width}_explanation.html', auto_open=False)
