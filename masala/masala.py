from .llc_explainer import LLCExplanation
from .llc_ensemble_generator import LLCGenerator

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors
import plotly.express as px



class MASALA:

    def __init__(self, model, model_type, x_test, y_test, y_pred, dataset, features, target_feature, discrete_features, sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold, num_workers=1, preload_clustering=True, experiment_id=1):
        self.model = model
        self.model_type = model_type
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.dataset = dataset
        self.features = features
        self.target_feature = target_feature
        self.discrete_features = discrete_features
        self.sparsity_threshold = sparsity_threshold
        self.coverage_threshold = coverage_threshold
        self.starting_k = starting_k
        self.neighbourhood_threshold = neighbourhood_threshold
        self.explanation_generator = None
        self.experiment_id = experiment_id
        self.num_workers = num_workers

        if preload_clustering:
            self.initialise_explainer()



    def run_clustering(self, ):
        self.clustering_generator = LLCGenerator(model=self.model, model_type=self.model_type, x_test=self.x_test, y_pred=self.y_pred, features=self.features, target_name=self.target_feature, discrete_features=self.discrete_features, dataset=self.dataset, sparsity_threshold=self.sparsity_threshold, coverage_threshold=self.coverage_threshold, starting_k=self.starting_k, neighbourhood_threshold=self.neighbourhood_threshold, experiment_id=self.experiment_id, num_workers=self.num_workers)
        self.feature_ensembles = self.clustering_generator.feature_ensembles
        self.initialise_explainer()

#        self.clustering_generator.matplot_all_clustering()

    def initialise_explainer(self,):
        self.explanation_generator = LLCExplanation(model=self.model, model_type=self.model_type, x_test=self.x_test, y_pred=self.y_pred, dataset=self.dataset, features=self.features, discrete_features=self.discrete_features, sparsity_threshold=self.sparsity_threshold, coverage_threshold=self.coverage_threshold, starting_k=self.starting_k, neighbourhood_threshold=self.neighbourhood_threshold, experiment_id=self.experiment_id)
        self.feature_ensembles = self.explanation_generator.feature_ensembles

    def explain_instance(self, instance, perturbations=False):
        explanation, perturbation_error = self.explanation_generator.generate_explanation(self.x_test[instance], instance, perturbations=perturbations)
#        if plotting:
#            self.explanation_generator.interactive_exp_plot(explanation)

        return explanation, perturbation_error




    def plot_data(self, explanation=None, features_to_plot=None, instances_to_show=None):
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

        for feature in features_to_plot:
            i = features_to_plot.index(feature)
            feature_index = self.features.index(feature)
            if i == 0:
                showlegend=True
            else:
                showlegend=False
            colour = np.random.rand(3)
            colour = plotly.colors.label_rgb((colour[0]*255, colour[1]*255, colour[2]*255))

            if explanation == None:
                fig.add_trace(go.Scatter(x=self.x_test[:,feature_index],y=self.y_pred,
                                         mode='markers', marker = dict(size=3, opacity=0.9, color='black'),
                                         showlegend=showlegend, name='Test Data', legendgroup='Test Data'),
                              row=axes[i][0], col=axes[i][1])
            else:
                data_instance = explanation.instance_data
                target_idx = explanation.target_idx
                local_x = explanation.local_x
                local_y_pred = explanation.local_model_y
                exp_local_y_pred = explanation.local_exp_y
                instance_prediction = explanation.target_model_y
                exp_instance_prediction = explanation.target_exp_y


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
                fig.layout.annotations[i].update(text=f'{feature} : <br> {round(data_instance[feature_index], 3)}')
#            fig.update_xaxes(title='Normalised Feature Value', range=[min([min(self.x_test[:,feature_index])*1.1, -0.1]), max(self.x_test[:,i])*1.1], row=axes[i][0], col=axes[i][1])
#            fig.update_yaxes(title='Predicted RUL',range=[min_y*-1.1, max_y*1.1], row=axes[i][0], col=axes[i][1])
        if len(features_to_plot) == 1:
            height = 750
        elif len(features_to_plot) in [2,3]:
            height = 600
        else:
            height=350*num_rows
        fig.update_layout(legend=dict(yanchor="top", xanchor="auto", orientation='h', y=-0.25),
                          height=height)
#        fig.write_html(f'Figures/{self.dataset}/perturbations_{target_idx}.html', auto_open=False)
        return fig

    def plot_all_clustering(self,instance=None, features_to_plot=None):
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


            min_y = np.min(self.y_pred)
            max_y = np.max(self.y_pred)
            fig.update_xaxes(title='Normalised Feature Value', range=[min([min(self.x_test[:,feature_index])*1.1, -0.1]), max(self.x_test[:,i])*1.1], row=axes[i][0], col=axes[i][1])
            fig.update_yaxes(title=self.target_feature,range=[min_y*-1.1, max_y*1.1], row=axes[i][0], col=axes[i][1])
        if len(features_to_plot) == 1:
            height = 750
        elif len(features_to_plot) in [2,3]:
            height = 500
        else:
            height=350*num_rows
        fig.update_layout(height=height+100)
#        fig.write_html(f'Figures/{self.dataset}/Clustering/{self.dataset}#_{self.experiment_id}_clustering_{self.sparsity_threshold}_{self.coverage_threshold}_{self.starting_k}_{self.neighbourhood_threshold}.html')

        return fig
