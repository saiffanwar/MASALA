from datasets import BaseModelData
from dash import Dash, dcc, html, Input, Output, callback, callback_context
import dash_daq as daq
from masala import masala
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import multiprocessing


model_type='GBR'
dataset='webTRIS'
load_model=True

base_model = BaseModelData(dataset, load_model, model_type)

MASALAExplainer = masala.MASALA(base_model.model, model_type, base_model.x_test, base_model.y_test, base_model.y_test_pred, dataset, base_model.features, base_model.target_feature, base_model.discrete_features, sparsity_threshold=0.02, coverage_threshold=0.05, starting_k=5, neighbourhood_threshold=0.05, preload_clustering=True)

fig = MASALAExplainer.plot_data()
app = Dash(__name__)



app.layout = html.Div(
    children=[
        html.H1(children='MASALA Viewer'),
        # Dropdown to select view
        html.Div(
                style={
                    'display': 'flex',
                    'gap': '20px',
                    'margin-bottom': '20px',
                    'flex-wrap': 'wrap'  # Allows wrapping on smaller screens
                },
            children=[
                    html.Div(
                        children=[
                            html.Label("View Mode", style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id='view-dropdown',
                                options=[
                                    {'label': 'Show Clustering', 'value': 'clustering'},
                                    {'label': 'Show Explanation', 'value': 'explanation'},
                                    {'label': 'Show Data', 'value': 'data'},
                                ],
                                value='clustering',
                                clearable=False
                            )
                        ],
                        style={'width': '30%'}
                    ),
                    html.Div(
                        children=[
                            html.Label("Dataset", style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id='dataset-dropdown',
                                options=[
                                    {'label': 'MIDAS', 'value': 'MIDAS'},
                                    {'label': 'California Housing', 'value': 'housing'},
                                    {'label': 'webTRIS', 'value': 'webTRIS'},
                                ],
                                value='MIDAS',
                                clearable=False
                            )
                        ],
                        style={'width': '30%'}
                    ),
                    html.Div(
                        children=[
                            html.Label("Model", style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=[
                                    {'label': 'Gradient Boosting Regressor', 'value': 'GBR'},
                                    {'label': 'Support Vector Regressor', 'value': 'SVR'},
                                    {'label': 'Random Forest', 'value': 'RF'},
                                ],
                                value='GBR',
                                clearable=False
                            )
                        ],
                        style={'width': '30%'}
                    ),
                ]
            ),

        dcc.Graph(
            id='masala-graph',
            figure=fig
        ),
    ]
)
@app.callback(
    Output('masala-graph', 'figure'),
    Output('model-dropdown', 'options'),
    Output('model-dropdown', 'value'),
    Input('view-dropdown', 'value'),
    Input('dataset-dropdown', 'value'),
    Input('model-dropdown', 'value'),
)
def toggle_view(view_mode, dataset, model_type):
    global MASALAExplainer
    if callback_context.triggered[0]['prop_id'] == 'dataset-dropdown.value':
        base_model = BaseModelData(dataset, load_model, model_type)
        MASALAExplainer = masala.MASALA(base_model.model, model_type, base_model.x_test, base_model.y_test, base_model.y_test_pred, dataset, base_model.features, base_model.target_feature, base_model.discrete_features, sparsity_threshold=0.02, coverage_threshold=0.05, starting_k=5, neighbourhood_threshold=0.05, preload_clustering=True)

    elif callback_context.triggered[0]['prop_id'] == 'model-dropdown.value':
        base_model = BaseModelData(dataset, load_model, model_type)
        MASALAExplainer = masala.MASALA(base_model.model, model_type, base_model.x_test, base_model.y_test, base_model.y_test_pred, dataset, base_model.features, base_model.target_feature, base_model.discrete_features, sparsity_threshold=0.02, coverage_threshold=0.05, starting_k=5, neighbourhood_threshold=0.05, preload_clustering=True)

    if dataset == 'housing':
        model_types = [
            {'label': 'Gradient Boosting Regressor', 'value': 'GBR'},
            {'label': 'Support Vector Regressor', 'value': 'SVR'},
            {'label': 'Random Forest', 'value': 'RF'},
        ]
    elif dataset == 'MIDAS':
        model_types = [
            {'label': 'Gradient Boosting Regressor', 'value': 'GBR'},
            {'label': 'Support Vector Regressor', 'value': 'SVR'},
            {'label': 'Recurrent Neural Network', 'value': 'RNN'},
        ]
    elif dataset == 'webTRIS':
        model_types = [
            {'label': 'Gradient Boosting Regressor', 'value': 'GBR'},
            {'label': 'Support Vector Regressor', 'value': 'SVR'},
            {'label': 'Random Forest', 'value': 'RF'},
        ]

    print(f'Generating Plot for {MASALAExplainer.dataset} with {MASALAExplainer.model_type} model')
    if view_mode == 'explanation':
        explanation, local_error = MASALAExplainer.explain_instance(instance=50)
        fig = MASALAExplainer.plot_data(explanation=explanation)
    elif view_mode == 'clustering':
        fig = MASALAExplainer.plot_all_clustering()
    elif view_mode == 'data':
        fig = MASALAExplainer.plot_data()
    return fig, model_types, model_type


if __name__ == "__main__":
    app.run_server(debug=True)
