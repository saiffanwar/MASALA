import numpy as np
import pickle as pck
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import sys
from chilli import exp_sorter
import argparse

features = {
        'MIDAS': ['heathrow wind_speed', 'heathrow wind_direction', 'heathrow cld_ttl_amt_id', 'heathrow cld_base_ht_id_1', 'heathrow visibility', 'heathrow msl_pressure', 'heathrow rltv_hum', 'heathrow prcp_amt', 'Date'],
        'housing': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'Latitude', 'Longitude'],
        'PHM08': ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'],
        'webTRIS': ['Time Interval', '0 - 520 cm', '521  - 660 cm', '661 - 1160 cm', 'Day', 'Avg mph']
        }

cleaned_dataset_names = {'MIDAS': 'MIDAS', 'housing': 'California Housing', 'PHM08': 'PHM08', 'webTRIS': 'WebTRIS'}

def calculate_local_fidelity(model_perturbation_predictions, exp_perturbation_predictions, weights=None):
    model_perturbation_predictions = np.array(model_perturbation_predictions).flatten()
    exp_perturbation_predictions = np.array(exp_perturbation_predictions).flatten()
#    print(f'Locally calculated error: {mean_absolute_error(model_perturbation_predictions, exp_perturbation_predictions)}')
    return mean_absolute_error(model_perturbation_predictions, exp_perturbation_predictions, sample_weight=weights)

def calculate_instance_fidelity(model_instance_prediction, exp_instance_prediction):
    return np.abs(model_instance_prediction - exp_instance_prediction)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='webTRIS')
parser.add_argument('--model_type', type=str, default='GBR')
parser.add_argument('--experiment_num', type=int, default=1)
parser.add_argument('--num_instances', type=int, default=30)
parser.add_argument('--kernel_width', type=float, default=None)
args = parser.parse_args()

print(f' Getting Results for {args.dataset}_{args.model_type}#_{args.experiment_num}_{args.num_instances}_kw={args.kernel_width}.pck')


chilli_color = '#dc0000'
chilli_color_2 = '#e77070'
lime_color = '#10a500'
lime_color_2 = '#84bc7e'

if args.kernel_width == None:
#    kernel_widths = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    kernel_widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
else:
    kernel_widths = [args.kernel_width]
#kernel_widths = [kernel_widths]
if args.dataset == 'MIDAS':
    models = ['GBR', 'SVR', 'RNN']
elif args.dataset == 'housing':
    models = ['GBR', 'SVR', 'RF']
elif args.dataset == 'PHM08':
    models = ['GBR', 'SVR', 'RF']
elif args.dataset == 'webTRIS':
    models = ['SVR', 'RNN', 'GBR']

def fidelities():

    fig, ax = plt.subplots(1, len(models), figsize = (8, 2), sharey=True)

    table_results = {kw: {model: {'LIME': {'local_fidelity': None, 'instance_fidelity': None}, 'CHILLI': {'local_fidelity': None, 'instance_fidelity': None}} for model in models} for kw in kernel_widths}

    for m, model in enumerate(models):
        lime_local_fidelity = []
        lime_instance_fidelity = []
        chilli_local_fidelity = []
        chilli_instance_fidelity = []
        for kernel_width in kernel_widths:
            print(f'\n --------- Kernel Width: {kernel_width} --------')

            for method in ['LIME', 'CHILLI']:
                with open(f'saved/results/{args.dataset}/{method}_{args.dataset}_{model}#_{args.experiment_num}_{args.num_instances}_kw={kernel_width}.pck', 'rb') as file:
                    result = pck.load(file)

                print(f'\n --------- {method} Results --------')
                print(result['instance_indices'])
                br
                # Local Fidelity
                local_fidelity_scores = []
                for n in range(len(result['model_perturbation_predictions'])):

#            perturbations_model_y, perturbations_exp_y = np.array(result['model_perturbation_predictions'][1:]), np.array(result['exp_perturbation_predictions'][1:])
#            perturbations_model_y = [i[0] for i in perturbations_model_y]
##        perturbations_model_y = [int(i) for i in perturbations_model_y]
#            explanation_error = mean_squared_error(perturbations_model_y, perturbations_exp_y)
#            print(f'Explanation Error: {explanation_error}')
                    local_fidelity_scores.append(calculate_local_fidelity(result['model_perturbation_predictions'][n],
                                                                result['exp_perturbation_predictions'][n], result['explanations'][n].weights))

                avg_local_fidelity = np.mean(local_fidelity_scores)
                if method == 'LIME':
                    lime_local_fidelity.append(avg_local_fidelity)
                else:
                    chilli_local_fidelity.append(avg_local_fidelity)
                print(f'Local Fidelity: {avg_local_fidelity}')

                # Instance Fidelity
                instance_fidelity_scores = []
                for n in range(len(result['model_instance_prediction'])):
                    instance_fidelity_scores.append(calculate_instance_fidelity(result['model_instance_prediction'][n], result['exp_instance_prediction'][n]))
                avg_instance_fidelity = np.mean(instance_fidelity_scores)
                if method == 'LIME':
                    lime_instance_fidelity.append(avg_instance_fidelity)
                else:
                    chilli_instance_fidelity.append(avg_instance_fidelity)

                table_results[kernel_width][model][method]['local_fidelity'] = avg_local_fidelity
                table_results[kernel_width][model][method]['instance_fidelity'] = avg_instance_fidelity

                print(f'Instance Fidelity: {avg_instance_fidelity}')

        ax[m].plot(kernel_widths, lime_local_fidelity, label='LIME', color=lime_color, linestyle='--')
        ax[m].plot(kernel_widths, chilli_local_fidelity, label='CHILLI', color=chilli_color, linestyle='--')
        ax[m].set_title(f'{model} - {cleaned_dataset_names[args.dataset]}')
        ax[m].set_xlabel(r'$\sigma$')
        ax[m].set_xticks(kernel_widths[::2])
#        ax[0][m].set_xticks([])

        ax[m].plot(kernel_widths, lime_instance_fidelity, label='LIME', color=lime_color)
        ax[m].plot(kernel_widths, chilli_instance_fidelity, label='CHILLI', color=chilli_color)

        if m == 0:
            ax[m].set_ylabel('MAE')
#            ax[1][m].set_ylabel('Instance MAE')

    print_tex_table(table_results)
    fig.legend(['LIME - Local', 'CHILLI - Local', 'LIME - Instance', 'CHILLI - Instance'], loc='upper center', ncols=2, bbox_to_anchor=(0.5, 1.30))
    fig.savefig(f'Figures/{args.dataset}_fidelities.pdf', bbox_inches='tight')

def masala_fidelities():
    for

def print_tex_table(table_results):
    for kw in table_results.keys():
        row = f'& {kw} & '
        for model in table_results[kw].keys():
            for method in table_results[kw][model].keys():
                row += f'{table_results[kw][model][method]['local_fidelity']:.3f} & {table_results[kw][model][method]['instance_fidelity']:.3f} & '
        print(row[:-2] + r'\\')


def exp_variance_box_plot():
    fig2, ax2 = plt.subplots(1, len(models), figsize=(9, 2.5))
    for m, model in enumerate(models):
        fig, ax = plt.subplots(2, len(kernel_widths), figsize=(20, 10 ))
        max_val = -np.inf
        variances = [[], []]
        for i, kernel_width in enumerate(kernel_widths):
            print(f'\n --------- Kernel Width: {kernel_width} --------')
            for method in ['LIME', 'CHILLI']:
                with open(f'saved/results/{args.dataset}/{method}_{args.dataset}_{model}#_{args.experiment_num}_{args.num_instances}_kw={kernel_width}.pck', 'rb') as file:
                    result = pck.load(file)
                print(f'\n --------- {method} Results --------')
                plot_num = ['LIME', 'CHILLI'].index(method)
                sorted_explanations = []
                for n in range(len(result['model_perturbation_predictions'])):
                    sorted_exp = exp_sorter(result['explanations'][n].as_list(), features[args.dataset])
                    sorted_exp = [i/np.sum(np.abs(sorted_exp)) for i in sorted_exp]
                    if max(abs(max(sorted_exp)), abs(min(sorted_exp))) > max_val:
                        max_val = max(abs(max(sorted_exp)), abs(min(sorted_exp)))
                    sorted_explanations.append(sorted_exp)
                sorted_explanations = np.array(sorted_explanations)

                feature_data = [sorted_explanations[:, i] for i in range(sorted_explanations.shape[1])]
                ax[plot_num][i].boxplot(feature_data, labels=features[args.dataset], vert=False, showfliers=False, notch=True, medianprops={'color': 'orange'})
                ax[plot_num][i].set_title(f'{method} - '+r'$\sigma$'+f'={kernel_width}')
                if i != 0:
                    ax[plot_num][i].set_yticklabels([])
                avg_variance = np.mean([np.var(sorted_explanations[:, f]) for f in range(len(features[args.dataset]))])
                print(f'Average Variance: {avg_variance}')
                variances[plot_num].append(avg_variance)

#    max_val = max_val
        for axes in fig.get_axes():
            axes.set_xlim(-max_val, max_val)
            axes.set_xlim(-max_val, max_val)
        fig.savefig(f'Figures/{args.dataset}_{model}_variance_box_plots.pdf', bbox_inches='tight')
        ax2[m].plot(kernel_widths, variances[0], label='LIME', color=lime_color)
        ax2[m].plot(kernel_widths, variances[1], label='CHILLI', color=chilli_color)
        if m == 0:
            ax2[m].set_ylabel('Average Variance')
        ax2[m].set_xlabel('Kernel Width')
        ax2[m].set_title(f'{cleaned_dataset_names[args.dataset]} - {model}')
    fig2.legend(['LIME', 'CHILLI'], loc='upper center', ncols=2, bbox_to_anchor=(0.5, 1.15))
    fig2.savefig(f'Figures/{args.dataset}_variances.pdf')

def predict_with_top_k_features(explanation, local_model, instance_data, k):
    explanation = np.array(explanation)
    sorted_indices = np.argsort([abs(e) for e in explanation])[::-1]
#    print(sorted_indices, explanation)
    top_k_indices = sorted_indices[:k+1]
    reduced_instance = np.zeros(len(features[args.dataset]))
    for i in top_k_indices:
        reduced_instance[i] = instance_data[i]
    return local_model.predict([reduced_instance])[0]


def varying_interpretability(kernel_widths):

    for m, model in enumerate(models):
        max_if = -np.inf
        min_if = np.inf
        max_lf = -np.inf
        min_lf = np.inf

        fig, axes = plt.subplots(1, len(kernel_widths), figsize=(2*len(kernel_widths), 2), sharey=True)
        for i, kernel_width in enumerate(kernel_widths):
            result_dict = {'LIME': {'local_fidelity': {k: [] for k in range(len(features[args.dataset]))},
                                    'instance_fidelity': {k: [] for k in range(len(features[args.dataset]))}},
                            'CHILLI': {'local_fidelity': {k: [] for k in range(len(features[args.dataset]))},
                                    'instance_fidelity': {k: [] for k in range(len(features[args.dataset]))}}}
            chilli_local_fidelity = {k: [] for k in range(len(features[args.dataset]))}
            chilli_instance_fidelity = {k: [] for k in range(len(features[args.dataset]))}
            lime_local_fidelity = {k: [] for k in range(len(features[args.dataset]))}
            lime_instance_fidelity = {k: [] for k in range(len(features[args.dataset]))}
            for method in ['LIME', 'CHILLI']:
                with open(f'saved/results/{args.dataset}/{method}_{args.dataset}_{model}#_{args.experiment_num}_{args.num_instances}_kw={kernel_width}.pck', 'rb') as file:
                    result = pck.load(file)
                print(f'\n --------- {method} Results --------')
                sorted_explanations = []
                for n in range(len(result['model_perturbation_predictions'])):
                    sorted_exp = exp_sorter(result['explanations'][n].as_list(), features[args.dataset])
                    instance_data = result['instance_data'][n]
                    local_model = result['explanations'][n].local_model
                    # Sort instance data and exp based on l)
                    for k in range(len(features[args.dataset])):
                        top_k_instance_prediction = predict_with_top_k_features(sorted_exp, local_model, instance_data, k)
                        result_dict[method]['instance_fidelity'][k].append(calculate_instance_fidelity(top_k_instance_prediction, result['model_instance_prediction'][n]))

#                        result['model_perturbation_predictions'][n] = result['model_perturbation_predictions'][n][:200]
                        top_k_perturbation_predictions = []
                        for p in range(len(result['model_perturbation_predictions'][n])):
                            top_k_perturbation_predictions.append(predict_with_top_k_features(sorted_exp, local_model, result['perturbations'][n][p], k))
                        result_dict[method]['local_fidelity'][k].append(calculate_local_fidelity(top_k_perturbation_predictions, result['model_perturbation_predictions'][n]))

            for method, color in zip(['LIME', 'CHILLI'], [lime_color, chilli_color]):
                local_fidelities = [np.mean(result_dict[method]['local_fidelity'][k]) for k in range(len(features[args.dataset]))]
                instance_fidelities = [np.mean(result_dict[method]['instance_fidelity'][k]) for k in range(len(features[args.dataset]))]
                axes[i].plot(local_fidelities, label=method, color=color, linestyle='--')
                axes[i].plot(instance_fidelities, label=method, color=color)

                if max(local_fidelities) > max_lf:
                    max_lf = max(local_fidelities)
                if min(local_fidelities) < min_lf:
                    min_lf = min(local_fidelities)
                if max(instance_fidelities) > max_if:
                    max_if = max(instance_fidelities)
                if min(instance_fidelities) < min_if:
                    min_if = min(instance_fidelities)


            axes[i].set_title(r'$\sigma$ = ' + str(kernel_width))
            if i == 0:
                axes[i].set_ylabel('MAE')
#            else:
#                axes[i].set_yticklabels([])
            axes[i].set_xticks(range(len(features[args.dataset]))[::2])
#        axes[0][i].set_xlabel('Number of Features')
            axes[i].set_xlabel('# of Features')
#    axes[0][i].legend()
#        for i in range(len(kernel_widths)):
#            axes[i].set_ylim(min_if, max_if)



#    axes[1][i].legend()
        fig.legend(['LIME - Local', 'CHILLI - Local', 'LIME - Instance', 'CHILLI - Instance'], loc='upper center', ncols=2, bbox_to_anchor=(0.5, 1.30))
        fig.savefig(f'Figures/{args.dataset}_{model}_interpretability.png', bbox_inches='tight', dpi=300)



for dataset in ['MIDAS', 'housing', 'webTRIS']:
    args.dataset = dataset
    if args.dataset == 'MIDAS':
        models = ['GBR', 'SVR', 'RNN']
    elif args.dataset == 'housing':
        models = ['GBR', 'SVR', 'RF']
    elif args.dataset == 'PHM08':
        models = ['GBR', 'SVR', 'RF']
    elif args.dataset == 'webTRIS':
        models = ['SVR', 'RF', 'GBR']
    fidelities()
#    exp_variance_box_plot()
#    varying_interpretability(kernel_widths[:5])





