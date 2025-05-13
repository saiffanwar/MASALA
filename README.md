# MASALA and CHILLI: XAI for Time-Series models

This repository contains implementation of the MASALA and CHILLI methods for explaining temporal models trained on numerical tabular data. The implementations follow the forms described in the following papers:

***CHILLI: A data context-aware perturbation method for XAI*** https://arxiv.org/pdf/2407.07521

***MASALA: Model-Agnostic Surrogate Explanations by Locality Adaptation*** https://arxiv.org/pdf/2408.10085



### Model Implementations

**The repository contains implementation for 4 model types:**

- Recurrent Neural Network
- XGBoost
- Support Vector Regressor
- Random Forest Regressor

**Each of these models can be applied to the 3 included datasets:**

- MIDAS - Weather forecasting at Heathrow Airport
- WebTRis - Traffic forecasting on the M6 motorway
- California Housing - Housing price prediction

### To train a model and then explain it you can run the following:

`python chilli_runner.py `

The following options are included for the methods:

`  -h, --help            show this help message and exit`

 ` -d DATASET, --dataset DATASET    Which dataset to work on` 

`--model_type MODEL_TYPE  Which model to use`

`-l LOAD_MODEL, --load_model LOAD_MODEL  Load the model from file`

 ` -m MODE, --mode MODE  Whether to generate ensembles or explanations.` 

`  -e EXP_MODE, --exp_mode EXP_MODE       Which instances to generate explanations for.`

`--sparsity SPARSITY   The sparsity threshold to use for the LLC explainer.`

` --coverage COVERAGE   The coverage threshold to use for the LLC explainer.`

`--starting_k STARTING_K The number of neighbours to use for the LLC explainer.`

`--neighbourhood NEIGHBOURHOOD  The neighbourhood threshold to use for the LLC explainer.
`

`-p PRIMARY_INSTANCE, --primary_instance PRIMARY_INSTANCE The instance to generate explanations for`

` -n NUM_INSTANCES, --num_instances NUM_INSTANCES  The number of instances to generate explanations for `

`--c_id C_ID           Clustering ID`

`--e_id E_ID           Experiment ID`  

`--lime_exp LIME_EXP   Whether to generate LIME explanations `

`--chilli_exp CHILLI_EXP  Whether to generate CHILLI explanations`

`--llc_exp LLC_EXP     Whether to generate LLC explanations `

`--kernel_width KERNEL_WIDTH        The kernel width to use for the explanations `

`--plots, --no-plots --results`

`--no-results`



### Visualisation App

You can use the included [Dash](https://dash.plotly.com/) App that allows visualisation of explanations, clusterings and test data.

![App Clsutering](/Users/saifanwar/PhD/MASALA/Figures/App Clustering.png)

![App Explanation.png](/Users/saifanwar/PhD/MASALA/Figures/App Explanation.png)

