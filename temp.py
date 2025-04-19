import pickle as pck



for kw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    for dataset in ['MIDAS', 'housing', 'webTRIS']:
        if dataset == 'MIDAS':
            models = ['GBR', 'SVR', 'RNN']
        elif dataset == 'housing':
            models = ['GBR', 'SVR', 'RF']
        elif dataset == 'webTRIS':
            models = ['SVR', 'RF', 'GBR']
        for model in models:
            with open(f'saved/results/{dataset}/{dataset}_{model}#_1_30_kw={kw}.pck', 'rb') as f:
                results = pck.load(f)
                lime_results = results[0]
                chilli_results = results[1]

            with open(f'saved/results/{dataset}/sep_results/LIME_{dataset}_{model}#_1_30_kw={kw}.pck', 'wb') as f:
                pck.dump(lime_results, f)
            with open(f'saved/results/{dataset}/sep_results/CHILLI_{dataset}_{model}#_1_30_kw={kw}.pck', 'wb') as f:
                pck.dump(chilli_results, f)
