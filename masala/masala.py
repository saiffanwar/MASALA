from .llc_explainer import LLCExplanation
from .llc_ensemble_generator import LLCGenerator



class MASALA:

    def __init__(self, model, model_type, x_test, y_test, y_pred, dataset, features, discrete_features, sparsity_threshold, coverage_threshold, starting_k, neighbourhood_threshold, num_workers=1):
        self.model = model
        self.model_type = model_type
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.dataset = dataset
        self.features = features
        self.discrete_features = discrete_features
        self.sparsity_threshold = sparsity_threshold
        self.coverage_threshold = coverage_threshold
        self.starting_k = starting_k
        self.neighbourhood_threshold = neighbourhood_threshold
        self.feature_ensemble = None
        self.explanation_generator = None
        self.experiment_id = 1
        self.num_workers = num_workers

        self.run_clustering(preload_clustering=True)


    def run_clustering(self, preload_clustering):
        self.clustering_generator = LLCGenerator(model=self.model, model_type=self.model_type, x_test=self.x_test, y_pred=self.y_pred, features=self.features, discrete_features=self.discrete_features, dataset=self.dataset, sparsity_threshold=self.sparsity_threshold, coverage_threshold=self.coverage_threshold, starting_k=self.starting_k, neighbourhood_threshold=self.neighbourhood_threshold, preload_clustering=preload_clustering, experiment_id=self.experiment_id, num_workers=self.num_workers)


    def explain_instance(self, instance, plotting=True):
        if self.explanation_generator is None:
            self.explanation_generator = LLCExplanation(model=self.model, model_type=self.model_type, x_test=self.x_test, y_pred=self.y_pred, dataset=self.dataset, features=self.features, discrete_features=self.discrete_features, sparsity_threshold=self.sparsity_threshold, coverage_threshold=self.coverage_threshold, starting_k=self.starting_k, neighbourhood_threshold=self.neighbourhood_threshold)
        explanation, perturbation_error = self.explanation_generator.generate_explanation(self.x_test[instance], instance)
        if plotting:
            self.explanation_generator.interactive_exp_plot(explanation)

        return explanation, perturbation_error





