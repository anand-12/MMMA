from sklearn.linear_model import LinearRegression
import botorch
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.test_functions import Hartmann, Ackley, Rosenbrock, Levy, Powell
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
import warnings

warnings.filterwarnings("ignore")

hart6 = Hartmann(dim=6)
ackley2 = Ackley(dim=2)

true_maxima = {
    "Hartmann": 3.32237,  
    "Ackley": 0.0  
}

class RFF:
    def __init__(self, input_dim, num_features, sigma=1.0):
        self.input_dim = input_dim
        self.num_features = num_features
        self.sigma = sigma
        self.W = np.random.normal(scale=1.0/self.sigma, size=(self.input_dim, self.num_features))
        self.b = np.random.uniform(0, 2 * np.pi, size=self.num_features)

    def transform(self, X):
        norm = 1.0 / np.sqrt(self.num_features)
        projection = X @ self.W + self.b
        return norm * np.sqrt(2.0) * np.concatenate([np.cos(projection), np.sin(projection)], axis=-1)
    
class RFFPredictionModel:
    def __init__(self, model, rff, pred_std):
        self.model = model
        self.rff = rff
        self.pred_std = pred_std
        self.num_outputs = 1

    def posterior(self, X):
        X_rff = torch.tensor(self.rff.transform(X.numpy()), dtype=torch.float32)
        mean = self.model.predict(X_rff)
        return gpytorch.distributions.MultivariateNormal(
            torch.tensor(mean, dtype=torch.float32),
            covariance_matrix=torch.eye(len(mean)) * self.pred_std**2
        )

    def forward(self, X):
        return self.posterior(X)

def gap_metric(f_start, f_current, f_star):
    return np.abs((f_start - f_current) / (f_start - f_star))

def target_function(individuals):
    result = []
    for x in individuals:
        result.append(-1.0 * hart6(x))
    return torch.tensor(result, dtype=torch.double)

def generate_initial_data(n, n_dim):
    train_x = torch.rand(n, n_dim, dtype=torch.double)  
    exact_obj = target_function(train_x).unsqueeze(-1)
    best_observed_value = exact_obj.max().item()
    return train_x, exact_obj, best_observed_value

def fit_model(train_x, train_y, kernel_type):
    if kernel_type == 'RBF':
        # Use RFF for RBF kernel
        num_features = 100  # You can adjust this
        rff = RFF(train_x.shape[1], num_features)
        X_rff = torch.tensor(rff.transform(train_x.numpy()), dtype=torch.float32)
        model = LinearRegression()
        model.fit(X_rff, train_y.numpy())
        return model, rff
    elif kernel_type == 'Matern':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
    elif kernel_type == 'RQ':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    class CustomGP(SingleTaskGP):
        def __init__(self, train_x, train_y):
            super().__init__(train_x, train_y)
            self.covar_module = covar_module

    model = CustomGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, mll

def calculate_weights(models, mlls, train_x, train_y):
    log_likelihoods = []
    for i, model in enumerate(models):
        if isinstance(model, LinearRegression):
            # For RFF
            rff = mlls[i]
            X_rff = torch.tensor(rff.transform(train_x.numpy()), dtype=torch.float32)
            pred = model.predict(X_rff)
            mse = np.mean((pred - train_y.numpy())**2)
            log_likelihood = -mse  # Use negative MSE as a proxy for log-likelihood
        else:
            # For GP models
            log_likelihood = mlls[i](models[i](train_x), train_y).sum().item()
        log_likelihoods.append(log_likelihood)
    
    log_likelihoods = np.array(log_likelihoods)
    max_log_likelihood = np.max(log_likelihoods)
    log_likelihoods -= max_log_likelihood
    weights = np.exp(log_likelihoods) / np.sum(np.exp(log_likelihoods))
    return weights

def select_model(weights):
    return np.random.choice(len(weights), p=weights)

def get_next_points(train_x, train_y, best_init_y, bounds, eta, n_points=1, gains=None, kernel_types=['RBF', 'Matern', 'RQ'], acq_func_types=['EI', 'UCB', 'PI']):
    models = []
    mlls = []
    for kernel in kernel_types:
        model, mll = fit_model(train_x, train_y, kernel)
        models.append(model)
        mlls.append(mll)

    weights = calculate_weights(models, mlls, train_x, train_y)
    selected_model_index = select_model(weights)
    selected_model = models[selected_model_index]
    selected_mll = mlls[selected_model_index]

    if isinstance(selected_model, LinearRegression):
        # For RFF
        rff = selected_mll
        X_rff = torch.tensor(rff.transform(train_x.numpy()), dtype=torch.float32)
        pred_mean = selected_model.predict(X_rff)
        pred_std = np.sqrt(np.mean((pred_mean - train_y.numpy())**2))
        
        selected_model = RFFPredictionModel(selected_model, rff, pred_std)

    with gpytorch.settings.cholesky_jitter(1e-1):  # Adding jitter
        EI = ExpectedImprovement(model=selected_model, best_f=best_init_y)
        UCB = UpperConfidenceBound(model=selected_model, beta=0.1)
        PI = ProbabilityOfImprovement(model=selected_model, best_f=best_init_y)

    acq_funcs = {'EI': EI, 'UCB': UCB, 'PI': PI}
    acquisition_functions = [acq_funcs[acq] for acq in acq_func_types]
    
    candidates_list = []
    for acq_function in acquisition_functions:
        with gpytorch.settings.cholesky_jitter(1e-1):
            candidates, acq_value = optimize_acqf(
                acq_function=acq_function, 
                bounds=bounds, 
                q=n_points, 
                num_restarts=10, 
                raw_samples=16, 
                options={"batch_limit": 5, "maxiter": 200}
            )
    candidates_list.append(candidates)

    logits = np.array(gains)
    logits -= np.max(logits)
    exp_logits = np.exp(eta * logits)
    probs = exp_logits / np.sum(exp_logits)
    chosen_acq_index = np.random.choice(len(acquisition_functions), p=probs)

    return candidates_list[chosen_acq_index], chosen_acq_index, selected_model_index, selected_model

def update_data(train_x, train_y, new_x, new_y):
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    return train_x, train_y

def run_experiment(n_iterations, kernel_types, acq_func_types, initial_data):
    # np.random.seed(42)
    bounds = initial_data["bounds"]
    train_x, train_y, best_init_y = initial_data["train_x"], initial_data["train_y"], initial_data["best_init_y"]
    
    gains = np.zeros(len(acq_func_types))
    eta = 0.1  

    best_observed_values = []
    chosen_acq_functions = []
    selected_models = []
    gap_metrics = []

    best_init_y = train_y.max().item()

    for t in range(n_iterations):
        print(f"Number of iterations done: {t}")
        new_candidates, chosen_acq_index, selected_model_index, selected_model = get_next_points(train_x, train_y, best_init_y, bounds, eta, 1, gains, kernel_types, acq_func_types)
        new_results = target_function(new_candidates).unsqueeze(-1)

        train_x, train_y = update_data(train_x, train_y, new_candidates, new_results)

        best_init_y = train_y.max().item()
        best_observed_values.append(best_init_y)
        chosen_acq_functions.append(chosen_acq_index)
        selected_models.append(selected_model_index)

        # Calculate gap metric
        g_i = gap_metric(initial_data["best_init_y"], best_init_y, initial_data["true_maximum"])
        gap_metrics.append(g_i)
        
        print(f"Best point performs this way: {best_init_y}")
        print(f"Gap metric G_i: {g_i}")

        posterior_mean = selected_model.posterior(new_candidates).mean
        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward

    return best_observed_values, chosen_acq_functions, selected_models, initial_data["true_maximum"], gap_metrics

def plot_results(results, titles, test_function_name, true_maximum):
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    for i, (best_observed_values, chosen_acq_functions, selected_models, true_maximum, gap_metrics) in enumerate(results):
        # axs[i].plot(best_observed_values, marker='o', linestyle='-', color='b', label='Best Objective Value')
        # axs[i].axhline(y=true_maximum, color='k', linestyle='--', label='True Maxima')
        # axs[i].set_title(titles[i])
        # axs[i].set_xlabel("Iteration")
        # axs[i].set_ylabel("Best Objective Function Value")
        # axs[i].legend()
        # axs[i].grid(True)

        ax2 = axs[i].twinx()
        ax2.plot(gap_metrics, marker='x', linestyle='-', color='r', label='Gap Metric')
        ax2.set_ylabel("Gap Metric G_i")
        ax2.legend(loc='lower right')
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"test_res.png")
    # plt.show()

n_iterations = 20

ackley_bounds = torch.tensor([[-32.768, -32.768], [32.768, 32.768]], dtype=torch.double)
hart6_bounds = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.double)
init_x, init_y, best_init_y = generate_initial_data(10, n_dim=hart6_bounds.size(1))
initial_data = {
    "train_x": init_x,
    "train_y": init_y,
    "best_init_y": best_init_y,
    "bounds": hart6_bounds,
    "true_maximum": true_maxima['Hartmann']
}

results1 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI', 'UCB', 'PI'], initial_data)

results2 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI'], initial_data)

results3 = run_experiment(n_iterations, ['Matern'], ['EI', 'UCB', 'PI'], initial_data)

results = [results1, results2, results3]
titles = ["All Models and All Acquisition Functions", "All Models and Only EI", "Only Matern Model and All Acquisition Functions"]

plot_results(results, titles, "Hartmann", true_maxima['Hartmann'])

