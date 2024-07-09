import botorch
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.test_functions import Hartmann
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
import warnings

warnings.filterwarnings("ignore")

hart6 = Hartmann(dim=6)

true_maxima = {
    "Hartmann": 3.32237,
}

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
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
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

def identify_promising_regions(model, bounds, n_regions=3):
    # This is a simple implementation. You might want to use a more sophisticated method.
    n_samples = 1000
    x = torch.rand(n_samples, bounds.size(1), dtype=bounds.dtype) * (bounds[1] - bounds[0]) + bounds[0]
    with torch.no_grad():
        y = model.posterior(x).mean
    _, indices = torch.topk(y.squeeze(), n_regions)
    regions = [x[i] for i in indices]
    return regions

def get_next_points_hierarchical(high_level_model, low_level_models, train_x, train_y, best_f, bounds, eta, gains, acq_func_types):
    regions = identify_promising_regions(high_level_model, bounds)
    
    candidates_list = []
    gains_list = []
    
    for i, region in enumerate(regions):
        for j, model in enumerate(low_level_models):
            for k, acq_type in enumerate(acq_func_types):
                if acq_type == 'EI':
                    acq_function = ExpectedImprovement(model=model, best_f=best_f)
                elif acq_type == 'UCB':
                    acq_function = UpperConfidenceBound(model=model, beta=0.1)
                elif acq_type == 'PI':
                    acq_function = ProbabilityOfImprovement(model=model, best_f=best_f)
                
                try:
                    with gpytorch.settings.cholesky_jitter(1e-1):
                        candidates, _ = optimize_acqf(
                            acq_function=acq_function, 
                            bounds=bounds, 
                            q=1, 
                            num_restarts=5, 
                            raw_samples=128, 
                            options={"batch_limit": 5, "maxiter": 200}
                        )
                    candidates_list.append(candidates)
                    gains_list.append(gains[j * len(acq_func_types) + k])
                except Exception as e:
                    print(f"Error optimizing acquisition function: {e}")

    # Also include a candidate from the high-level model
    high_level_acq = ExpectedImprovement(model=high_level_model, best_f=best_f)
    high_level_candidate, _ = optimize_acqf(
        acq_function=high_level_acq, 
        bounds=bounds, 
        q=1, 
        num_restarts=5, 
        raw_samples=128, 
        options={"batch_limit": 5, "maxiter": 200}
    )
    candidates_list.append(high_level_candidate)
    gains_list.append(0)  # Initialize gain for high-level model

    # Select candidate based on gains
    logits = np.array(gains_list)
    logits -= np.max(logits)
    exp_logits = np.exp(eta * logits)
    probs = exp_logits / np.sum(exp_logits)
    chosen_index = np.random.choice(len(candidates_list), p=probs)

    return candidates_list[chosen_index], chosen_index

def update_models(high_level_model, low_level_models, new_x, new_y):
    # Ensure new_y is 2D
    if new_y.dim() == 1:
        new_y = new_y.unsqueeze(-1)

    # Update high-level model
    train_x = high_level_model.train_inputs[0]
    train_y = high_level_model.train_targets.unsqueeze(-1)  # Ensure train_y is 2D
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    high_level_model.set_train_data(train_x, train_y.squeeze(-1), strict=False)

    # Update low-level models
    for model in low_level_models:
        train_x = model.train_inputs[0]
        train_y = model.train_targets.unsqueeze(-1)  # Ensure train_y is 2D
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        model.set_train_data(train_x, train_y.squeeze(-1), strict=False)

    return high_level_model, low_level_models

def run_hierarchical_experiment(n_iterations, kernel_types, acq_func_types, initial_data):
    bounds = initial_data["bounds"]
    train_x, train_y, best_f = initial_data["train_x"], initial_data["train_y"], initial_data["best_init_y"]
    
    # Initialize high-level model
    high_level_model, _ = fit_model(train_x, train_y.unsqueeze(-1), 'RBF')
    
    # Initialize low-level models
    low_level_models = []
    for kernel in kernel_types:
        model, _ = fit_model(train_x, train_y.unsqueeze(-1), kernel)
        low_level_models.append(model)
    
    gains = np.zeros(len(kernel_types) * len(acq_func_types))
    eta = 0.1

    best_observed_values = []
    gap_metrics = []

    for t in range(n_iterations):
        print(f"Iteration: {t}")
        new_candidates, chosen_index = get_next_points_hierarchical(
            high_level_model, low_level_models, train_x, train_y.unsqueeze(-1), best_f, bounds, eta, gains, acq_func_types
        )
        new_results = target_function(new_candidates).unsqueeze(-1)  # Ensure new_results is 2D

        high_level_model, low_level_models = update_models(high_level_model, low_level_models, new_candidates, new_results)

        best_f = max(best_f, new_results.max().item())
        best_observed_values.append(best_f)

        g_i = gap_metric(initial_data["best_init_y"], best_f, initial_data["true_maximum"])
        gap_metrics.append(g_i)
        
        print(f"Best value: {best_f}")
        print(f"Gap metric G_i: {g_i}")

        # Update gains
        if chosen_index < len(gains):
            gains[chosen_index] += new_results.item()

    return best_observed_values, gap_metrics

n_iterations = 50

hart6_bounds = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.double)
init_x, init_y, best_init_y = generate_initial_data(10, n_dim=hart6_bounds.size(1))
initial_data = {
    "train_x": init_x,
    "train_y": init_y,
    "best_init_y": best_init_y,
    "bounds": hart6_bounds,
    "true_maximum": true_maxima['Hartmann']
}

results_hierarchical = run_hierarchical_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI', 'UCB', 'PI'], initial_data)

# Modify plot_results function to handle the new format of results
def plot_results(results, titles, test_function_name, true_maximum):
    fig, ax = plt.subplots(figsize=(10, 6))

    best_observed_values, gap_metrics = results

    ax.plot(best_observed_values, marker='o', linestyle='-', color='b', label='Best Objective Value')
    ax.axhline(y=true_maximum, color='k', linestyle='--', label='True Maximum')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Objective Function Value")
    ax.legend(loc='upper left')
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(gap_metrics, marker='x', linestyle='-', color='r', label='Gap Metric')
    ax2.set_ylabel("Gap Metric G_i")
    ax2.legend(loc='lower right')

    plt.title(f"Hierarchical GP-Hedge on {test_function_name}")
    plt.tight_layout()
    plt.savefig(f"hierarchical_gp_hedge_results.png")

plot_results(results_hierarchical, "Hierarchical GP-Hedge", "Hartmann", true_maxima['Hartmann'])