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
    "Ackley": 0.0,  
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

def dynamic_kernel_selection(kernel_performances, t):
    dp = [[0 for _ in range(len(kernel_performances[0]))] for _ in range(t)]
    
    for j in range(len(kernel_performances[0])):
        dp[0][j] = kernel_performances[0][j]
    
    for i in range(1, t):
        for j in range(len(kernel_performances[0])):
            dp[i][j] = max(dp[i-1][k] + kernel_performances[i][j] for k in range(len(kernel_performances[0])))
    
    best_kernels = []
    current_best = max(dp[t-1])
    for i in range(t-1, -1, -1):
        for j in range(len(kernel_performances[0])):
            if dp[i][j] == current_best:
                best_kernels.append(j)
                if i > 0:
                    current_best -= kernel_performances[i][j]
                break
    
    return list(reversed(best_kernels))

def adaptive_learning_rate(past_rewards, t):
    dp = [0] * (t + 1)
    dp[0] = 0.1  # Initial learning rate
    
    for i in range(1, t + 1):
        if i == 1:
            dp[i] = dp[i-1]
        else:
            if past_rewards[i-1] > past_rewards[i-2]:
                dp[i] = min(dp[i-1] * 1.1, 1.0)  # Increase learning rate
            else:
                dp[i] = max(dp[i-1] * 0.9, 0.01)  # Decrease learning rate
    
    return dp[t]

def get_next_points(train_x, train_y, best_init_y, bounds, eta, n_points=1, gains=None, kernel_types=['RBF', 'Matern', 'RQ'], acq_func_types=['EI', 'UCB', 'PI'], kernel_performances=None, t=0):
    models = []
    mlls = []
    for kernel in kernel_types:
        model, mll = fit_model(train_x, train_y, kernel)
        models.append(model)
        mlls.append(mll)

    if kernel_performances is not None and t > 0:
        best_kernels = dynamic_kernel_selection(kernel_performances, t)
        selected_model_index = best_kernels[-1]
    else:
        selected_model_index = np.random.choice(len(kernel_types))
    
    selected_model = models[selected_model_index]

    with gpytorch.settings.cholesky_jitter(1e-1):  # Adding jitter
        EI = ExpectedImprovement(model=selected_model, best_f=best_init_y)
        UCB = UpperConfidenceBound(model=selected_model, beta=0.1)
        PI = ProbabilityOfImprovement(model=selected_model, best_f=best_init_y)

    acq_funcs = {'EI': EI, 'UCB': UCB, 'PI': PI}
    acquisition_functions = [acq_funcs[acq] for acq in acq_func_types]
    
    candidates_list = []
    for acq_function in acquisition_functions:
        try:
            with gpytorch.settings.cholesky_jitter(1e-1):
                candidates, acq_value = optimize_acqf(
                    acq_function=acq_function, 
                    bounds=bounds, 
                    q=n_points, 
                    num_restarts=20, 
                    raw_samples=512, 
                    options={"batch_limit": 5, "maxiter": 200}
                )
        except Exception as e:
            print(f"Error optimizing acquisition function {acq_function}: {e}")
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
    bounds = initial_data["bounds"]
    train_x, train_y, best_init_y = initial_data["train_x"], initial_data["train_y"], initial_data["best_init_y"]
    
    gains = np.zeros(len(acq_func_types))
    past_rewards = []

    best_observed_values = []
    chosen_acq_functions = []
    selected_models = []
    gap_metrics = []

    best_init_y = train_y.max().item()
    kernel_performances = []

    for t in range(n_iterations):
        print(f"Number of iterations done: {t}")
        
        eta = adaptive_learning_rate(past_rewards, t) if past_rewards else 0.1
        
        new_candidates, chosen_acq_index, selected_model_index, selected_model = get_next_points(
            train_x, train_y, best_init_y, bounds, eta, 1, gains, kernel_types, acq_func_types, 
            kernel_performances, t
        )
        new_results = target_function(new_candidates).unsqueeze(-1)

        train_x, train_y = update_data(train_x, train_y, new_candidates, new_results)

        best_init_y = train_y.max().item()
        best_observed_values.append(best_init_y)
        chosen_acq_functions.append(chosen_acq_index)
        selected_models.append(selected_model_index)

        g_i = gap_metric(initial_data["best_init_y"], best_init_y, initial_data["true_maximum"])
        gap_metrics.append(g_i)
        
        print(f"Best point performs this way: {best_init_y}")
        print(f"Gap metric G_i: {g_i}")

        posterior_mean = selected_model.posterior(new_candidates).mean
        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward
        past_rewards.append(reward)

        # Refit models and compute log marginal likelihood for kernel performance
        current_kernel_performance = []
        for kernel in kernel_types:
            model, mll = fit_model(train_x, train_y, kernel)
            current_kernel_performance.append(mll(model(train_x), train_y).item())
        kernel_performances.append(current_kernel_performance)

    return best_observed_values, chosen_acq_functions, selected_models, initial_data["true_maximum"], gap_metrics

def plot_results(results, titles, test_function_name, true_maximum):
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    for i, (best_observed_values, chosen_acq_functions, selected_models, true_maximum, gap_metrics) in enumerate(results):
        ax2 = axs[i].twinx()
        ax2.plot(gap_metrics, marker='x', linestyle='-', color='r', label='Gap Metric')
        ax2.set_ylabel("Gap Metric G_i")
        ax2.legend(loc='lower right')
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"test_res.png")

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

results1 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI', 'UCB', 'PI'], initial_data)

results2 = run_experiment(n_iterations, ['RBF', 'Matern', 'RQ'], ['EI'], initial_data)

results3 = run_experiment(n_iterations, ['Matern'], ['EI', 'UCB', 'PI'], initial_data)

results = [results1, results2, results3]
titles = ["All Models and All Acquisition Functions", "All Models and Only EI", "Only Matern Model and All Acquisition Functions"]

plot_results(results, titles, "Hartmann", true_maxima['Hartmann'])
