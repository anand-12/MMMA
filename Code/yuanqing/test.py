import botorch
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.test_functions import Hartmann, Ackley
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
from rfgp_yq import RfgpModel
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
    return torch.tensor(result, dtype=torch.float)

def generate_initial_data(n, n_dim):
    train_x = torch.rand(n, n_dim, dtype=torch.float)  
    exact_obj = target_function(train_x).unsqueeze(-1)
    best_observed_value = exact_obj.max().item()
    return train_x, exact_obj, best_observed_value

def fit_model(train_x, train_y, kernel_type):
    model = RfgpModel(in_dim=train_x.shape[1], out_dim=1, J=20)
    model.fit(train_x, train_y)
    return model, None  

def calculate_weights(models, train_x, train_y):
    return np.ones(len(models)) / len(models)

def select_model(weights):
    return np.random.choice(len(weights), p=weights)

def get_next_points(train_x, train_y, best_init_y, bounds, eta, n_points=1, gains=None, kernel_types=['RBF', 'Matern', 'RQ'], acq_func_types=['EI', 'UCB', 'PI']):
    models = []
    for _ in kernel_types:
        model, _ = fit_model(train_x, train_y, None)
        models.append(model)

    weights = calculate_weights(models, train_x, train_y)
    selected_model_index = select_model(weights)
    selected_model = models[selected_model_index]

    EI = ExpectedImprovement(model=selected_model, best_f=best_init_y)
    UCB = UpperConfidenceBound(model=selected_model, beta=0.1)
    PI = ProbabilityOfImprovement(model=selected_model, best_f=best_init_y)

    acq_funcs = {'EI': EI, 'UCB': UCB, 'PI': PI}
    acquisition_functions = [acq_funcs[acq] for acq in acq_func_types]
    
    candidates_list = []
    for acq_function in acquisition_functions:
        # try:
        candidates, acq_value = optimize_acqf(
            acq_function=acq_function, 
            bounds=bounds, 
            q=n_points, 
            num_restarts=10, 
            raw_samples=16, 
            options={"batch_limit": 5, "maxiter": 200}
        )
        # except Exception as e:
            # print(f"Error optimizing acquisition function {acq_function}: {e}")
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
        ax2 = axs[i].twinx()
        ax2.plot(gap_metrics, marker='x', linestyle='-', color='r', label='Gap Metric')
        ax2.set_ylabel("Gap Metric G_i")
        ax2.legend(loc='lower right')
        ax2.grid(True)

    plt.tight_layout()
    # plt.savefig(f"test_res.png")
    plt.show()

n_iterations = 20

hart6_bounds = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float)
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