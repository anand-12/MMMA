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
from rfgp_yq import RfgpModel
import warnings

warnings.filterwarnings("ignore")

hart6 = Hartmann(dim=6)#hartmann is base line

def target_function(individuals):
    result = []
    for x in individuals:
        result.append(x ** 2)
    return torch.tensor(result)

def generate_initial_data(n=10):
    train_x = torch.rand(n, 1, dtype=torch.double)#改成float
    exact_obj = target_function(train_x).unsqueeze(-1)
    best_observed_value = exact_obj.max().item()
    return train_x, exact_obj, best_observed_value

init_x, init_y, best_init_y = generate_initial_data(20)
init_x = torch.tensor(init_x).float()
init_y = torch.tensor(init_y).float().squeeze()
best_init_y = torch.tensor(best_init_y).float().squeeze()
# bounds = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1.]])  
bounds = torch.tensor([[0.], [1.]])
gains = np.zeros(3)
eta = 0.1  
def get_next_points(init_x, init_y, best_init_y, bounds, n_points=1, gains=None):
    # single_model = SingleTaskGP(init_x, init_y)
    # mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    # fit_gpytorch_mll(mll)
    #replace these lines with
    single_model = RfgpModel(1,1,20)
    single_model.fit(init_x, init_y)

    EI = ExpectedImprovement(model=single_model, best_f=best_init_y)
    UCB = UpperConfidenceBound(model=single_model, beta=0.1)
    PI = ProbabilityOfImprovement(model=single_model, best_f=best_init_y)
    
    acquisition_functions = [EI, UCB, PI]
    
    candidates_list = []
    for acq_function in acquisition_functions:
        # with torch.no_grad():
        candidates, _ = optimize_acqf(acq_function=acq_function, bounds=bounds, q=n_points, num_restarts=20, raw_samples=512)
        candidates_list.append(candidates)

    logits = np.array(gains)
    logits -= np.max(logits)
    exp_logits = np.exp(eta * logits)
    probs = exp_logits / np.sum(exp_logits)
    chosen_acq_index = np.random.choice(len(acquisition_functions), p=probs)

    return candidates_list[chosen_acq_index], chosen_acq_index, single_model

best_observed_values = []
chosen_acq_functions = []

def plot_best_observed(best_observed_values):
    plt.figure(figsize=(10, 6))
    plt.plot(best_observed_values, marker='o', linestyle='-', color='b')
    plt.title("Best Objective Function Value at Each Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Function Value")
    plt.grid(True)
    plt.show()

def plot_chosen_acq_functions(chosen_acq_functions):
    plt.figure(figsize=(10, 6))
    plt.plot(chosen_acq_functions, marker='o', linestyle='-', color='r')
    plt.title("Acquisition Function Selected at Each Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Acquisition Function Index")
    plt.yticks([0, 1, 2], ["EI", "UCB", "PI"])
    plt.grid(True)
    plt.show()

n_iterations = 10
init_x, init_y, best_init_y = generate_initial_data(20)

for i in range(n_iterations):
    print(f"Number of iterations done: {i}")
    new_candidates, chosen_acq_index, single_model = get_next_points(init_x, init_y, best_init_y, bounds, 1, gains)
    new_results = target_function(new_candidates).unsqueeze(-1)

    print(f"New candidates are: {new_candidates}")
    init_x = torch.cat([init_x, new_candidates])
    init_y = torch.cat([init_y, new_results])

    best_init_y = init_y.max().item()
    best_observed_values.append(best_init_y)
    chosen_acq_functions.append(chosen_acq_index)
    print(f"Best point performs this way: {best_init_y}")

    posterior_mean = single_model.posterior(new_candidates)
    reward = posterior_mean.mean().item()
    gains[chosen_acq_index] += reward

print(f"Best observed result: {best_init_y}")
best_candidate = init_x[((init_y == best_init_y).nonzero(as_tuple=True)[0])][0][0]
print(f"Best location of observed result: {best_candidate}")

plot_best_observed(best_observed_values)

plot_chosen_acq_functions(chosen_acq_functions)

