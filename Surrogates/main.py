import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from skopt.learning import GaussianProcessRegressor

from Surrogates.connector import runModelGetLastPeriod
from Surrogates.functions import *
from Surrogates.islands import evaluate_islands_on_set
from Surrogates.samplers import get_sobol_samples, get_unirand_samples

""" Default Algorithm Tuning Constants """
_N_EVALS = 10
_N_SPLITS = 5

budget = 500

# Set out-of-sample test and montecarlo sizes
test_size = 100
monteCarlos = 100

# Get an on out-of-sample test set that does not have combinations from the
# batch or iterative experiments
final_test_size = (test_size * monteCarlos)

# Set the ABM parameters and support
islands_exploration_range = np.array([
    (0.0, 10),  # rho
    (0.8, 2.0),  # alpha
    (0.0, 1.0),  # phi
    (0.0, 1.0),  # pi
    (0.0, 1.0)])  # eps

gol_exp_range = np.array([
    (0, 500)
])

GOLInputs = {
    "gridSize": (0, 500)
}

param_dims = islands_exploration_range.shape[0]  # returns number of params to explore

load_data = False

if load_data:  # This is only for the budget = 500 setting
    evaluated_set_X_batch = pd.read_csv('Surrogates/InputData/X.csv', index_col=0).values
    evaluated_set_y_batch = pd.read_csv('Surrogates/InputData/y.csv', index_col=0).values
    oos_set = pd.read_csv('Surrogates/InputData/X_oos.csv', index_col=0).values
    y_test = pd.read_csv('Surrogates/InputData/y_oos.csv', index_col=0).values
else:
    start = time.time()
    # Generate Sobol samples for training set
    n_dimensions = islands_exploration_range.shape[0]
    evaluated_set_X_batch = get_sobol_samples(n_dimensions, budget, islands_exploration_range)
    evaluated_set_y_batch = evaluate_islands_on_set(evaluated_set_X_batch)

    result = runModelGetLastPeriod("Game of Life", 100, {"gridSize": 300, "x": 100})


    print("Finished Evaluation on Islands")
    print("Running next step...")

    pd.DataFrame(evaluated_set_X_batch).to_csv("Surrogates/Data/X.csv")
    pd.DataFrame(evaluated_set_y_batch).to_csv("Surrogates/Data/y.csv")

    # At this point we have the Sobol sampled parameters and their ABM evaluations (i.e. GDP from the ABM).

    # Build Out-of-sample set
    oos_set = get_unirand_samples(n_dimensions, final_test_size * budget, islands_exploration_range)

    selections = []
    for i, v in enumerate(oos_set):
        if v not in evaluated_set_X_batch:
            selections.append(i)
    oos_set = unique_rows(oos_set[selections])[:final_test_size]

    print("Finished building OOS set")
    print("Running next step...")

    # Evaluate the test set for the ABM response
    y_test =  evaluate_islands_on_set(oos_set)

    pd.DataFrame(oos_set).to_csv("Surrogates/Data/X_oos.csv")
    pd.DataFrame(y_test).to_csv("Surrogates/Data/y_oos.csv")

    end = time.time()
    print(end - start)
    print("Finished building test set for ABM response")

# Compute the Kriging surrogate
surrogate_models_kriging = GaussianProcessRegressor(random_state=0)
surrogate_models_kriging.fit(evaluated_set_X_batch, evaluated_set_y_batch)

# Compute the XGBoost surrogate
surrogate_model_XGBoost = fit_surrogate_model(evaluated_set_X_batch, evaluated_set_y_batch)

# At this point, we have the XGBoost surrogate model.  What we need next is the bit which returns the parameterisations
# for positive calibrations.


y_hat_test = [None] * 2
y_hat_test[0] = surrogate_models_kriging.predict(oos_set)
y_hat_test[1] = surrogate_model_XGBoost.predict(oos_set)

# MSE performance
mse_perf = np.zeros((2, monteCarlos))
for sur_idx in range(len(y_hat_test)):
    for i in range(monteCarlos):
        mse_perf[sur_idx, i] = mean_squared_error(y_test[i * test_size:(i + 1) * test_size],
                                                  y_hat_test[int(sur_idx)][i * test_size:(i + 1) * test_size])

experiment_labels = ["Kriging", "XGBoost (Batch)"]

mse_perf = pd.DataFrame(mse_perf, index=experiment_labels)

k_label = "Kriging: Mean " + '{:2.5f}'.format(mse_perf.iloc[0, :].mean()) + ", Variance " + '{:2.5f}'.format(
    mse_perf.iloc[0, :].var())
xgb_label = "XGBoost: Mean " + '{:2.5f}'.format(mse_perf.iloc[1, :].mean()) + ", Variance " + '{:2.5f}'.format(
    mse_perf.iloc[1, :].var())

fig, ax = plt.subplots(figsize=(12, 5))
sns.distplot(mse_perf.iloc[0, :], label=k_label, ax=ax)
sns.distplot(mse_perf.iloc[1, :], label=xgb_label, ax=ax)

plt.title("Out-Of-Sample Prediction Performance")
plt.xlabel('Mean-Squared Error')
plt.yticks([])

plt.legend()

fig.savefig("Surrogates/Plots/xgboost_kriging_ba_comparison_" + str(budget) + ".png");
