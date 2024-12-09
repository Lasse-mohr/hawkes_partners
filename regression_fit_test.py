import numpy as np
import time
from scipy.special import expit
from src.self_exciting.hawkes_regression import (
    SelfExcitingLogisticRegression,
    likelihood_ratio_test,
)

max_iter = int(10**3)

# Set random seed for reproducibility
rng = np.random.default_rng(seed=42)

# Parameters for synthetic data generation
n_individuals = int(10**3)  # Number of individuals
max_events_per_individual = 5 # Maximum number of events per individual
total_events = n_individuals * max_events_per_individual  # Approximate total events

# Generate synthetic data
individuals = np.repeat(np.arange(n_individuals), max_events_per_individual)
n_events_per_individual = rng.integers(1, max_events_per_individual, size=n_individuals)


continuous_time = []
discrete_time = []
events = []
covariates = []


# Generate synthetic partner change events
true_alpha, true_beta_s, true_beta_c, true_gamma, true_delta_s, true_delta_c = (
    0.5,
    1.2,
    0.5,
    [0.8, -0.5, 0.3],
    0.1,
    0.2,
)

# Generate synthetic data for each individual
for i, n_events in enumerate(n_events_per_individual):
    # Generate continuous and discrete times for this individual
    continuous_times = np.sort(rng.uniform(0, 10, n_events))
    discrete_times = np.arange(1, n_events + 1)

    # Generate covariates for this individual
    age = rng.uniform(20, 50, n_events).reshape(-1, 1)
    sex = rng.choice([0, 1], size=n_events).reshape(-1, 1)
    cum_partner_count = rng.choice(np.arange(1, 10), size=n_events).reshape(-1, 1)
    covariates_individual = np.hstack([age, sex, cum_partner_count])

    s = np.zeros(n_events)
    c = np.zeros(n_events)

    for j in range(1, n_events):
        discrete_diff = discrete_times[j] - discrete_times[:j]
        continuous_diff = continuous_times[j] - continuous_times[:j]
        s[j] = np.sum(np.exp(-true_delta_s * discrete_diff) * rng.choice([0, 1], size=j))
        c[j] = np.sum(np.exp(-true_delta_c * continuous_diff) * rng.choice([0, 1], size=j))

    if n_events > 1:
        if np.max(s) > 0:
            s /= np.max(s) # Normalize to [0, 1]
        if np.max(c) > 0:
            c /= np.max(c)  # Normalize to [0, 1]

    # Define the linear predictor for this individual
    linear_pred = (
        true_alpha
        + true_beta_s * s
        + true_beta_c * c
        + np.dot(covariates_individual, true_gamma)
        + 0.02 * age.ravel()
        - 0.0005 * (age.ravel() - 35) ** 2  # Non-monotonic effect of age
    )
    # normalize linear predictions
    linear_pred -= linear_pred.mean()

    # Compute probabilities and generate binary events
    probs = expit(linear_pred)
    events_individual = rng.binomial(1, probs)

    # Store this individual's data
    continuous_time.append(continuous_times)
    discrete_time.append(discrete_times)
    events.append(events_individual)
    covariates.append(covariates_individual)


# Combine all individuals' data
continuous_time = np.concatenate(continuous_time)
discrete_time = np.concatenate(discrete_time)
events = np.concatenate(events)

print(f'Proportion of nonzero events: {np.sum(events)/len(events)}')

covariates = np.vstack(covariates)
covariates = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0)

individuals = np.repeat(np.arange(n_individuals), max_events_per_individual)[: len(continuous_time)]



# Combine continuous and discrete times into a 2xN array
times = np.vstack([continuous_time, discrete_time])

# Fit the two-dim time model

complex_model = SelfExcitingLogisticRegression(
    max_iter=max_iter,
    time_types=['discrete', 'continuous'],
    ignore_errors=True
    )
then = time.time()
complex_model.fit_with_gradients(events, times, covariates, individuals)
now = time.time()
print(f"Serial implementation: Time elapsed: {now - then:.2f}")

print(f"True parameters:{true_alpha}, {true_gamma}, {true_beta_c}, {true_delta_c}, {true_beta_s},{true_delta_s}")
print(f"Estimated parameters:{complex_model.params_}")