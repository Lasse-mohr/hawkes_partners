import numpy as np
from scipy.special import expit
from src.self_exciting.hawkes_regression import (
    SelfExcitingLogisticRegression,
    SelfExcitingLogisticRegressionWithSplines
)

# Set random seed for reproducibility
rng = np.random.default_rng(seed=42)

# Generate synthetic data
n_samples = 1_000_000
times = np.sort(rng.uniform(0, 10, n_samples))  # Random event times
age = rng.uniform(20, 50, n_samples).reshape(-1, 1)  # Random ages
sex = rng.choice([0, 1], size=n_samples).reshape(-1, 1)  # Binary sex covariate
cum_partner_count = rng.choice(np.arange(1, 10), size=n_samples).reshape(-1, 1)  # Cumulative partner count

# Combine covariates into a single matrix
covariates = np.hstack([age, sex, cum_partner_count])

# Generate synthetic partner change events
true_alpha, true_beta, true_gamma, true_delta = 0.5, 1.2, [0.8, -0.5, 0.3], 0.2
true_s = np.zeros(n_samples)

for i in range(1, n_samples):
    time_diff = times[i] - times[:i]
    true_s[i] = np.sum(np.exp(-true_delta * time_diff) * rng.choice([0, 1], size=i))

# Define the linear predictor
linear_pred = (
    true_alpha
    + true_beta * true_s
    + np.dot(covariates, true_gamma)
    + 0.02 * age.ravel() - 0.0005 * (age.ravel() - 35) ** 2  # Non-monotonic effect of age
)

# Compute probabilities and generate binary events
probs = expit(linear_pred)
events = rng.binomial(1, probs)


# Fit and evaluate the base model
base_model = SelfExcitingLogisticRegression(max_iter=200)
base_model.fit(events, times, covariates)

print("Base Model Results:")
print("AIC:", base_model.aic())
print("BIC:", base_model.bic(n_samples))

# Fit and evaluate the spline-enhanced model
spline_model = SelfExcitingLogisticRegressionWithSplines(max_iter=200)
spline_model.fit(events, times, covariates, age.ravel())

print("\nSpline Model Results:")
print("AIC:", spline_model.aic())
print("BIC:", spline_model.bic(n_samples))
