import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import expit
from src.self_exciting.hawkes_regression import (
    SelfExcitingLogisticRegression,
    SelfExcitingLogisticRegressionWithSplines,
)

# Settings
n_samples_range = np.logspace(2, 5.3, num=10, dtype=int)  # Logarithmic spacing: 100 to ~200,000
fit_times_base = []
fit_times_spline = []

# Generate synthetic data and test
rng = np.random.default_rng(seed=42)
true_alpha, true_beta, true_gamma, true_delta = 0.5, 1.2, [0.8, -0.5, 0.3], 0.2

for n_samples in n_samples_range:
    # Generate synthetic data
    times = np.sort(rng.uniform(0, 10, n_samples))
    age = rng.uniform(20, 50, n_samples).reshape(-1, 1)
    sex = rng.choice([0, 1], size=n_samples).reshape(-1, 1)
    cum_partner_count = rng.choice(np.arange(1, 10), size=n_samples).reshape(-1, 1)
    covariates = np.hstack([age, sex, cum_partner_count])

    # Generate synthetic partner change events
    true_s = np.zeros(n_samples)
    for i in range(1, n_samples):
        time_diff = times[i] - times[:i]
        true_s[i] = np.sum(np.exp(-true_delta * time_diff) * rng.choice([0, 1], size=i))
    linear_pred = (
        true_alpha
        + true_beta * true_s
        + np.dot(covariates, true_gamma)
        + 0.02 * age.ravel()
        - 0.0005 * (age.ravel() - 35) ** 2
    )
    probs = expit(linear_pred)
    events = rng.binomial(1, probs)

    # Time base model
    base_model = SelfExcitingLogisticRegression(max_iter=200)
    start_time = time.time()
    base_model.fit(events, times, covariates)
    fit_times_base.append(time.time() - start_time)

    # Time spline model
    spline_model = SelfExcitingLogisticRegressionWithSplines(max_iter=200)
    start_time = time.time()
    spline_model.fit(events, times, covariates, age.ravel())
    fit_times_spline.append(time.time() - start_time)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(n_samples_range, fit_times_base, label="Base Model", marker="o")
plt.plot(n_samples_range, fit_times_spline, label="Spline Model", marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Observations (log scale)")
plt.ylabel("Fit Time (seconds, log scale)")
plt.title("Model Fit Time vs Number of Observations")
plt.legend()
plt.grid(True)

# Save the plot
plot_path = "model_fit_time_comparison.png"
plt.savefig(plot_path)
plt.show()