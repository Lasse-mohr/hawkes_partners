import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # Logistic sigmoid function
from patsy import dmatrix

class SelfExcitingLogisticRegression:
    def __init__(self, max_iter=100, tol=1e-6, random_state=None):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.params_ = None
        self.success_ = False
        self.nll_ = None  # Negative log-likelihood after fit
        self.n_params_ = None  # Number of parameters in the model

    def kernel(self, delta, time_diff):
        return np.exp(-delta * time_diff)

    def compute_s(self, events, times, delta):
        n = len(events)
        s = np.zeros(n)
        for i in range(1, n):
            time_diff = times[i] - times[:i]
            kernel_values = self.kernel(delta, time_diff)
            s[i] = np.sum(kernel_values * events[:i])
        return s

    def log_likelihood(self, params, events, times, covariates):
        alpha = params[0]
        beta = params[1]
        gamma = params[2:-1]
        delta = params[-1]

        s = self.compute_s(events, times, delta)
        linear_pred = alpha + beta * s + np.dot(covariates, gamma)
        probs = expit(linear_pred)

        nll = -np.sum(events * np.log(probs + 1e-8) + (1 - events) * np.log(1 - probs + 1e-8))
        return nll

    def fit(self, events, times, covariates):
        np.random.seed(self.random_state)
        n_covariates = covariates.shape[1]
        initial_params = np.random.rand(2 + n_covariates + 1)  # alpha, beta, gamma..., delta

        result = minimize(
            self.log_likelihood,
            initial_params,
            args=(events, times, covariates),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter},
            bounds=[(None, None)] * (2 + n_covariates) + [(1e-4, None)]
        )

        self.params_ = result.x
        self.success_ = result.success
        self.nll_ = result.fun  # Store the negative log-likelihood
        self.n_params_ = len(self.params_)  # Number of model parameters

        if not self.success_:
            raise ValueError("Optimization did not converge.")
        return self

    def aic(self):
        """Compute Akaike Information Criterion (AIC)."""
        if self.nll_ is None or self.n_params_ is None:
            raise ValueError("Model must be fitted before computing AIC.")
        return 2 * self.nll_ + 2 * self.n_params_

    def bic(self, n_samples):
        """Compute Bayesian Information Criterion (BIC)."""
        if self.nll_ is None or self.n_params_ is None:
            raise ValueError("Model must be fitted before computing BIC.")
        return 2 * self.nll_ + np.log(n_samples) * self.n_params_

    def predict_proba(self, times, covariates):
        if self.params_ is None:
            raise ValueError("The model is not fitted yet.")

        alpha = self.params_[0]
        beta = self.params_[1]
        gamma = self.params_[2:-1]
        delta = self.params_[-1]

        s = self.compute_s(np.zeros(len(times)), times, delta)
        linear_pred = alpha + beta * s + np.dot(covariates, gamma)
        return expit(linear_pred)

    def predict(self, times, covariates, threshold=0.5):
        probs = self.predict_proba(times, covariates)
        return (probs >= threshold).astype(int)


class SelfExcitingLogisticRegressionWithSplines(SelfExcitingLogisticRegression):
    def fit(self, events, times, covariates, age):
        spline_basis = dmatrix("bs(age, df=5, degree=3)", {"age": age})
        full_covariates = np.hstack([covariates, spline_basis])
        self.n_spline_params_ = spline_basis.shape[1]
        super().fit(events, times, full_covariates)

    def aic(self):
        if self.nll_ is None or self.params_ is None:
            raise ValueError("Model must be fitted before computing AIC.")
        return 2 * self.nll_ + 2 * (len(self.params_) + self.n_spline_params_)

    def bic(self, n_samples):
        if self.nll_ is None or self.params_ is None:
            raise ValueError("Model must be fitted before computing BIC.")
        return 2 * self.nll_ + np.log(n_samples) * (len(self.params_) + self.n_spline_params_)

    def predict_proba(self, times, covariates, age):
        spline_basis = dmatrix("bs(age, df=5, degree=3)", {"age": age})
        full_covariates = np.hstack([covariates, spline_basis])
        return super().predict_proba(times, full_covariates)

