import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # Logistic sigmoid function
from numpy.random import default_rng


class SelfExcitingLogisticRegression:
    """ 
    
    """
    def __init__(self, max_iter=100, tol=1e-6, rng=default_rng(1), time_types=['continuous', 'discrete']):
        self.time_types = time_types
        self.rng = rng
        self.max_iter = max_iter
        self.tol = tol
        self.params_ = None
        self.success_ = False
        self.nll_ = None  # Negative log-likelihood after fit
        self.n_params_ = None  # Number of parameters in the model

    def compute_kernels(self, events_all, times_all, delta_s, delta_c, individuals_all):
        """
        Computes the discrete and continuous kernels s_ij and c_ij for all individuals.
        
        Args:
            events_all: Binary event array for all individuals (1D).
            times_all: 2D array with continuous time (row 0) and discrete time (row 1).
            delta_s: Decay parameter for discrete time.
            delta_c: Decay parameter for continuous time.
            individuals_all: Array of individual identifiers (1D).
        
        Returns:
            s: Array of discrete kernel values (1D).
            c: Array of continuous kernel values (1D).
        """
        n_total = len(events_all)
        s = np.zeros(n_total)
        c = np.zeros(n_total)
        
        # Split continuous and discrete time
        continuous_time = times_all[0, :]
        discrete_time = times_all[1, :]

        last_individual = individuals_all[0]
        for j in range(n_total):
            if individuals_all[j] != last_individual:
                # New individual, reset last_individual and skip summation
                last_individual = individuals_all[j]
                continue

            # Indices of past events for the same individual
            past_indices = np.where((individuals_all[:j] == individuals_all[j]))[0]

            if len(past_indices) == 0:
                continue

            # Continuous kernel: c_ij
            time_diff = continuous_time[j] - continuous_time[past_indices]
            c[j] = np.sum(np.exp(-delta_c * time_diff) * events_all[past_indices])

            # Discrete kernel: s_ij
            discrete_diff = discrete_time[j] - discrete_time[past_indices]
            s[j] = np.sum(np.exp(-delta_s * discrete_diff) * events_all[past_indices])

        return s, c

    def log_likelihood(self, params):
        """
        Negative log-likelihood for the logistic regression with self-exciting kernels.
        """
        alpha = params[0]
        beta_s = params[1]
        beta_c = params[2]
        gamma = params[3:-2]
        delta_s = params[-2]
        delta_c = params[-1]

        # Compute kernels
        s, c = self.compute_kernels(
            self.events_all,
            self.times_all,
            delta_s,
            delta_c,
            self.individuals_all
        )

        # Compute linear predictor
        linear_pred = alpha +  np.dot(self.covariates_all, gamma)
        if 'continious' in self.time_types:
            linear_pred += beta_s * s 
        if 'discrete' in self.time_types:
            linear_pred = beta_c * c

        probs = expit(linear_pred)

        # Negative log-likelihood
        nll = -np.sum(
            self.events_all * np.log(probs + 1e-8) +
            (1 - self.events_all) * np.log(1 - probs + 1e-8)
        )
        return nll

    def fit(self, events_all, times_all, covariates_all, individuals_all):
        """
        Fits the logistic regression model.
        """
        self.events_all = events_all
        self.times_all = times_all  # 2xN array: continuous and discrete time
        self.covariates_all = covariates_all
        self.individuals_all = individuals_all

        n_covariates = covariates_all.shape[1]
        initial_params = self.rng.uniform(0, 1 ,size=3 + n_covariates + 2)  # alpha, beta_s, beta_c, gamma..., delta_s, delta_c

        # Optimize
        result = minimize(
            self.log_likelihood,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter},
            bounds=[(None, None)] * (3 + n_covariates) + [(1e-4, None), (1e-4, None)]
        )

        params = result.x
        # If the model does not include continuous or discrete time, remove the corresponding parameters
        if 'continuous' not in self.time_types and 'discrete' not in self.time_types:
            params = params[:-2]
        else:
            if 'continuous' not in self.time_types:
                params = params[:-1]
            elif 'discrete' not in self.time_types:
                params = params[:-2] + params[-1]

        self.params_ = params
        self.success_ = result.success
        self.nll_ = result.fun  # Store the negative log-likelihood
        self.n_params_ = len(self.params_)  # Number of model parameters

        if not self.success_:
            raise ValueError("Optimization did not converge.")
        return self

    def predict_proba(self, times_all, covariates_all, individuals_all):
        """
        Predict probabilities for new data.
        """
        if self.params_ is None:
            raise ValueError("The model is not fitted yet.")

        alpha = self.params_[0]
        beta_s = self.params_[1]
        beta_c = self.params_[2]
        gamma = self.params_[3:-2]
        delta_s = self.params_[-2]
        delta_c = self.params_[-1]

        # Assuming events are zeros during prediction
        events_all = np.zeros(times_all.shape[1])

        # Compute kernels
        s, c = self.compute_kernels(
            events_all,
            times_all,
            delta_s,
            delta_c,
            individuals_all
        )

        # Compute linear predictor
        linear_pred = alpha + beta_s * s + beta_c * c + np.dot(covariates_all, gamma)
        return expit(linear_pred)

    def predict(self, times_all, covariates_all, individuals_all, threshold=0.5):
        """
        Predict binary outcomes based on a threshold.
        """
        probs = self.predict_proba(times_all, covariates_all, individuals_all)
        return (probs >= threshold).astype(int)

    def aic(self):
        """
        Computes the Akaike Information Criterion (AIC).
        AIC = 2 * number_of_parameters - 2 * log_likelihood
        """
        if self.nll_ is None or self.n_params_ is None:
            raise ValueError("Model must be fitted before computing AIC.")
        return 2 * self.n_params_ + 2 * self.nll_

    def bic(self, n_samples):
        """
        Computes the Bayesian Information Criterion (BIC).
        BIC = log(n_samples) * number_of_parameters - 2 * log_likelihood
        """
        if self.nll_ is None or self.n_params_ is None:
            raise ValueError("Model must be fitted before computing BIC.")
        return np.log(n_samples) * self.n_params_ + 2 * self.nll_