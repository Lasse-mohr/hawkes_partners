import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.special import expit  # Logistic sigmoid function
from scipy.stats import norm
from prettytable import PrettyTable


class SelfExcitingLogisticRegression:
    def __init__(self, max_iter=100, tol=1e-6, rng=None, time_types=None, n_covariates=0):
        """
        Initialize the logistic regression model with optional continuous and/or discrete kernels.
        
        Args:
            max_iter: Maximum number of iterations for optimization.
            tol: Tolerance for optimization.
            rng: Random number generator for parameter initialization.
            time_types: List specifying which time kernels to include: 'continuous', 'discrete', or both.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.rng = rng or np.random.default_rng(1)
        self.time_types = time_types or []

        # Dynamically determine parameter structure
        self.use_continuous = 'continuous' in self.time_types
        self.use_discrete = 'discrete' in self.time_types

        # Parameter index mapping
        self.param_indices = {'alpha': 0}
        idx = 1
        if self.use_discrete:
            self.param_indices.update({'beta_s': idx, 'delta_s': idx + 1})
            idx += 2
        if self.use_continuous:
            self.param_indices.update({'beta_c': idx, 'delta_c': idx + 1})
            idx += 2

        self.param_indices['gamma'] = list(range(idx, idx + n_covariates))  # Covariate coefficients


    def compute_kernels(self, events_all, times_all, delta_s, delta_c, individuals_all):
        """
        Computes the discrete and continuous kernels s_ij and c_ij for all individuals.
        """
        n_total = len(events_all)
        s = np.zeros(n_total)
        c = np.zeros(n_total)

        continuous_time, discrete_time = times_all

        last_individual = individuals_all[0]
        for j in range(n_total):
            if individuals_all[j] != last_individual:
                last_individual = individuals_all[j]
                continue

            past_indices = np.where(individuals_all[:j] == individuals_all[j])[0]
            if len(past_indices) == 0:
                continue

            if self.use_continuous:
                time_diff = continuous_time[j] - continuous_time[past_indices]
                c[j] = np.sum(np.exp(-delta_c * time_diff) * events_all[past_indices])

            if self.use_discrete:
                discrete_diff = discrete_time[j] - discrete_time[past_indices]
                s[j] = np.sum(np.exp(-delta_s * discrete_diff) * events_all[past_indices])

        return s, c


    def log_likelihood(self, params):
        """
        Negative log-likelihood for the logistic regression with self-exciting kernels.
        """
        alpha = params[self.param_indices['alpha']]
        beta_s = params[self.param_indices['beta_s']] if self.use_discrete else 0
        beta_c = params[self.param_indices['beta_c']] if self.use_continuous else 0
        gamma = params[self.param_indices['gamma']]
        delta_s = params[self.param_indices['delta_s']] if self.use_discrete else 0
        delta_c = params[self.param_indices['delta_c']] if self.use_continuous else 0

        s, c = self.compute_kernels(
            self.events_all,
            self.times_all,
            delta_s,
            delta_c,
            self.individuals_all
        )

        linear_pred = alpha + np.dot(self.covariates_all, gamma)
        if self.use_discrete:
            linear_pred += beta_s * s
        if self.use_continuous:
            linear_pred += beta_c * c

        probs = expit(linear_pred)
        nll = -np.sum(
            self.events_all * np.log(probs + 1e-8) +
            (1 - self.events_all) * np.log(1 - probs + 1e-8)
        )
        return nll

    def compute_hessian(self, params):
        """
        Compute the Hessian matrix analytically for the negative log-likelihood.
        """
        alpha = params[self.param_indices['alpha']]
        beta_s = params[self.param_indices['beta_s']] if self.use_discrete else 0
        beta_c = params[self.param_indices['beta_c']] if self.use_continuous else 0
        gamma = params[self.param_indices['gamma']]
        delta_s = params[self.param_indices['delta_s']] if self.use_discrete else 0
        delta_c = params[self.param_indices['delta_c']] if self.use_continuous else 0

        s, c = self.compute_kernels(
            self.events_all,
            self.times_all,
            delta_s,
            delta_c,
            self.individuals_all
        )

        continuous_time, discrete_time = self.times_all

        # Compute second derivatives
        d2s_deltas2 = np.zeros(len(self.events_all))
        d2c_deltac2 = np.zeros(len(self.events_all))

        for j in range(len(self.events_all)):
            # [Same logic for individuals]
            past_indices = np.where(individuals_all[:j] == individuals_all[j])[0]

            if self.use_discrete:
                discrete_diff = discrete_time[j] - discrete_time[past_indices]
                exp_term = np.exp(-delta_s * discrete_diff)
                d2s_deltas2[j] = np.sum((discrete_diff ** 2) * exp_term * self.events_all[past_indices])

            if self.use_continuous:
                time_diff = continuous_time[j] - continuous_time[past_indices]
                exp_term = np.exp(-delta_c * time_diff)
                d2c_deltac2[j] = np.sum((time_diff ** 2) * exp_term * self.events_all[past_indices])

        # Compute linear predictor, probs, weights as before

        # Compute gradients of eta
        grad_eta = np.zeros((len(self.events_all), len(params)))
        # [Fill in grad_eta as in compute_gradient]

        # Compute second derivatives of eta
        hess_eta = np.zeros((len(self.events_all), len(params), len(params)))

        # Initialize hess_eta to zeros

        # Second derivative terms
        for i in range(len(self.events_all)):
            idx = 0
            # With respect to delta_s
            if self.use_discrete:
                # Second derivative w.r.t delta_s
                hess_eta[i, self.param_indices['delta_s'], self.param_indices['delta_s']] = beta_s * d2s_deltas2[i]
                # Mixed partials
                hess_eta[i, self.param_indices['beta_s'], self.param_indices['delta_s']] = ds_deltas[i]
                hess_eta[i, self.param_indices['delta_s'], self.param_indices['beta_s']] = ds_deltas[i]

            # Similarly for delta_c
            if self.use_continuous:
                hess_eta[i, self.param_indices['delta_c'], self.param_indices['delta_c']] = beta_c * d2c_deltac2[i]
                hess_eta[i, self.param_indices['beta_c'], self.param_indices['delta_c']] = dc_deltac[i]
                hess_eta[i, self.param_indices['delta_c'], self.param_indices['beta_c']] = dc_deltac[i]

        # Compute Hessian
        hessian = np.zeros((len(params), len(params)))
        for i in range(len(self.events_all)):
            # First term: (y_i - p_i) * second derivatives
            term1 = -residuals[i] * hess_eta[i]

            # Second term: p_i * (1 - p_i) * grad_eta_i * grad_eta_i^T
            term2 = probs[i] * (1 - probs[i]) * np.outer(grad_eta[i], grad_eta[i])

            hessian += term1 + term2

        return hessian


    def fit(self, events_all, times_all, covariates_all, individuals_all):
        """
        Fit the model using maximum likelihood estimation.
        """
        self.events_all = events_all
        self.times_all = times_all
        self.covariates_all = covariates_all
        self.individuals_all = individuals_all

        n_covariates = covariates_all.shape[1]
        n_params = 1 + 2 * (self.use_continuous + self.use_discrete) + n_covariates
        initial_params = self.rng.uniform(0, 1, size=n_params)

        result = minimize(
            self.log_likelihood,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'ftol': self.tol},
            bounds=[(None, None)] * n_params,
        )

        self.params_ = result.x
        self.success_ = result.success
        self.nll_ = result.fun
        self.n_params_ = len(self.params_)

        if not self.success_:
            raise ValueError("Optimization did not converge.")
        return self


    def compute_significance(self):
        """
        Computes standard errors, confidence intervals, and p-values for parameters.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted before computing significance.")

        # Compute Hessian and variance-covariance matrix
        hessian = self.compute_hessian(self.params_)
        cov_matrix = np.linalg.inv(hessian)

        # Extract standard errors
        standard_errors = np.sqrt(np.diag(cov_matrix))

        # Compute confidence intervals
        z_critical = norm.ppf(0.975)  # 95% confidence
        confidence_intervals = [
            (param - z_critical * se, param + z_critical * se)
            for param, se in zip(self.params_, standard_errors)
        ]

        # Compute p-values
        z_scores = self.params_ / standard_errors
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

        # Store results
        self.standard_errors_ = standard_errors
        self.confidence_intervals_ = confidence_intervals
        self.p_values_ = p_values

        return {
            "params": self.params_,
            "standard_errors": standard_errors,
            "confidence_intervals": confidence_intervals,
            "p_values": p_values,
        }


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


    def summary(self, n_samples):
        """
        Prints a nicely formatted table of model results.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted before generating a summary.")

        # Ensure significance results are computed
        results = self.compute_significance()

        table = PrettyTable()
        table.field_names = ["Parameter", "Estimate", "Std. Error", "z-score", "p-value", "95% CI"]

        for i, (param, se, ci, p) in enumerate(
            zip(results["params"], results["standard_errors"], results["confidence_intervals"], results["p_values"])
        ):
            z_score = param / se
            table.add_row([
                f"Param {i+1}",
                f"{param:.3f}",
                f"{se:.3f}",
                f"{z_score:.3f}",
                f"{p:.3f}",
                f"({ci[0]:.2f}, {ci[1]:.2f})"
                ])

        aic = self.aic()
        bic = self.bic(n_samples)

        print(table)
        print("\nModel Fit Statistics:")
        print(f"Log-Likelihood: {-self.nll_:.3f}")
        print(f"AIC: {aic:.3f}")
        print(f"BIC: {bic:.3f}")
        print('')

