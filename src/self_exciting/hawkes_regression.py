import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # Logistic sigmoid function
from numpy.random import default_rng
from scipy.stats import chi2
from multiprocessing import cpu_count
from sklearn.linear_model import LogisticRegression
from memory_profiler import profile
from joblib import Parallel, delayed

class SelfExcitingLogisticRegression:
    """ 
    Logistic regression with self-exciting kernels for continuous and discrete time. 
    """
    def __init__(self, max_iter=100, tol=1e-6, rng=default_rng(1),
                 time_types=['continuous', 'discrete'],
                 ignore_errors=True):
        self.time_types = time_types
        self.rng = rng
        self.max_iter = max_iter
        self.tol = tol
        self.success_ = False
        self.nll_ = None  # Negative log-likelihood after fit
        self.n_params_ = None  # Number of parameters in the model
        self.ignore_errors = ignore_errors
        self.alpha = 0
        self.gamma = 0
        self.beta_c = 0
        self.delta_c = 0
        self.beta_s = 0
        self.delta_s = 0

    def compute_kernels(self, events_all, times_all, delta_s, delta_c, positive_obs_in_groups_idx):
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

        # we need to iterate each individual and their events 
        # These groups have been restricted to only include positive events
        #   and have at least two such events.
        for group_idx in positive_obs_in_groups_idx:
            # Extract data for the current individual
            individual_continuous_time = continuous_time[group_idx]
            individual_discrete_time = discrete_time[group_idx]

            start_idx = group_idx[0]
            # Compute kernels for the current individual
            for j in range(1, len(group_idx)):
                past_indices = np.arange(0, j)

                # Continuous kernel: c_ij
                time_diff = individual_continuous_time[j] - individual_continuous_time[past_indices]
                c[start_idx + j] = np.sum(np.exp(-delta_c * time_diff))

                # Discrete kernel: s_ij
                discrete_diff = individual_discrete_time[j] - individual_discrete_time[past_indices]
                s[start_idx + j] = np.sum(np.exp(-delta_s * discrete_diff))

        return s, c

    def set_params(self, params):
        """
        Set the model parameters.

        This function is used to set the model parameters after optimization.
        Params is a 1D array with the following order:
        [alpha, gamma..., beta_c, delta_c, beta_s, delta_s]
        If the model does not include continuous or discrete time, the corresponding parameters are set to zero.
        """
        self.alpha = params[0]
        self.gamma=params[1:-4]
        if ('continuous' not in self.time_types) and ('discrete' not in self.time_types):
            self.n_params_ = len(params) - 4

        else:
            if ('continuous' in self.time_types) and ('discrete' not in self.time_types):
                self.beta_c = params[-4]
                self.delta_c = params[-3]
                self.n_params_ = len(params) - 2

            elif ('continuous' not in self.time_types) and ('discrete' in self.time_types):
                self.beta_s = params[-2]
                self.delta_s = params[-1]
                self.n_params_ = len(params) - 2
            else:
                self.beta_c = params[-4]
                self.delta_c = params[-3]
                self.beta_s = params[-2]
                self.delta_s = params[-1]
                self.n_params_ = len(params)
        print(f'Parameters: {self.alpha}, {self.gamma}, {self.beta_c}, {self.delta_c}, {self.beta_s}, {self.delta_s}')
        print(f'Number of parameters: {self.n_params_}')

    def load_params(self):
        """ 
        Helper function to Load the parameters from the optimization result avoiding indexing errors.
        """
        if self.n_params_ is None and not self.ignore_errors:
            raise ValueError("The model is not fitted yet.")
        else:
            return self.alpha, self.gamma, self.beta_c, self.delta_c, self.beta_s, self.delta_s
    
    def compute_linear_probs(self, alpha, gamma, beta_s, beta_c, s, c, covariates_all):
        # Compute linear predictor
        linear_pred = alpha + np.dot(covariates_all, gamma)
        if 'continious' in self.time_types:
            linear_pred += beta_s * s 
        if 'discrete' in self.time_types:
            linear_pred += beta_c * c

        probs = expit(linear_pred)

        return probs

    def log_likelihood(self, params, events_all, times_all, covariates_all, positive_obs_in_groups_idx):
        """
        Negative log-likelihood for the logistic regression with self-exciting kernels.
        """
        alpha = params[0]
        gamma = params[1:-4]
        beta_c = params[-4]
        delta_c = params[-3]
        beta_s = params[-2]
        delta_s = params[-1]

        # Compute kernels
        s, c = self.compute_kernels(
            events_all=events_all,
            times_all=times_all,
            delta_s=delta_s,
            delta_c=delta_c,
            positive_obs_in_groups_idx=positive_obs_in_groups_idx
        )

        probs = self.compute_linear_probs(
            alpha=alpha,
            gamma=gamma,
            beta_s=beta_s,
            beta_c=beta_c,
            s=s,
            c=c,
            covariates_all=covariates_all)

        # Negative log-likelihood
        nll = -np.sum(
            events_all * np.log(probs + 1e-10) +
            (1 - events_all) * np.log(1 - probs + 1e-10)
        )
        
        return nll
    
    def compute_time_series_groups_idx(self, individuals_all, events_all):
        """ 
        Use numpy to compute the indices for each individual in the time series data. 

        We restrict each time series to positive elements, and require they have
        at least two such events. The reason for this, is that the kernel values
        are unaffected by negative events and time-series of length 1. Therefore,
        we can safely ignore them when computing the likelihood values.
        """
        _, indices = np.unique(individuals_all, return_index=True)
        indices = np.append(indices, len(individuals_all))
        group_idxs = [
            list(range(indices[i], indices[i + 1]))
                for i in range(len(indices) - 1)
                if indices[i + 1] - indices[i] > 1 # Require at least two events
            ]

        # filter out negative event observations
        group_idxs = [[idx for idx in idxs if events_all[idx] > 0] for idxs in group_idxs]

        # filter out groups of that are now of length 1 after negative event filtering
        #   as they don't contribute to the kernel values
        positive_obs_in_group_idx = [idx for idx in group_idxs if len(idx) > 1]

        return positive_obs_in_group_idx

    def estimate_initial_params(self, events_all, covariates_all, method='logistic'):
        """
        Method for making qualified initial estimates of the model parameters.

        The method is based on the assumption that the model is logistic, and that the
        covariates are standardized. The initial estimates are computed by fitting a logistic
        regression model to the data only using the covariates.
        
        kernel parameters are set to 0.1 and 0.01 for delta_s and delta_c respectively.

        Args:
            events_all: Binary event outcomes for all observations (1D).
            covariates_all: Covariates matrix (N x M).
            method: Method for estimating initial parameters. Currently only 'logistic' is supported. 
        """
        if method != 'logistic':
            raise ValueError("Only 'logistic' method is supported for estimating initial parameters.")

        # Fit a logistic regression model to the data
        clf = LogisticRegression(solver='lbfgs', penalty=None)
        clf.fit(covariates_all, events_all)

        # Extract the estimated coefficients
        gamma = clf.coef_[0]
        alpha = clf.intercept_[0]

        # Set initial estimates for the kernel parameters
        beta_s = 0.1
        delta_s = 0.01
        beta_c = 0.1
        delta_c = 0.01

        return np.concatenate(([alpha], gamma, [beta_c, delta_c, beta_s, delta_s]))

    def fit(self, events_all, times_all, covariates_all, individuals_all):
        """
        fits the logistic regression model.
        """
        positive_obs_in_groups_idx = self.compute_time_series_groups_idx(individuals_all, events_all)

        n_covariates = covariates_all.shape[1]

        if 'continuous' in self.time_types or 'discrete' in self.time_types:
            initial_params = self.estimate_initial_params(events_all, covariates_all, method='logistic')

            #initial_params = self.rng.uniform(0, 1 ,size=1 + n_covariates + 4)  # alpha, beta_s, beta_c, gamma..., delta_s, delta_c

            alpha_bound = [(None, None)]
            gamma_bound = [(None, None) for _ in range(n_covariates)]
            beta_s_bound = [(None, None)]
            delta_s_bound = [(1e-4, None)]
            beta_c_bound = [(None, None)]
            delta_c_bound = [(1e-4, None)]

            bounds = alpha_bound + gamma_bound + beta_s_bound + delta_s_bound + beta_c_bound + delta_c_bound

            def objective(params):
                return self.log_likelihood(
                    params=params,
                    events_all=events_all,
                    times_all=times_all,
                    covariates_all=covariates_all,
                    positive_obs_in_groups_idx=positive_obs_in_groups_idx
                )

            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': self.max_iter},
                bounds=bounds
            )

            self.set_params(result.x)
            self.success_ = result.success
            self.nll_ = result.fun  # Store the negative log-likelihood

            if not self.success_ and not self.ignore_errors:
                raise ValueError("Optimization did not converge.")
            elif not self.success_:
                print("Warning: Optimization did not converge.")

        else:
            # Fit a logistic regression model to the data
            clf = LogisticRegression(solver='lbfgs', penalty=None)
            clf.fit(covariates_all, events_all)

            # Extract the estimated coefficients
            params = list(clf.intercept_[0]) + list(clf.coef_[0]) + [0, 0, 0, 0]
            self.set_params(params)
            self.success_ = True
            self.nll_= -clf.score(covariates_all, events_all)

    def predict_proba(self, times, covariates, prior_event_sequence):
        """
        Predict probabilities for new data.
        """
        if self.n_params_ is None:
            raise ValueError("The model is not fitted yet.")

        alpha, gamma, beta_c, delta_c, beta_s, delta_s = self.load_params()

        if alpha is None or gamma is None or beta_s is None or beta_c is None or delta_s is None or delta_c is None:
            raise ValueError("One or more model parameters None. Model have been fitted, but not successfully.")

        # Assuming events are zeros during prediction

        # Compute kernels
        s, c = self.compute_kernels(
            events_all = prior_event_sequence,
            times_all=times,
            delta_s=delta_s,
            delta_c=delta_c,
            positive_obs_in_groups_idx=np.ones_like(prior_event_sequence)
        )

        # Compute linear predictor
        linear_pred = alpha + np.dot(covariates, gamma)
        if 'continuous' in self.time_types:
            linear_pred += beta_c * c
        if 'discrete' in self.time_types:
            linear_pred += beta_s * s

        return expit(linear_pred)

    def predict(self, times, covariates, prior_event_sequence, threshold=0.5):
        """
        Predict binary outcomes based on a threshold.
        """
        probs = self.predict_proba(
            times=times,
            covariates=covariates,
            prior_event_sequence=prior_event_sequence
            )

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

############ Method with gradients #####################

class SelfExcitingLogisticRegressionWithGrad(SelfExcitingLogisticRegression):

    def fit(self, events_all, times_all, covariates_all, individuals_all):
        """
        Fits the logistic regression model using the gradient for optimization.
        """

        positive_obs_in_groups_idx = self.compute_time_series_groups_idx(individuals_all, events_all)

        if 'continuous' in self.time_types and 'discrete' in self.time_types:
            # Initialize data

            # Set initial parameters
            n_covariates = covariates_all.shape[1]

            initial_params = self.estimate_initial_params(events_all, covariates_all, method='logistic') 
            print(f'Initial parameters: {initial_params}')

            # Define parameter bounds
            alpha_bound = [(None, None)]
            gamma_bound = [(None, None) for _ in range(n_covariates)]
            beta_c_bound = [(None, None)]
            delta_c_bound = [(1e-4, None)]  # Ensure positive decay
            beta_s_bound = [(None, None)]
            delta_s_bound = [(1e-4, None)]  # Ensure positive decay

            bounds = alpha_bound + gamma_bound + beta_c_bound + delta_c_bound + beta_s_bound + delta_s_bound

            # Objective function: log-likelihood
            def objective(params):
                return self.log_likelihood(
                    params=params,
                    events_all=events_all,
                    times_all=times_all,
                    covariates_all=covariates_all,
                    positive_obs_in_groups_idx=positive_obs_in_groups_idx
                )

            # Gradient function
            def gradient(params):
                return compute_gradients(
                    params=params,
                    events_all=events_all,
                    times_all=times_all,
                    covariates_all=covariates_all,
                    positive_obs_in_groups_idx=positive_obs_in_groups_idx,
                    time_types=self.time_types,
                    compute_kernels=self.compute_kernels
                )

            # Use scipy.optimize.minimize with gradients
            result = minimize(
                objective,
                initial_params,
                method='trust-constr',
                jac=gradient,  # Pass the gradient function here
                options={'maxiter': self.max_iter, 'disp': True},
                bounds=bounds
            )

            # Store the results
            self.set_params(result.x)
            self.success_ = result.success
            self.nll_ = result.fun  # Store the negative log-likelihood

            if not self.success_ and not self.ignore_errors:
                raise ValueError("Optimization did not converge.")
            elif not self.success_:
                print("Warning: Optimization did not converge.")

        else:
            # Fit a logistic regression model to the data
            clf = LogisticRegression(solver='lbfgs', penalty=None)
            clf.fit(covariates_all, events_all)

            # Extract the estimated coefficients
            print(clf)
            params = clf.intercept_.tolist() + clf.coef_[0].tolist() + [0, 0, 0, 0]
            self.set_params(params)
            self.success_ = True
            self.nll_= self.log_likelihood(
                    params=params,
                    events_all=events_all,
                    times_all=times_all,
                    covariates_all=covariates_all,
                    positive_obs_in_groups_idx=positive_obs_in_groups_idx
                )

############# Method with gradients #####################

def compute_partial_derivatives_delta(events_all, times_all, delta_s, delta_c, positive_obs_in_groups_idx):
    """
    Compute partial derivatives of the kernel functions with respect to delta_s and delta_c.
    This function mirrors the structure of compute_kernels, but computes the partials instead.

    Args:
        events_all: Binary events array (1D).
        times_all: 2D array with continuous time in row 0 and discrete time in row 1 (shape: 2 x N).
        delta_s: Decay parameter for discrete time kernel.
        delta_c: Decay parameter for continuous time kernel.
        positive_obs_in_groups_idx: List of indices arrays, each corresponding to an individual's 
                                  subset of positive events with at least two positive events.

    Returns:
        d_s: Partial derivative of s w.r.t. delta_s (1D).
        d_c: Partial derivative of c w.r.t. delta_c (1D).
    """
    n_total = len(events_all)
    d_s = np.zeros(n_total)
    d_c = np.zeros(n_total)

    continuous_time = times_all[0, :]
    discrete_time = times_all[1, :]

    # For each group (individual) with multiple positive events
    for group_idx in positive_obs_in_groups_idx:
        individual_continuous_time = continuous_time[group_idx]
        individual_discrete_time = discrete_time[group_idx]

        start_idx = group_idx[0]
        for j in range(1, len(group_idx)):
            past_indices = np.arange(0, j)
            time_diff = individual_continuous_time[j] - individual_continuous_time[past_indices]
            discrete_diff = individual_discrete_time[j] - individual_discrete_time[past_indices]

            # Partial derivative wrt delta_c:
            # c_i(t_j) = sum_{k<j} exp(-delta_c * time_diff_kj)
            # d/d(delta_c) c_i(t_j) = sum_{k<j} [-time_diff_kj * exp(-delta_c * time_diff_kj)]
            d_c[start_idx + j] = np.sum(-time_diff * np.exp(-delta_c * time_diff))

            # Partial derivative wrt delta_s:
            # s_i(j) = sum_{k<j} exp(-delta_s * discrete_diff_kj)
            # d/d(delta_s) s_i(j) = sum_{k<j} [-(discrete_diff_kj) * exp(-delta_s * discrete_diff_kj)]
            d_s[start_idx + j] = np.sum(-(discrete_diff) * np.exp(-delta_s * discrete_diff))

    return d_s, d_c


def compute_gradients(params, events_all, times_all, covariates_all, positive_obs_in_groups_idx,
                      time_types, compute_kernels):
    """
    Compute the gradient of the log-likelihood with respect to the model parameters.

    Parameters:
        params: Current parameter estimates (1D array).
                Order: [alpha, gamma..., beta_c, delta_c, beta_s, delta_s]
        events_all: Binary event outcomes for all observations (1D).
        times_all: 2D array with continuous (row 0) and discrete (row 1) times.
        covariates_all: Covariates matrix (N x M).
        individuals_all: Array of individual identifiers (1D).
        positive_obs_in_groups_idx: List of index arrays for individuals with multiple positive events.
        time_types: List specifying which types ('continuous', 'discrete') are included.
        compute_kernels: Function that computes s and c arrays.

    Returns:
        grad: 1D array containing the gradient w.r.t. [alpha, gamma..., beta_c, delta_c, beta_s, delta_s].
    """
    # Extract parameters
    alpha = params[0]
    gamma = params[1:-4]
    beta_c = params[-4]
    delta_c = params[-3]
    beta_s = params[-2]
    delta_s = params[-1]

    # Compute kernels
    s, c = compute_kernels(
        events_all=events_all,
        times_all=times_all,
        delta_s=delta_s,
        delta_c=delta_c,
        positive_obs_in_groups_idx=positive_obs_in_groups_idx
        )

    # Compute linear predictor
    linear_pred = alpha + np.dot(covariates_all, gamma)
    if 'continuous' in time_types:
        linear_pred += beta_c * c
    if 'discrete' in time_types:
        linear_pred += beta_s * s

    # Compute probabilities
    p = expit(linear_pred)

    # Residuals
    r = events_all - p

    # Gradients w.r.t. alpha and gamma
    g_alpha = np.sum(r)
    g_gamma = covariates_all.T @ r

    # Gradient w.r.t. beta_c
    if 'continuous' in time_types:
        g_beta_c = np.sum(r * c)
    else:
        g_beta_c = 0.0

    # Gradient w.r.t. beta_s
    if 'discrete' in time_types:
        g_beta_s = np.sum(r * s)
    else:
        g_beta_s = 0.0

    # Compute partial derivatives of kernels w.r.t delta_s and delta_c
    d_s, d_c = compute_partial_derivatives_delta(
        events_all=events_all,
        times_all=times_all,
        delta_s=delta_s,
        delta_c=delta_c,
        positive_obs_in_groups_idx=positive_obs_in_groups_idx
        )

    # Gradient w.r.t. delta_c
    # dLL/d(delta_c) = sum_{i,j} (y_{ij}-p_{ij}) * beta_c * d(c_i(t_j))/d(delta_c)
    if 'continuous' in time_types:
        g_delta_c = np.sum(r * beta_c * d_c)
    else:
        g_delta_c = 0.0

    # Gradient w.r.t. delta_s
    # dLL/d(delta_s) = sum_{i,j} (y_{ij}-p_{ij}) * beta_s * d(s_i(j))/d(delta_s)
    if 'discrete' in time_types:
        g_delta_s = np.sum(r * beta_s * d_s)
    else:
        g_delta_s = 0.0

    # Combine all gradients into a single vector
    grad = np.concatenate(([g_alpha], g_gamma, [g_beta_c, g_delta_c, g_beta_s, g_delta_s]))

    return grad


##### Parallel implementation #####

def compute_kernels_for_individual(args):
    """
    Computes the kernels s and c for a single individual.
    """
    individual_continuous_time, individual_discrete_time, delta_s, delta_c, indices = args

    n_events = len(individual_continuous_time)
    s_values = np.zeros(n_events)
    c_values = np.zeros(n_events)

    for j in range(1, n_events):
        time_diff = individual_continuous_time[j] - individual_continuous_time[:j]
        c_values[j] = np.sum(np.exp(-delta_c * time_diff))

        discrete_diff = individual_discrete_time[j] - individual_discrete_time[:j]
        s_values[j] = np.sum(np.exp(-delta_s * discrete_diff))

    return indices, s_values, c_values


class ParallelSelfExcitingLogisticRegression(SelfExcitingLogisticRegression):
    """ 
    Logistic regression with self-exciting kernels for continuous and discrete time.

    This class uses parallel processing to compute the kernels for all individuals.
    """
    def __init__(self, max_iter=100, tol=1e-6, rng=default_rng(1),
                 time_types=['continuous', 'discrete'], n_jobs=1,
                 ignore_errors=True):
        super().__init__(max_iter, tol, rng, time_types, ignore_errors)
        self.n_jobs=n_jobs

    def compute_kernels(self, events_all, times_all, delta_s, delta_c, positive_obs_in_groups_idx):
        """
        Computes the discrete and continuous kernels s and c for all individuals using parallel processing.
        """
        n_total = len(events_all)
        s = np.zeros(n_total)
        c = np.zeros(n_total)

        # Split continuous and discrete time
        continuous_time = times_all[0, :]
        discrete_time = times_all[1, :]

        # Prepare arguments for each individual
        args_list = [
            (
                list(continuous_time[group_idx]), # time data
                list(discrete_time[group_idx]), # time data
                delta_s, # decay parameters
                delta_c, # decay parameters
                list(group_idx) # indices to reasemble the results after computation
            )
            for group_idx in positive_obs_in_groups_idx
        ]
        # Use joblib for efficient parallel computation
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(compute_kernels_for_individual)(args) for args in args_list
        )

        if results is None:
            raise ValueError("Parallel computation failed.")

        # Collect results and fill s and c arrays
        for indices, s_values, c_values in results:
            s[indices] = s_values
            c[indices] = c_values

        return s, c


def likelihood_ratio_test(
        submodel: SelfExcitingLogisticRegression,
        fullmodel: SelfExcitingLogisticRegression,
        alpha=0.05
        ):
    """
    Perform a likelihood ratio test to compare two nested models.

    Returns a tuple with three elements:
    - A boolean indicating whether the null hypothesis is rejected (whether to reject the submodel).
    - The p-value of the test.
    - The test statistic.
    """
    if submodel.n_params_ is None or fullmodel.n_params_ is None or submodel.n_params_ >= fullmodel.n_params_:
        raise ValueError("The submodel must have fewer parameters than the full model.")

    if submodel.nll_ is None or fullmodel.nll_ is None:
        raise ValueError("Both models must be fitted before performing the test.")

    # Compute log-likelihoods
    ll_submodel = - submodel.nll_
    ll_fullmodel = - fullmodel.nll_

    # Compute test statistic
    test_stat = 2 * (ll_fullmodel - ll_submodel)

    # Compute p-value
    p_value = 1 - chi2.cdf(test_stat, fullmodel.n_params_ - submodel.n_params_)

    return p_value < alpha, p_value, test_stat