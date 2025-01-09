import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # Logistic sigmoid function
from numpy.random import default_rng
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression



def compute_linear_probs(alpha, gamma, beta_s, beta_c, s, c, covariates_all, time_types):
    # Compute linear predictor
    linear_pred = alpha + np.dot(covariates_all, gamma)
    if 'continuous' in time_types:
        linear_pred += beta_s * s 
    if 'discrete' in time_types:
        linear_pred += beta_c * c

    probs = expit(linear_pred)

    return probs


def compute_time_series_groups_idx(individuals_all, events_all):
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


def estimate_initial_params(events_all, covariates_all, method='logistic'):
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


def compute_partial_derivatives_delta(events_all, times_all, delta_s, delta_c, positive_obs_in_groups_idx):
    """
    Compute partial derivatives of the kernel functions with respect to delta_s and delta_c.
    This function mirrors the structure of compute_kernels_sparse, but computes the partials instead.

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
        # Extract data for the current individual
        individual_continuous_time = continuous_time[group_idx]
        individual_discrete_time = discrete_time[group_idx]

        # Compute kernels for the current individual
        for event_number, row_idx in enumerate(group_idx[1:], start=1):
            past_indices = np.arange(0, event_number)

            # Continuous kernel: c_ij
            time_diff = individual_continuous_time[event_number] - individual_continuous_time[past_indices]
            # Discrete kernel: s_ij
            discrete_diff = individual_discrete_time[event_number] - individual_discrete_time[past_indices]

            # Partial derivative wrt delta_c:
            # c_i(t_j) = sum_{k<j} exp(-delta_c * time_diff_kj)
            # d/d(delta_c) c_i(t_j) = sum_{k<j} [-time_diff_kj * exp(-delta_c * time_diff_kj)]
            d_c[row_idx] = np.sum(-time_diff * np.exp(-delta_c * time_diff))

            # Partial derivative wrt delta_s:
            # s_i(j) = sum_{k<j} exp(-delta_s * discrete_diff_kj)
            # d/d(delta_s) s_i(j) = sum_{k<j} [-(discrete_diff_kj) * exp(-delta_s * discrete_diff_kj)]
            d_s[row_idx] = np.sum(-(discrete_diff) * np.exp(-delta_s * discrete_diff))

    return d_s, d_c


def compute_gradients(params:np.ndarray, events_all, times_all, covariates_all, positive_obs_in_groups_idx,
                      time_types, kernel_fun):
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
        kernel_fun: Function that computes s and c arrays.

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
    s, c = kernel_fun(events_all=events_all, times_all=times_all, delta_s=delta_s,
                           delta_c=delta_c, positive_obs_in_groups_idx=positive_obs_in_groups_idx
                           )
    p = compute_linear_probs(alpha=alpha, gamma=gamma, beta_s=beta_s, beta_c=beta_c, s=s,
                             c=c, covariates_all=covariates_all, time_types=time_types
                             )
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


class SelfExcitingLogisticRegression:
    """ 
    Logistic regression with self-exciting kernels for continuous and discrete time. 
    """
    def __init__(self, max_iter=100, tol=1e-9, rng=default_rng(1),
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
        self.shapes_checked=False

    def check_array_shapes(self, events_all, times_all, covariates_all, individuals_all):
        if not self.shapes_checked:
            n_obs = times_all.shape[1]
            if len(individuals_all) != n_obs or len(events_all) != n_obs or len(covariates_all) != n_obs:
                raise ValueError(
                    f"Lengths of individuals ({len(individuals_all)}), events ({len(events_all)}), coviraites ({len(covariates_all)}) and times ({n_obs}) do not match"
                )
            else:
                self.shapes_checked = True


    def compute_kernels_single_timepoint(self, time_point, past_events, past_times, delta_s, delta_c):
        """
        Compute the kernels s_ij and c_ij for a single time point for a single individual.

        Args:
            time_point: Time point to compute kernels for.
            past_events: Events prior to the time point.
            past_times: 2D array with continuous time (row 0) and discrete time (row 1) of past events.
            delta_s: Decay parameter for discrete time.
            delta_c: Decay parameter for continuous time.
        """

        # check if we have any past events
        if len(past_events) == 0:
            return 0, 0
        
        past_pos_events = past_events > 0

        # Split continuous and discrete time
        past_continuous_times = past_times[0, :]
        past_discrete_times = past_times[1, :]

        # Continuous kernel: c_ij
        time_diff = time_point - past_continuous_times
        filtered_time_diff = time_diff[past_pos_events]
        c = np.sum(np.exp(-delta_c * filtered_time_diff))

        # Discrete kernel: s_ij
        discrete_diff = max(past_discrete_times) - past_discrete_times
        filtered_discrete_diff = discrete_diff[past_pos_events]
        s = np.sum(np.exp(-delta_s * filtered_discrete_diff))

        return s, c


    def compute_kernels_sparse(self, events_all, times_all, delta_s, delta_c, positive_obs_in_groups_idx):
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

            # Compute kernels for the current individual
            for event_number, row_idx in enumerate(group_idx[1:], start=1):
                past_indices = np.arange(0, event_number)

                # Continuous kernel: c_ij
                time_diff = individual_continuous_time[event_number] - individual_continuous_time[past_indices]
                c[row_idx] = np.sum(np.exp(-delta_c * time_diff))

                # Discrete kernel: s_ij
                discrete_diff = individual_discrete_time[event_number] - individual_discrete_time[past_indices]
                s[row_idx] = np.sum(np.exp(-delta_s * discrete_diff))

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
    

    def log_likelihood(self, params: np.ndarray, events_all: np.ndarray, times_all: np.ndarray,
                       covariates_all:np.ndarray, positive_obs_in_groups_idx:list, negative=True):
        """
        Log-likelihood for the logistic regression with self-exciting kernels.

        Negative log-likelihood is returned if negative is set to True.
        """
        
        if type(params) is not np.ndarray:
            raise ValueError(f"Parameters must be a numpy array and not {type(params)}")

        alpha = params[0]
        gamma = params[1:-4]
        beta_c = params[-4]
        delta_c = params[-3]
        beta_s = params[-2]
        delta_s = params[-1]

        # Compute kernels
        s, c = self.compute_kernels_sparse(events_all=events_all, times_all=times_all,
                                    delta_s=delta_s, delta_c=delta_c,
                                    positive_obs_in_groups_idx=positive_obs_in_groups_idx,
        )
        probs = compute_linear_probs(alpha=alpha, gamma=gamma, beta_s=beta_s, beta_c=beta_c,
                                    s=s, c=c, covariates_all=covariates_all,
                                    time_types=self.time_types,
        )

        # log-likelihood
        ll = np.sum(
            events_all * np.log(probs + 10**(-8)) +
            (1 - events_all) * np.log(1 - probs + 10**(-8))
        )

        if negative: 
           return -ll 

        return ll


    def fit(self, events_all, times_all, covariates_all, individuals_all):
        """
        fits the logistic regression self-exciting model.
        """
        self.check_array_shapes(events_all, times_all, covariates_all, individuals_all)

        positive_obs_in_groups_idx = compute_time_series_groups_idx(individuals_all, events_all)

        n_covariates = covariates_all.shape[1]

        if 'continuous' in self.time_types or 'discrete' in self.time_types:
            #initial_params = estimate_initial_params(events_all, covariates_all, method='logistic')
            initial_params = self.rng.uniform(-1, 1, size=1 + n_covariates + 4)  # alpha, gamma..., beta_c, delta_c, beta_s, delta_s
            print(initial_params)

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
                    positive_obs_in_groups_idx=positive_obs_in_groups_idx,
                    negative=True
                )

            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': self.max_iter},
                bounds=bounds,
                tol=self.tol
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
            params = np.array(list(clf.intercept_) + list(clf.coef_[0]) + [0, 0, 0, 0])
            self.set_params(params)
            self.success_ = True
            self.nll_= self.log_likelihood(
                    params=params,
                    events_all=events_all,
                    times_all=times_all,
                    covariates_all=covariates_all,
                    positive_obs_in_groups_idx=positive_obs_in_groups_idx
            )


    def predict_probas(self, time_points, event_times, covariates, events, return_kernels=False):
        """
        Predict probabilities for binary outcomes at given time points for a single individual.

        Args:
            time_points: Array of continuous time points to predict at.
            event_times: 2D array with continuous time (row 0) and discrete time (row 1) of the observed events.
                        Difference between time_points and event_times is that event_times are the (discrete and cont)
                        time points from observations, while time_points are the time points we want to predict at.
            covariates: Covariates matrix (N x M) for N events and M covariates.
            events: Binary event outcomes for all observations of the individual (1D).
            return_kernels: If True, the kernel values are also returned.

        """

        if self.n_params_ is None:
            raise ValueError("The model is not fitted yet.")

        alpha, gamma, beta_c, delta_c, beta_s, delta_s = self.load_params()

        if alpha is None or gamma is None or beta_s is None or beta_c is None or delta_s is None or delta_c is None:
            raise ValueError("One or more model parameters None. Model have been fitted, but not successfully.")

        probas = np.zeros(len(time_points))
        c = np.zeros(len(time_points))
        s = np.zeros(len(time_points))

        for idx, time_point in enumerate(time_points):
            # If the time point is before the first event, return the baseline probability
            past_event_indices = event_times[0,:]< time_point # obs continuous time prior to time_point
            if not np.any(past_event_indices):
                probas[idx] = expit(alpha + np.dot(covariates[0], gamma))

            else:
                past_times = event_times[:, past_event_indices]
                past_events = events[past_event_indices]
                current_covariate = covariates[past_event_indices][-1] # only the current features are needed

                # Compute kernels
                s_val, c_val = self.compute_kernels_single_timepoint(time_point=time_point, past_events=past_events,
                                                            past_times=past_times, delta_s=delta_s, delta_c=delta_c,
                )

                proba = compute_linear_probs( alpha=alpha, gamma=gamma, beta_s=beta_s, beta_c=beta_c,
                    s=s_val, c=c_val, covariates_all=current_covariate, time_types=self.time_types
                )

                probas[idx] = proba
                c[idx] = c_val
                s[idx] = s_val

        if return_kernels:
            return probas, c, s
        else:
            return probas

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
    
    def save_model(self, filename):
        """
        Save the model to a file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load_model(self, filename):
        """
        Load the model from a file.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        
        self.__dict__.update(model.__dict__)



    def __str__(self):
        return f"SelfExcitingLogisticRegression(time_types={self.time_types})"



############ Method with gradients #####################

class SelfExcitingLogisticRegressionWithGrad(SelfExcitingLogisticRegression):

    def fit(self, events_all, times_all, covariates_all, individuals_all):
        """
        Fits the logistic regression model using the gradient for optimization.
        """

        positive_obs_in_groups_idx = compute_time_series_groups_idx(individuals_all, events_all)

        if 'continuous' in self.time_types and 'discrete' in self.time_types:
            # Initialize data

            # Set initial parameters
            n_covariates = covariates_all.shape[1]

            initial_params = estimate_initial_params(events_all, covariates_all, method='logistic') 
            print(f'Initial parameters: {initial_params}')

            # Define parameter bounds
            alpha_bound = [(None, None)]
            gamma_bound = [(None, None) for _ in range(n_covariates)]
            beta_c_bound = [(None, None)]
            delta_c_bound = [(None, None)]  # Ensure positive decay
            beta_s_bound = [(None, None)]
            delta_s_bound = [(None, None)]  # Ensure positive decay

            bounds = alpha_bound + gamma_bound + beta_c_bound + delta_c_bound + beta_s_bound + delta_s_bound

            # Objective function: log-likelihood
            def objective(params: np.ndarray):
                return self.log_likelihood(
                    params=params,
                    events_all=events_all,
                    times_all=times_all,
                    covariates_all=covariates_all,
                    positive_obs_in_groups_idx=positive_obs_in_groups_idx
                )

            # Gradient function
            def gradient(params: np.ndarray):
                return compute_gradients(
                    params=params,
                    events_all=events_all,
                    times_all=times_all,
                    covariates_all=covariates_all,
                    positive_obs_in_groups_idx=positive_obs_in_groups_idx,
                    time_types=self.time_types,
                    compute_kernels=self.compute_kernels_sparse
                )

            # Use scipy.optimize.minimize with gradients
            result = minimize(
                objective,
                initial_params,
                method='trust-constr',
                jac=gradient,  # Pass the gradient function here
                options={'maxiter': self.max_iter, 'disp': True},
                bounds=bounds,
                tol=self.tol
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
            params = np.array(clf.intercept_.tolist() + clf.coef_[0].tolist() + [0, 0, 0, 0])
            self.set_params(params)
            self.success_ = True
            self.nll_= self.log_likelihood(
                    params=params,
                    events_all=events_all,
                    times_all=times_all,
                    covariates_all=covariates_all,
                    positive_obs_in_groups_idx=positive_obs_in_groups_idx
                )


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