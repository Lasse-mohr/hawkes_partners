
from scipy.special import expit
import numpy as np

def create_fake_seq(n_individuals, max_events_per_individual, true_alpha,
                    true_gamma, true_beta_c, true_delta_c, true_beta_s, true_delta_s,
                    seed=1
                    ):
    rng = np.random.default_rng(seed=seed)

    # Generate synthetic data
    n_events_per_individual = rng.integers(1, max_events_per_individual, size=n_individuals)

    continuous_time = []
    discrete_time = []
    events = []
    covariates = []
    individuals = []

    # Generate synthetic data for each individual
    for i, n_events in enumerate(n_events_per_individual):
        # Generate continuous and discrete times for this individual
        continuous_times = np.sort(rng.uniform(0, 1, n_events))
        discrete_times = np.arange(0, n_events)

        # Generate covariates for this individual
        age = np.sort(rng.uniform(0, 1, n_events).reshape(-1, 1), axis=0)
        sex = np.repeat(rng.choice([0, 1]), n_events).reshape(-1, 1)
        covariates_individual = np.hstack([age, sex])

        s = np.zeros(n_events)
        c = np.zeros(n_events)

        events_individual = np.zeros(n_events)
        events_individual[0] = rng.binomial( 1, expit(true_alpha) )

        for j in range(1, n_events):
            past_pos_events = events_individual[:j] > 0
            discrete_diff = discrete_times[j] - discrete_times[:j]
            continuous_diff = continuous_times[j] - continuous_times[:j]

            s[j] = np.sum(
                np.exp(-true_delta_s * discrete_diff)[past_pos_events]
                )
            c[j] = np.sum(
                np.exp(-true_delta_c * continuous_diff)[past_pos_events]
                )
            event_prob = expit(
                true_alpha + true_beta_s * s[j] + true_beta_c * c[j] + np.dot(covariates_individual[j], true_gamma)
                )

            events_individual[j] = rng.choice([0, 1], p=[1-event_prob, event_prob])

        individual = [i] * n_events
        # Store this individual's data
        continuous_time.append(continuous_times)
        discrete_time.append(discrete_times)
        events.append(events_individual)
        covariates.append(covariates_individual)
        individuals.append(individual)

    # Combine all individuals' data
    continuous_time = np.concatenate(continuous_time)
    discrete_time = np.concatenate(discrete_time)
    events = np.concatenate(events)
    individuals = np.concatenate(individuals)

    print(f'Proportion of nonzero events: {np.sum(events)/len(events)}')

    covariates = np.vstack(covariates)
    covariates = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0)

    # Combine continuous and discrete times into a 2xN array
    times = np.vstack([continuous_time, discrete_time])

    return times, events, covariates, individuals