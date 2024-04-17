import numpy as np

def sequential_ab_test(N, conversion_rates, seed=None):
    if seed is not None:
        np.random.seed(seed)

    control_outcomes = np.random.binomial(1, conversion_rates[0], N)
    test_outcomes = np.random.binomial(1, conversion_rates[1], N)

    T = 0  # Successes in treatment group
    C = 0  # Successes in control group
    i = 0  # Iteration counter

    while i < N and T + C < N:
        T += test_outcomes[i]
        C += control_outcomes[i]

        if T - C >= 2 * np.sqrt(N):
            return {'winner': 'test version'}
        i += 1

    if T + C == N:
        return {'winner': 'no winner'}

    return {'winner': 'test not conclusive'}


N = 10000
control_conversion = 0.1
treatment_conversion = 0.12

result = sequential_ab_test(N, conversion_rates=(control_conversion, treatment_conversion), seed=42)
print(result)


