from scipy import stats
import numpy as np


def calculate_sample_size_corrected(baseline_conversion, mde_relative, power, alpha):
    # Conversion rate of the control and treatment groups
    p1 = baseline_conversion
    p2 = baseline_conversion * (1 + mde_relative)

    # Average conversion rate and difference between the two conversion rates
    p_bar = (p1 + p2) / 2
    delta = p2 - p1

    # Z-scores for alpha and power
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    # Calculating sample size
    n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_power * np.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / delta ** 2

    return int(np.ceil(n))


baseline_conversion = 0.1
mde_relative = 0.2
power = 0.8
alpha = 0.05


sample_size_per_group_corrected = calculate_sample_size_corrected(baseline_conversion, mde_relative, power, alpha)
print(sample_size_per_group_corrected)