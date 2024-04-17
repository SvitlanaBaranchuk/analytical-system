import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')


data = {
    'split_id': ['test1', 'test2', 'test3', 'test4'],
    'N_A': [47751, 4541, 23999, 34400],
    'C_A': [10576, 159, 16310, 26097],
    'N_B': [48264, 4815, 23978, 34811],
    'C_B': [11023, 199, 16128, 23429]
}
df = pd.DataFrame(data)


def bayesian_probability(successes_a, trials_a, successes_b, trials_b, sim_size=100000):
    alpha_a = successes_a + 1
    beta_a = trials_a - successes_a + 1
    alpha_b = successes_b + 1
    beta_b = trials_b - successes_b + 1

    samples_a = np.random.beta(alpha_a, beta_a, sim_size)
    samples_b = np.random.beta(alpha_b, beta_b, sim_size)

    probability_a_better_than_b = np.mean(samples_a > samples_b)
    probability_b_better_than_a = np.mean(samples_b > samples_a)

    return round(probability_a_better_than_b, 3), round(probability_b_better_than_a, 3)


def loss_function(successes_a, trials_a, successes_b, trials_b, N_mc=100000):
    alpha_a = successes_a + 1
    beta_a = trials_a - successes_a + 1
    alpha_b = successes_b + 1
    beta_b = trials_b - successes_b + 1

    samples_a = np.random.beta(alpha_a, beta_a, N_mc)
    samples_b = np.random.beta(alpha_b, beta_b, N_mc)

    prob_b_better_a = np.mean(samples_b > samples_a)

    loss_a = np.mean(np.maximum(0, samples_b - samples_a))
    loss_b = np.mean(np.maximum(0, samples_a - samples_b))

    return round(loss_a, 4), round(loss_b, 4), prob_b_better_a


baseline_cr = 3.5
critical_cr = 3
e = (baseline_cr / 100) * (critical_cr / 100)

df[['L(A)', 'L(B)', 'P(B>A)']] = df.apply(
    lambda row: pd.Series(loss_function(row['C_A'], row['N_A'], row['C_B'], row['N_B'])), axis=1)


def verdict(l_a, l_b, e):
    if abs(l_a) < e and abs(l_b) < e:
        return 'no difference'
    elif abs(l_a) < e:
        return '<A>'
    elif abs(l_b) < e:
        return '<B>'
    else:
        return 'continue experiment'


df['Verdict'] = df.apply(lambda row: verdict(row['L(A)'], row['L(B)'], e), axis=1)
df['P(A>B)'], df['P(B>A)'] = zip(
    *df.apply(lambda row: bayesian_probability(row['C_A'], row['N_A'], row['C_B'], row['N_B']), axis=1))
df = df[['split_id', 'N_A', 'C_A', 'N_B', 'C_B', 'P(A>B)', 'P(B>A)', 'L(A)', 'L(B)', 'Verdict']]

alpha_a = df['C_A'][0] + 1
beta_a = df['N_A'][0] - df['C_A'][0] + 1
alpha_b = df['C_B'][0] + 1
beta_b = df['N_B'][0] - df['C_B'][0] + 1

samples_a = np.random.beta(alpha_a, beta_a, 100000)
samples_b = np.random.beta(alpha_b, beta_b, 100000)

# Графік
mean_a = np.mean(samples_a)
std_a = np.std(samples_a)
mean_b = np.mean(samples_b)
std_b = np.std(samples_b)

sns.histplot(samples_a, color='orange', kde=True, stat='density', label='conv_a')
sns.histplot(samples_b, color='blue', kde=True, stat='density', label='conv_b')
plt.axvline(mean_a, color='red', linestyle='dashed')
plt.axvline(mean_b, color='blue', linestyle='dashed')
plt.axvline(mean_a + std_a, color='red', linestyle='dotted')
plt.axvline(mean_a - std_a, color='red', linestyle='dotted')
plt.axvline(mean_b + std_b, color='blue', linestyle='dotted')
plt.axvline(mean_b - std_b, color='blue', linestyle='dotted')
plt.legend()
plt.xlabel('Конверсії (стандартні відхилення)')
plt.ylabel('Щільність')
plt.title('Апостеріорні розподіли конверсій у групі A і B')
plt.show()

print(df)