import pandas as pd
import numpy as np
import statsmodels.stats.power as smp

with open("/Users/svitlana/Desktop/data3.csv", 'r') as file:
    headers = file.readline().strip().split(',')
headers = [header.replace('\"', '').replace('\t', ' ').strip() for header in headers]
df = pd.read_csv("/Users/svitlana/Desktop/data3.csv", skiprows=1, header=None)
df = df[0].str.split(',', expand=True)
df.columns = headers
df = df.replace('\"', '', regex=True).replace('\t', ' ', regex=True).replace(r'^\s*$', 0, regex=True)

df.drop(df.columns[[1, 2, 3]], axis=1, inplace=True)

for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

grouped_df = df.groupby(df.columns[0]).sum()

day = 14
increase_percent = 15
arpu_values = df.columns[day+1]

mean_arpu = np.mean(df[arpu_values].dropna())
std_arpu = np.std(df[arpu_values].dropna(), ddof=1)

alpha = 0.05
power = 0.8
effect_size = (mean_arpu * increase_percent / 100) / std_arpu

sample_size = smp.TTestIndPower().solve_power(effect_size=effect_size,
                                              power=power,
                                              alpha=alpha,
                                              ratio=1,
                                              alternative='two-sided')

print("Середнє ARPU:", mean_arpu)
print("Стандартне відхилення ARPU:", std_arpu)
print("Необхідний розмір вибірки:", int(np.ceil(sample_size)))