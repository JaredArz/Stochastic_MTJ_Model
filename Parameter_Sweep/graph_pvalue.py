import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter


df = pd.read_csv("MTJ_Param_Results.csv")

chi2_df = df["chi2Data_path"]

config_nums = []
p_values = []
for chi2_path in chi2_df:
  config_nums.append(chi2_path.split("_")[-1][:-4])
  
  with open(chi2_path) as f:
    chi2 = float(f.read())
    p_values.append(1 - stats.chi2.cdf(chi2, 256))

unique_pval = list(Counter(p_values).keys())
pval_counts = list(Counter(p_values).values())

# plt.bar(unique_pval, pval_counts)
bins = np.linspace(0, 1, num=100)
plt.hist(p_values, bins=bins)
plt.xticks(rotation=90)
plt.xlabel("P-Value")
plt.ylabel("Counts")
plt.title("Cluster Runs: P-Value")
plt.show()
