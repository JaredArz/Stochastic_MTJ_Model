import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv("MTJ_Param_Results.csv")
chi2_df = df["chi2Data_path"]

config_nums = []
p_values = []
for chi2_path in chi2_df:
  config_nums.append(chi2_path.split("_")[-1][:-4])
  
  with open(chi2_path) as f:
    chi2 = float(f.read())
    p_values.append(1 - stats.chi2.cdf(chi2, 256))

plt.bar(config_nums, p_values)
plt.show()

# plt.subplot(1,2,1)
# b1 = plt.bar(x_axis-0.2, wine_training, 0.4, label='Training')
# b2 = plt.bar(x_axis+0.2, wine_testing, 0.4, label='Testing')
# plt.title('Wine', fontweight='bold', size=10)
# plt.xlabel('Version', fontweight='bold', size=10)
# plt.ylabel('Accuracy', fontweight='bold', size=10)
# plt.xticks(x_axis, versions)
# for bars in [b1, b2]:
#   for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, "{:.2f}".format(yval), ha="center")

# plt.subplot(1,2,2)
# b3 = plt.bar(x_axis-0.2, iris_training, 0.4)
# b4 = plt.bar(x_axis+0.2, iris_testing, 0.4)
# plt.title('Iris', fontweight='bold', size=10)
# plt.xlabel('Version', fontweight='bold', size=10)
# plt.ylabel('Accuracy', fontweight='bold', size=10)
# plt.xticks(x_axis, versions)
# for bars in [b3, b4]:
#   for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, "{:.2f}".format(yval), ha="center")

# plt.figlegend(loc="upper right")
# plt.show()