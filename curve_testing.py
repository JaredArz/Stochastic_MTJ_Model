import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import rel_entr


with open("curve_testing_pdf.pkl", "rb") as file: 
  pdf_info = pickle.load(file) 

countData_path = "results/countData/countData_0.txt"
with open(countData_path, "rb") as file: 
  counts_goodfit = []
  for line in file:
    counts_goodfit.append(float(line.strip()))
  
countData_path = "results/countData/countData_6.txt"
with open(countData_path, "rb") as file: 
  counts_badfit = []
  for line in file:
    counts_badfit.append(float(line.strip()))

xxis = pdf_info["xxis"]
exp_pdf = pdf_info["exp_pdf"]

# RMSE test
rmse_goodfit = np.sqrt(((counts_goodfit - exp_pdf) ** 2).mean())
rmse_badfit = np.sqrt(((counts_badfit - exp_pdf) ** 2).mean())
print("RMSE (good fit):", rmse_goodfit)
print("RMSE (bad fit) :", rmse_badfit)

# Chi2 test
chi2_goodfit = 0
chi2_badfit = 0
for i in range(len(exp_pdf)):
  chi2_goodfit += ((counts_goodfit[i]-exp_pdf[i])**2)/exp_pdf[i]
  chi2_badfit += ((counts_badfit[i]-exp_pdf[i])**2)/exp_pdf[i]
print("\nChi2 (good fit):", chi2_goodfit)
print("Chi2 (bad fit) :", chi2_badfit)

# KL-Divergence test
kl_div_goodfit = sum(rel_entr(counts_goodfit, exp_pdf))
kl_div_badfit = sum(rel_entr(counts_badfit, exp_pdf))
print("\nKL-Divergence (good fit):", kl_div_goodfit)
print("KL-Divergence (bad fit) :", kl_div_badfit)

# CDF_MSE test
exp_cdf = np.cumsum(exp_pdf)
goodfit_cdf = np.cumsum(counts_goodfit)
badfit_cdf = np.cumsum(counts_badfit)
cdf_mse_goodfit = ((goodfit_cdf - exp_cdf) ** 2).mean()
cdf_mse_badfit = ((badfit_cdf - exp_cdf) ** 2).mean()
print("\nCDF MSE (good fit):", cdf_mse_goodfit)
print("CDF MSE (bad fit) :", cdf_mse_badfit)

plt.subplot(1,2,1)
plt.plot(xxis, counts_goodfit, color="green", label="Good Fit")
plt.plot(xxis, counts_badfit, color="red", label="Bad Fit")
plt.plot(xxis, exp_pdf,'k--')
plt.xlabel("Generated Number")
plt.ylabel("Normalized")
plt.title("PDF Comparison")
plt.legend()

plt.subplot(1,2,2)
plt.plot(xxis, goodfit_cdf, color="green", label="Good Fit")
plt.plot(xxis, badfit_cdf, color="red", label="Bad Fit")
plt.plot(xxis, exp_cdf,'k--')
plt.title("CDF Comparison")
plt.legend()
plt.show()