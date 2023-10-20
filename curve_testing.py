import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import rel_entr
from mtj_RL_dev import mtj_run

SAMPLES = 100000

def get_params(version):
  if version == "NoNorm_Episode1":
    # No Norm; Episode 1
    alpha = 0.01
    Ki = 0.0002
    Ms = 300000.0
    Rp = 500.0
    TMR = 0.3
    eta = 0.26508163213729863
    J_she = 1000000000000.0
    t_pulse = 3.790034800767898e-08
    t_relax = 3.790034800767898e-08
    d = 3e-09
    tf = 1.1e-09

  elif version == "NoNorm_Episode4":
    # No Norm; Episode 4
    alpha = 0.035038104653358465
    Ki = 0.0002
    Ms = 300000.0
    Rp = 500.0
    TMR = 0.3
    eta = 0.1
    J_she = 1000000000000.0
    t_pulse = 7.5e-08
    t_relax = 7.5e-08
    d = 3e-09
    tf = 1.1e-09

  elif version == "NoNorm_Episode14":
    # No Norm; Episode 14
    alpha = 0.022418085932731632
    Ki = 0.0002
    Ms =  300000.0
    Rp = 50000.0
    TMR = 0.3
    eta = 0.1
    J_she = 1000000000000.0
    t_pulse = 5.169894516468048e-08
    t_relax = 5.169894516468048e-08
    d = 3e-09
    tf = 1.1e-09

  elif version == "NoNorm_Episode56":
    # No Norm; Episode 56
    alpha = 0.058964042067527776
    Ki = 0.0002
    Ms = 570791.1014556885
    Rp = 500.0
    TMR = 0.3
    eta = 0.1
    J_she = 1000000000000.0
    t_pulse = 7.5e-08
    t_relax = 7.5e-08
    d = 3e-09
    tf = 1.1e-09

  elif version == "Norm_Episode43":
    # Norm; Episode 43
    alpha = 0.01
    Ki = 0.0002
    Ms = 300000.0
    Rp = 50000.0
    TMR = 0.3
    eta = 0.589673674106598
    J_she = 456989129185.6766
    t_pulse = 4.097234883904457e-08
    t_relax = 4.097234883904457e-08
    d = 3e-09
    tf = 1.1e-09
  
  elif version == "Norm_Episode87":
    # Norm; Episode 87
    alpha = 0.01
    Ki = 0.0002
    Ms = 300000.0
    Rp = 500.0
    TMR = 0.3
    eta = 0.8
    J_she = 10000000000.0
    t_pulse = 1.840546451508999e-09
    t_relax = 1.840546451508999e-09
    d = 3e-09
    tf = 1.1e-09

  elif version == "Norm_Episode89":
    # Norm; Episode 89
    alpha = 0.01
    Ki = 0.0006466779708862305
    Ms = 300000.0
    Rp = 500.0
    TMR = 2.476738339662552
    eta = 0.1
    J_she = 10000000000.0
    t_pulse = 5e-10
    t_relax = 5e-10
    d = 3e-09
    tf = 1.1e-09
  
  return alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax


if __name__ == "__main__":
  # versions = ["NoNorm_Episode1", "NoNorm_Episode4", "NoNorm_Episode14", "NoNorm_Episode56", 
  #             "Norm_Episode43", "Norm_Episode87", "Norm_Episode89"]
  versions = ["NoNorm_Episode1", "Norm_Episode43"]
  
  for version in versions:
    print(version)
    alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax = get_params(version)
    _, _, _, countData, _, xxis, exp_pdf = mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples=SAMPLES)

    # RMSE test
    rmse = np.sqrt(((countData - exp_pdf) ** 2).mean())
    print("RMSE:", rmse)

    # Chi2 test
    chi2 = 0
    for i in range(len(exp_pdf)):
      chi2 += ((countData[i]-exp_pdf[i])**2)/exp_pdf[i]
    print("Chi2:", chi2)

    # KL-Divergence test
    kl_div = sum(rel_entr(countData, exp_pdf))
    print("KL-Div:", kl_div)

    # CDF_MSE test
    exp_cdf = np.cumsum(exp_pdf)
    countData_cdf = np.cumsum(countData)
    cdf_mse = ((countData_cdf - exp_cdf) ** 2).mean()
    print("CDF MSE:", cdf_mse)


    plt.plot(xxis, countData, color="red", label="Actual PDF")
    plt.plot(xxis, exp_pdf,'k--', label="Expected PDF")
    plt.xlabel("Generated Number")
    plt.ylabel("Normalized")
    if version == "NoNorm_Episode1":
      title = "PDF Comparison (w/o Normalization)"
    elif version == "Norm_Episode43":
      title = "PDF Comparison (w/ Normalization)"
    plt.title(title)
    plt.legend()
    plot_path = f"graphs/{version}.png"
    plt.savefig(plot_path)
    plt.clf()
    plt.close()
    # plt.show()