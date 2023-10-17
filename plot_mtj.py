import os
import random
import argparse
import numpy as np
from scipy import stats
from scipy.special import rel_entr

from mtj_RL_dev import mtj_run


def get_config_score(chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf):
    if chi2 == None:
      invalid_config = 1
      return current_config_score
    
    w1 = 0.5
    w2 = 0.5
    # p_value = 1 - stats.chi2.cdf(chi2, 256)
    kl_div_score = sum(rel_entr(countData, exp_pdf))
    print("KL Div", kl_div_score)
    energy = np.mean(energy_avg)
    print("Energy", energy)
    
    
    # score = w1*p_value + w2*(1-energy)  # (1-energy) attempts to maximize a minimization parameter
    score = w1*kl_div_score + w2*energy*10**11
    return score


  
#Norm Best Score 1
#KL Div 0.05381614302638101
#Energy 5.20178086450265e-13
#0.05291697583570376
#{'alpha': 0.01, 'Ki': 0.0002, 'Ms': 300000.0, 'Rp': 50000.0, 'TMR': 0.3, 'eta': 0.589673674106598, 'J_she': 456989129185.6766, 't_pulse': 4.097234883904457e-08, 't_relax': 4.097234883904457e-08, 'd': 3e-09, 'tf': 1.1e-09}


#No Norm Best Score 1 (-99)
#KL Div 0.05925236188028843
#Energy 2.274049386709158e-12
#0.1433286502756021
#{'alpha': 0.01, 'Ki': 0.0002, 'Ms': 300000.0, 'Rp': 500, 'TMR': 0.3, 'eta': 0.26508163213729863, 'J_she': 1000000000000.0, 't_pulse': 3.790034800767898e-08, 't_relax': 3.790034800767898e-08, 'd': 3e-09, 'tf': 1.1e-09}

alpha = 0.01
Ki = 0.0002
Ms = 300000.0
Rp = 500
TMR = 0.3
eta = 0.26508163213729863
J_she = 1000000000000.0
t_pulse = 3.790034800767898e-08
t_relax = 3.790034800767898e-08
d = 3e-09
tf = 1.1e-09
samples=2500



chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf = mtj_run(alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, samples)
#print(exp_pdf)
current_config_score = get_config_score(chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf)
print(current_config_score)
best_config = {"alpha":alpha, "Ki":Ki, "Ms":Ms, "Rp":Rp, "TMR":TMR, "eta":eta, "J_she":J_she, "t_pulse":t_pulse, "t_relax":t_relax, "d":d, "tf":tf}
print(best_config)
