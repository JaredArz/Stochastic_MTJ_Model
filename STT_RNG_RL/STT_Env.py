import pickle
import random
import numpy as np
from scipy.special import rel_entr
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 

import sys
sys.path.append("../")
sys.path.append("../fortran_source")
from STT_model import STT_Model


# Hyperparameters
EPISODE_LENGTH = 150
DEV_SAMPLES = 2500


# Helper Functions
def normalize(x, min, max):
  return (x-min) / (max-min)

def unnormalize(n, min, max):
  return n*(max-min) + min


class STT_Env(Env):
  def __init__(self, pdf_type="exp"):
    self.pdf_type = pdf_type
    self.episode_length = EPISODE_LENGTH
    self.dev_samples = DEV_SAMPLES
    self.invalid_config = 0

    # Initial parameter values
    self.alpha = 0.03
    self.K_295 = 1.0056364e-3
    self.Ms_295 = 1.2e6
    self.Rp = 5e3
    self.TMR = 3
    self.t_pulse = 1e-9
    self.t_relax = 10e-9
    self.d = 3e-09
    self.tf = 1.1e-09
    
    # Parameter ranges
    self.alpha_range = [0.01, 0.1]
    self.K_295_range = [0.2e-3, 1e-3]
    self.Ms_295_range = [0.3e6, 2e6]
    self.Rp_range = [500, 50000]
    self.t_pulse_range = [0.5e-9, 75e-9]
    self.t_relax_range = [0.5e-9, 75e-9]

    # Get initial config score
    chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = STT_Model(self.alpha, self.K_295, self.Ms_295, self.Rp, self.TMR, self.d, self.tf, self.t_pulse, self.t_relax, samples=self.dev_samples, pdf_type=self.pdf_type)
    self.current_config_score = self.get_config_score(chi2, bitstream, energy_avg, countData, bitData, xxis, pdf)
    self.best_config_score = self.current_config_score
    self.best_config = {"alpha":self.alpha, "K_295":self.K_295, "Ms_295":self.Ms_295, "Rp":self.Rp, "TMR":self.TMR, "t_pulse":self.t_pulse, "t_relax":self.t_relax, "d":self.d, "tf":self.tf, "kl_div_score":self.kl_div_score, "energy":self.energy}

    # Continuous Actions: modify the 8 parameters; normalized values between 0-1 for best practice
    self.action_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)

    # Observations: current parameter values, current config score, best config score discovered
    self.observation_space = Dict({"alpha":Box(low=self.alpha_range[0], high=self.alpha_range[1], shape=(1,), dtype=np.float32),
                                   "K_295":Box(low=self.K_295_range[0], high=self.K_295_range[1], shape=(1,), dtype=np.float32),
                                   "Ms_295":Box(low=self.Ms_295_range[0], high=self.Ms_295_range[1], shape=(1,), dtype=np.float32),
                                   "Rp":Box(low=self.Rp_range[0], high=self.Rp_range[1], shape=(1,), dtype=np.float32),
                                   "t_pulse":Box(low=self.t_pulse_range[0], high=self.t_pulse_range[1], shape=(1,), dtype=np.float32),
                                   "t_relax":Box(low=self.t_relax_range[0], high=self.t_relax_range[1], shape=(1,), dtype=np.float32),
                                   "current_config_score":Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                                   "best_config_score":Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)})
    
    # Gather observations
    self.obs = self.observation_space.sample()
    self.obs["alpha"] = np.array([self.alpha], dtype=np.float32)
    self.obs["K_295"] = np.array([self.K_295], dtype=np.float32)
    self.obs["Ms_295"] = np.array([self.Ms_295], dtype=np.float32)
    self.obs["Rp"] = np.array([self.Rp], dtype=np.float32)
    self.obs["t_pulse"] = np.array([self.t_pulse], dtype=np.float32)
    self.obs["t_relax"] = np.array([self.t_relax], dtype=np.float32)
    self.obs["current_config_score"] = np.array([self.current_config_score], dtype=np.float32)
    self.obs["best_config_score"] = np.array([self.best_config_score], dtype=np.float32)
    
  
  def apply_continuous_action(self, actions):
    self.alpha = unnormalize(actions[0], self.alpha_range[0], self.alpha_range[1])
    self.K_295 = unnormalize(actions[1], self.K_295_range[0], self.K_295_range[1])
    self.Ms_295 = unnormalize(actions[2], self.Ms_295_range[0], self.Ms_295_range[1])
    self.Rp = unnormalize(actions[3], self.Rp_range[0], self.Rp_range[1])
    self.t_pulse = unnormalize(actions[4], self.t_pulse_range[0], self.t_pulse_range[1])
    self.t_relax = unnormalize(actions[4], self.t_relax_range[0], self.t_relax_range[1]) # potentially change
    self.TMR = 3
    self.d = 3e-09
    self.tf = 1.1e-09


  def get_config_score(self, chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf):
    if chi2 == None:
      self.invalid_config = 1
      self.kl_div_score = 1000000
      self.energy = 1000000
      score = 1000000
      # score = self.current_config_score
      return score
    
    w1 = 0.5
    w2 = 0.5
    self.kl_div_score = sum(rel_entr(countData, exp_pdf))
    self.energy = np.mean(energy_avg)
    score = w1*self.kl_div_score + w2*self.energy
    return score


  def reward_function(self):
    if self.invalid_config == 1:
      reward = -1
      self.invalid_config = 0
    elif self.current_config_score < self.best_config_score:  # Minimization reward scheme
      self.best_config_score = self.current_config_score
      self.best_config = {"alpha":self.alpha, "K_295":self.K_295, "Ms_295":self.Ms_295, "Rp":self.Rp, "TMR":self.TMR, "t_pulse":self.t_pulse, "t_relax":self.t_relax, "d":self.d, "tf":self.tf, "kl_div_score":self.kl_div_score, "energy":self.energy}
      reward = 1 
    else: 
      reward = 0 
    
    return reward


  def step(self, action):
    # Apply action
    self.apply_continuous_action(action)
    
    # Sample new configuration
    chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = STT_Model(self.alpha, self.K_295, self.Ms_295, self.Rp, self.TMR, self.d, self.tf, self.t_pulse, self.t_relax, samples=self.dev_samples, pdf_type=self.pdf_type)
    self.current_config_score = self.get_config_score(chi2, bitstream, energy_avg, countData, bitData,  xxis, pdf)
    
    # Calculate reward
    self.reward = self.reward_function()

    # Gather observations
    self.obs = self.observation_space.sample()
    self.obs["alpha"] = np.array([self.alpha], dtype=np.float32)
    self.obs["K_295"] = np.array([self.K_295], dtype=np.float32)
    self.obs["Ms_295"] = np.array([self.Ms_295], dtype=np.float32)
    self.obs["Rp"] = np.array([self.Rp], dtype=np.float32)
    self.obs["t_pulse"] = np.array([self.t_pulse], dtype=np.float32)
    self.obs["t_relax"] = np.array([self.t_relax], dtype=np.float32)
    self.obs["current_config_score"] = np.array([self.current_config_score], dtype=np.float32)
    self.obs["best_config_score"] = np.array([self.best_config_score], dtype=np.float32)
    
    # Set placeholder for info
    self.info = {"alpha"  : self.alpha,
                "K_295"   : self.K_295,
                "Ms_295"  : self.Ms_295,
                "Rp"      : self.Rp,
                "TMR"     : self.TMR,
                "t_pulse" : self.t_pulse,
                "t_relax" : self.t_relax,
                "d"       : self.d,
                "tf"      : self.tf,
                "energy"               : self.energy,
                "kl_div_score"         : self.kl_div_score,
                "current_config_score" : self.current_config_score,
                "best_config_score"    : self.best_config_score,
                "best_config"          : self.best_config}

    # Check if episode is done
    if self.episode_length <= 0: 
      truncated = True
    else:
      truncated = False
    
    # Reduce episode length
    self.episode_length -= 1 

    # Return step information
    return self.obs, self.reward, False, truncated, self.info


  def render(self):
    # Implement viz
    pass
  

  def reset(self, seed=None, options=None):
    # Initial parameter values
    self.alpha = 0.03
    self.K_295 = 1.0056364e-3
    self.Ms_295 = 1.2e6
    self.Rp = 5e3
    self.TMR = 3
    self.t_pulse = 1e-9
    self.t_relax = 10e-9
    self.d = 3e-09
    self.tf = 1.1e-09

    # Get initial config score
    chi2, bitstream, energy_avg, countData, bitData, xxis, pdf = STT_Model(self.alpha, self.K_295, self.Ms_295, self.Rp, self.TMR, self.d, self.tf, self.t_pulse, self.t_relax, samples=self.dev_samples, pdf_type=self.pdf_type)
    self.current_config_score = self.get_config_score(chi2, bitstream, energy_avg, countData, bitData, xxis, pdf)
    # self.best_config_score = self.current_config_score

    # Gather observations
    self.obs = self.observation_space.sample()
    self.obs["alpha"] = np.array([self.alpha], dtype=np.float32)
    self.obs["K_295"] = np.array([self.K_295], dtype=np.float32)
    self.obs["Ms_295"] = np.array([self.Ms_295], dtype=np.float32)
    self.obs["Rp"] = np.array([self.Rp], dtype=np.float32)
    self.obs["t_pulse"] = np.array([self.t_pulse], dtype=np.float32)
    self.obs["t_relax"] = np.array([self.t_relax], dtype=np.float32)
    self.obs["current_config_score"] = np.array([self.current_config_score], dtype=np.float32)
    self.obs["best_config_score"] = np.array([self.best_config_score], dtype=np.float32)

    # Reset episode length
    self.episode_length = EPISODE_LENGTH
    
    # Reset config validity
    self.invalid_config = 0
    
    return self.obs, None