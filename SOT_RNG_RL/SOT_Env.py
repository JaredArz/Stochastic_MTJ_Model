import pickle
import random
import numpy as np
from scipy.special import rel_entr
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 

import sys
sys.path.append("../")
sys.path.append("../fortran_source")
# from mtj_model import mtj_run
from SOT_model import SOT_Model


# Hyperparameters
EPISODE_LENGTH = 150
DEV_SAMPLES = 2500


# Helper Functions
def normalize(x, min, max):
  return (x-min) / (max-min)

def unnormalize(n, min, max):
  return n*(max-min) + min


class SOT_Env(Env):
  def __init__(self):
    self.PID = random.randint(0, 100000)
    self.dev_samples = DEV_SAMPLES
    self.invalid_config = 0

    # Initial parameter values
    # self.alpha = 0.03
    # self.Ki = 0.0009725695027196851
    # self.Ms = 1200000.0
    # self.Rp = 4602.402954025149
    # self.TMR = 1.1829030593531298
    # self.eta = 0.3
    # self.J_she = 500000000000.0
    # self.t_pulse = 1e-08
    # self.t_relax = 1.5e-08
    # self.d = 3e-09
    # self.tf = 1.1e-09

    # Seed initial state
    with open("seed_params.pkl", "rb") as pklFile:
      self.seed_params = pickle.load(pklFile)
    while(True):
      params = self.seed_params[random.randint(0, len(self.seed_params)-1)]
      self.alpha = params["alpha"]
      self.Ki = params["Ki"]
      self.Ms = params["Ms"]
      self.Rp = params["Rp"]
      self.TMR = params["TMR"]
      self.eta = params["eta"]
      self.J_she = params["J_she"]
      self.t_pulse = params["t_pulse"]
      self.t_relax = params["t_relax"]
      self.d = 3e-09
      self.tf = 1.1e-09

      chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf = SOT_Model(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, self.t_pulse, self.t_relax, samples=self.dev_samples, runID=self.PID)
      if chi2 != None:
        break
    
    # Parameter ranges
    self.alpha_range = [0.01, 0.1]
    self.Ki_range = [0.2e-3, 1e-3]
    self.Ms_range = [0.3e6, 2e6]
    self.Rp_range = [500, 50000]
    self.TMR_range = [0.3, 6]
    self.eta_range = [0.1, 0.8]
    self.J_she_range = [0.01e12, 1e12]
    self.t_pulse_range = [0.5e-9, 75e-9]
    self.t_relax_range = [0.5e-9, 75e-9]

    # Get initial config score
    # chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf = SOT_Model(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, self.t_pulse, self.t_relax, samples=self.dev_samples, runID=self.PID)
    self.current_config_score = self.get_config_score(chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf)
    self.best_config_score = self.current_config_score
    self.best_config = {"alpha":self.alpha, "Ki":self.Ki, "Ms":self.Ms, "Rp":self.Rp, "TMR":self.TMR, "eta":self.eta, "J_she":self.J_she, "t_pulse":self.t_pulse, "t_relax":self.t_relax, "d":self.d, "tf":self.tf, "kl_div_score":self.kl_div_score, "energy":self.energy}

    # Continuous Actions: modify the 8 parameters; normalized values between 0-1 for best practice
    self.action_space = Box(low=0, high=1, shape=(8,), dtype=np.float32)

    # Observations: current parameter values, current config score, best config score discovered
    self.observation_space = Dict({"alpha":Box(low=self.alpha_range[0], high=self.alpha_range[1], shape=(1,), dtype=np.float32),
                                   "Ki":Box(low=self.Ki_range[0], high=self.Ki_range[1], shape=(1,), dtype=np.float32),
                                   "Ms":Box(low=self.Ms_range[0], high=self.Ms_range[1], shape=(1,), dtype=np.float32),
                                   "Rp":Box(low=self.Rp_range[0], high=self.Rp_range[1], shape=(1,), dtype=np.float32),
                                   "TMR":Box(low=self.TMR_range[0], high=self.TMR_range[1], shape=(1,), dtype=np.float32),
                                   "eta":Box(low=self.eta_range[0], high=self.eta_range[1], shape=(1,), dtype=np.float32),
                                   "J_she":Box(low=self.J_she_range[0], high=self.J_she_range[1], shape=(1,), dtype=np.float32),
                                   "t_pulse":Box(low=self.t_pulse_range[0], high=self.t_pulse_range[1], shape=(1,), dtype=np.float32),
                                   "t_relax":Box(low=self.t_relax_range[0], high=self.t_relax_range[1], shape=(1,), dtype=np.float32),
                                   "current_config_score":Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                                   "best_config_score":Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)})
    
    # Gather observations
    self.obs = self.observation_space.sample()
    self.obs["alpha"] = np.array([self.alpha], dtype=np.float32)
    self.obs["Ki"] = np.array([self.Ki], dtype=np.float32)
    self.obs["Ms"] = np.array([self.Ms], dtype=np.float32)
    self.obs["Rp"] = np.array([self.Rp], dtype=np.float32)
    self.obs["TMR"] = np.array([self.TMR], dtype=np.float32)
    self.obs["eta"] = np.array([self.eta], dtype=np.float32)
    self.obs["J_she"] = np.array([self.J_she], dtype=np.float32)
    self.obs["t_pulse"] = np.array([self.t_pulse], dtype=np.float32)
    self.obs["t_relax"] = np.array([self.t_relax], dtype=np.float32)
    self.obs["current_config_score"] = np.array([self.current_config_score], dtype=np.float32)
    self.obs["best_config_score"] = np.array([self.best_config_score], dtype=np.float32)
    
    # Set episode length
    self.episode_length = EPISODE_LENGTH


  def set_limit(self, current, lower, upper):
    x = abs(current-lower)
    y = abs(current-upper)
    rv = lower if x < y else upper
    return rv
  

  def is_out_of_bounds(self):
    if self.alpha < self.alpha_range[0] or self.alpha > self.alpha_range[1]:
      self.alpha = self.set_limit(self.alpha, self.alpha_range[0], self.alpha_range[1])
      return True
    elif self.Ki < self.Ki_range[0] or self.Ki > self.Ki_range[1]:
      self.Ki = self.set_limit(self.Ki, self.Ki_range[0], self.Ki_range[1])
      return True
    elif self.Ms < self.Ms_range[0] or self.Ms > self.Ms_range[1]:
      self.Ms = self.set_limit(self.Ms, self.Ms_range[0], self.Ms_range[1])
      return True
    elif self.Rp < self.Rp_range[0] or self.Rp > self.Rp_range[1]:
      self.Rp = self.set_limit(self.Rp, self.Rp_range[0], self.Rp_range[1])
      return True
    elif self.TMR < self.TMR_range[0] or self.TMR > self.TMR_range[1]:
      self.TMR = self.set_limit(self.TMR, self.TMR_range[0], self.TMR_range[1])
      return True
    elif self.eta < self.eta_range[0] or self.eta > self.eta_range[1]:
      self.eta = self.set_limit(self.eta, self.eta_range[0], self.eta_range[1])
      return True
    elif self.J_she < self.J_she_range[0] or self.J_she > self.J_she_range[1]:
      self.J_she = self.set_limit(self.J_she, self.J_she_range[0], self.J_she_range[1])
      return True
    elif self.t_pulse < self.t_pulse_range[0] or self.t_pulse > self.t_pulse_range[1]:
      self.t_pulse = self.set_limit(self.t_pulse, self.t_pulse_range[0], self.t_pulse_range[1])
      return True
    elif self.t_relax < self.t_relax_range[0] or self.t_relax > self.t_relax_range[1]:
      self.t_relax = self.set_limit(self.t_relax, self.t_relax_range[0], self.t_relax_range[1])
      return True
    else:
      return False
    
  
  def apply_continuous_action(self, actions):
    self.alpha = unnormalize(actions[0], self.alpha_range[0], self.alpha_range[1])
    self.Ki = unnormalize(actions[1], self.Ki_range[0], self.Ki_range[1])
    self.Ms = unnormalize(actions[2], self.Ms_range[0], self.Ms_range[1])
    self.Rp = unnormalize(actions[3], self.Rp_range[0], self.Rp_range[1])
    self.TMR = unnormalize(actions[4], self.TMR_range[0], self.TMR_range[1])
    self.eta = unnormalize(actions[5], self.eta_range[0], self.eta_range[1])
    self.J_she = unnormalize(actions[6], self.J_she_range[0], self.J_she_range[1])
    self.t_pulse = unnormalize(actions[7], self.t_pulse_range[0], self.t_pulse_range[1])
    self.t_relax = unnormalize(actions[7], self.t_relax_range[0], self.t_relax_range[1]) # potentially change


  def get_config_score(self, chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf):
    if chi2 == None:
      self.invalid_config = 1
      return self.current_config_score
    
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
    elif self.is_out_of_bounds():
      reward = -1
    elif self.current_config_score < self.best_config_score:  # Minimization reward scheme
      self.best_config_score = self.current_config_score
      self.best_config = {"alpha":self.alpha, "Ki":self.Ki, "Ms":self.Ms, "Rp":self.Rp, "TMR":self.TMR, "eta":self.eta, "J_she":self.J_she, "t_pulse":self.t_pulse, "t_relax":self.t_relax, "d":self.d, "tf":self.tf, "kl_div_score":self.kl_div_score, "energy":self.energy}
      reward = 1 
    else: 
      reward = 0 
    
    return reward


  def step(self, action):
    # Apply action
    self.apply_continuous_action(action)
    
    # Sample new configuration
    chi2, bitstream, energy_avg, countData, bitData,  xxis, exp_pdf = SOT_Model(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, self.t_pulse, self.t_relax, samples=self.dev_samples, runID=self.PID)
    self.current_config_score = self.get_config_score(chi2, bitstream, energy_avg, countData, bitData,  xxis, exp_pdf)
    
    # Calculate reward
    self.reward = self.reward_function()

    # Gather observations
    self.obs = self.observation_space.sample()
    self.obs["alpha"] = np.array([self.alpha], dtype=np.float32)
    self.obs["Ki"] = np.array([self.Ki], dtype=np.float32)
    self.obs["Ms"] = np.array([self.Ms], dtype=np.float32)
    self.obs["Rp"] = np.array([self.Rp], dtype=np.float32)
    self.obs["TMR"] = np.array([self.TMR], dtype=np.float32)
    self.obs["eta"] = np.array([self.eta], dtype=np.float32)
    self.obs["J_she"] = np.array([self.J_she], dtype=np.float32)
    self.obs["t_pulse"] = np.array([self.t_pulse], dtype=np.float32)
    self.obs["t_relax"] = np.array([self.t_relax], dtype=np.float32)
    self.obs["current_config_score"] = np.array([self.current_config_score], dtype=np.float32)
    self.obs["best_config_score"] = np.array([self.best_config_score], dtype=np.float32)
    
    # Set placeholder for info
    self.info = {"alpha"  : self.alpha,
                "Ki"      : self.Ki,
                "Ms"      : self.Ms,
                "Rp"      : self.Rp,
                "TMR"     : self.TMR,
                "eta"     : self.eta,
                "J_she"   : self.J_she,
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
    # super().reset(seed=seed)

    # Initial parameter values
    # self.alpha = 0.03
    # self.Ki = 0.0009725695027196851
    # self.Ms = 1200000.0
    # self.Rp = 4602.402954025149
    # self.TMR = 1.1829030593531298
    # self.eta = 0.3
    # self.J_she = 500000000000.0
    # self.t_pulse = 1e-08
    # self.t_relax = 1.5e-08
    # self.d = 3e-09
    # self.tf = 1.1e-09

    # Seed new initial state
    while(True):
      params = self.seed_params[random.randint(0, len(self.seed_params)-1)]
      self.alpha = params["alpha"]
      self.Ki = params["Ki"]
      self.Ms = params["Ms"]
      self.Rp = params["Rp"]
      self.TMR = params["TMR"]
      self.eta = params["eta"]
      self.J_she = params["J_she"]
      self.t_pulse = params["t_pulse"]
      self.t_relax = params["t_relax"]
      self.d = 3e-09
      self.tf = 1.1e-09

      chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf = SOT_Model(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, self.t_pulse, self.t_relax, samples=self.dev_samples, runID=self.PID)
      if chi2 != None:
        break

    # Get initial config score
    # chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf = SOT_Model(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, self.t_pulse, self.t_relax, samples=self.dev_samples, runID=self.PID)
    self.current_config_score = self.get_config_score(chi2, bitstream, energy_avg, countData, bitData, xxis, exp_pdf)
    # self.best_config_score = self.current_config_score

    # Gather observations
    self.obs = self.observation_space.sample()
    self.obs["alpha"] = np.array([self.alpha], dtype=np.float32)
    self.obs["Ki"] = np.array([self.Ki], dtype=np.float32)
    self.obs["Ms"] = np.array([self.Ms], dtype=np.float32)
    self.obs["Rp"] = np.array([self.Rp], dtype=np.float32)
    self.obs["TMR"] = np.array([self.TMR], dtype=np.float32)
    self.obs["eta"] = np.array([self.eta], dtype=np.float32)
    self.obs["J_she"] = np.array([self.J_she], dtype=np.float32)
    self.obs["t_pulse"] = np.array([self.t_pulse], dtype=np.float32)
    self.obs["t_relax"] = np.array([self.t_relax], dtype=np.float32)
    self.obs["current_config_score"] = np.array([self.current_config_score], dtype=np.float32)
    self.obs["best_config_score"] = np.array([self.best_config_score], dtype=np.float32)

    # Reset episode length
    self.episode_length = EPISODE_LENGTH
    
    # Reset config validity
    self.invalid_config = 0
    
    return self.obs, None