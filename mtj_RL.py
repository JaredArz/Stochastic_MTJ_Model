import os
import random
import argparse
import numpy as np
from scipy import stats

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from mtj_RL_dev import mtj_run

# Hyperparameters
EPISODE_LENGTH = 150
DEV_SAMPLES = 250


class MTJ_Env(Env):
  def __init__(self):
    self.dev_samples = DEV_SAMPLES
    self.invalid_config = 0
    self.current_config_score = np.inf

    # Initial parameter values
    self.alpha = 0.03
    self.Ki = 0.0009725695027196851
    self.Ms = 1200000.0
    self.Rp = 4602.402954025149
    self.TMR = 1.1829030593531298
    self.eta = 0.3
    self.J_she = 500000000000.0
    self.t_pulse = 1e-08
    self.t_relax = 1.5e-08
    # self.t_pulse = 1e-08
    # self.t_relax = 1e-08
    self.d = 3e-09
    self.tf = 1.1e-09
    
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

    # Parameter step sizes
    self.alpha_step = 0.01
    self.Ki_step = 0.1e-3
    self.Ms_step = 0.1e6
    self.Rp_step = 500
    self.TMR_step = 0.5
    self.eta_step = 0.4
    self.J_she_step = 0.25e12
    self.t_pulse_step = 5e-9
    self.t_relax_step = 5e-9

    # Get initial config score
    chi2, bitstream, energy_avg, countData, bitData = mtj_run(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, self.t_pulse, self.t_relax, samples=self.dev_samples)
    self.current_config_score = self.get_config_score(chi2, bitstream, energy_avg, countData, bitData)
    self.best_config_score = self.current_config_score
    self.best_config = {"alpha":self.alpha, "Ki":self.Ki, "Ms":self.Ms, "Rp":self.Rp, "TMR":self.TMR, "eta":self.eta, "J_she":self.J_she, "t_pulse":self.t_pulse, "t_relax":self.t_relax, "d":self.d, "tf":self.tf}

    # Discrete Actions: increase/decrease the 8 parameters
    # self.action_space = Discrete(16)

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
  

  # def apply_discrete_action(self, action):
  #   if action == 0:
  #     temp = self.alpha + self.alpha_step
  #     self.alpha = min(temp, self.alpha_range[1])
  #   elif action == 1:
  #     temp = self.alpha - self.alpha_step
  #     self.alpha = max(temp, self.alpha_range[0])

  #   elif action == 2:
  #     temp = self.Ki + self.Ki_step
  #     self.Ki = min(temp, self.Ki_range[1])
  #   elif action == 3:
  #     temp = self.Ki - self.Ki_step
  #     self.Ki = max(temp, self.Ki_range[0])
    
  #   elif action == 4:
  #     temp = self.Ms + self.Ms_step
  #     self.Ms = min(temp, self.Ms_range[1])
  #   elif action == 5:
  #     temp = self.Ms - self.Ms_step
  #     self.Ms = max(temp, self.Ms_range[0])
    
  #   elif action == 6:
  #     temp = self.Rp + self.Rp_step
  #     self.Rp = min(temp, self.Rp_range[1])
  #   elif action == 7:
  #     temp = self.Rp - self.Rp_step
  #     self.Rp = max(temp, self.Rp_range[0])
    
  #   elif action == 8:
  #     temp = self.TMR + self.TMR_step
  #     self.TMR = min(temp, self.TMR_range[1])
  #   elif action == 9:
  #     temp = self.TMR - self.TMR_step
  #     self.TMR = max(temp, self.TMR_range[0])
    
  #   elif action == 10:
  #     temp = self.eta + self.eta_step
  #     self.eta = min(temp, self.eta_range[1])
  #   elif action == 11:
  #     temp = self.eta - self.eta_step
  #     self.eta = max(temp, self.eta_range[0])
    
  #   elif action == 12:
  #     temp = self.J_she + self.J_she_step
  #     self.J_she = min(temp, self.J_she_range[1])
  #   elif action == 13:
  #     temp = self.J_she - self.J_she_step
  #     self.J_she = max(temp, self.J_she_range[0])


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
    
  
  def apply_discrete_action(self, action):
    if action == 0:
      self.alpha = self.alpha + self.alpha_step
    elif action == 1:
      self.alpha = self.alpha - self.alpha_step

    elif action == 2:
      self.Ki = self.Ki + self.Ki_step
    elif action == 3:
      self.Ki = self.Ki - self.Ki_step
    
    elif action == 4:
      self.Ms = self.Ms + self.Ms_step
    elif action == 5:
      self.Ms = self.Ms - self.Ms_step
    
    elif action == 6:
      self.Rp = self.Rp + self.Rp_step
    elif action == 7:
      self.Rp = self.Rp - self.Rp_step
    
    elif action == 8:
      self.TMR = self.TMR + self.TMR_step
    elif action == 9:
      self.TMR = self.TMR - self.TMR_step
    
    elif action == 10:
      self.eta = self.eta + self.eta_step
    elif action == 11:
      self.eta = self.eta - self.eta_step
    
    elif action == 12:
      self.J_she = self.J_she + self.J_she_step
    elif action == 13:
      self.J_she = self.J_she - self.J_she_step
    
    elif action == 14:
      self.t_pulse = self.t_pulse + self.t_pulse_step
      self.t_relax = self.t_relax + self.t_relax_step
    elif action == 15:
      self.t_pulse = self.t_pulse - self.t_pulse
      self.t_relax = self.t_relax - self.t_relax
    
  
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


  def get_config_score(self, chi2, bitstream, energy_avg, countData, bitData):
    if chi2 == None:
      self.invalid_config = 1
      return self.current_config_score
    
    w1 = 1
    w2 = 1
    p_value = 1 - stats.chi2.cdf(chi2, 256)
    energy = np.mean(energy_avg) 
    
    score = w1*p_value + w2*(1-energy)  # (1-energy) attempts to maximize a minimization parameter
    return score


  def reward_function(self):
    if self.invalid_config == 1:
      reward = -1
      self.invalid_config = 0
    elif self.is_out_of_bounds():
      reward = -1
    # elif self.current_config_score < self.best_config_score:  # Minimization reward scheme
    elif self.current_config_score > self.best_config_score:  # Maximization reward scheme
      self.best_config_score = self.current_config_score
      self.best_config = {"alpha":self.alpha, "Ki":self.Ki, "Ms":self.Ms, "Rp":self.Rp, "TMR":self.TMR, "eta":self.eta, "J_she":self.J_she, "t_pulse":self.t_pulse, "t_relax":self.t_relax, "d":self.d, "tf":self.tf}
      reward = 1 
    else: 
      reward = 0 
    
    return reward


  def step(self, action):
    # Apply action
    # self.apply_discrete_action(action)
    self.apply_continuous_action(action)
    
    # Sample new configuration
    chi2, bitstream, energy_avg, countData, bitData = mtj_run(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, self.t_pulse, self.t_relax, samples=self.dev_samples)
    self.current_config_score = self.get_config_score(chi2, bitstream, energy_avg, countData, bitData)
    
    # Calculate reward
    reward = self.reward_function()

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
    self.info = {'alpha'  : self.alpha,
                'Ki'      : self.Ki,
                'Ms'      : self.Ms,
                'Rp'      : self.Rp,
                'TMR'     : self.TMR,
                'eta'     : self.eta,
                'J_she'   : self.J_she,
                't_pulse' : self.t_pulse,
                't_relax' : self.t_relax,
                "d"       : self.d,
                "tf"      : self.tf}

    # Check if episode is done
    if self.episode_length <= 0: 
      done = True
    else:
      done = False
    
    # Reduce episode length
    self.episode_length -= 1 

    # Return step information
    return self.obs, reward, done, self.info


  def render(self):
    # Implement viz
    pass
  

  def reset(self):
    # Initial parameter values
    self.alpha = 0.03
    self.Ki = 0.0009725695027196851
    self.Ms = 1200000.0
    self.Rp = 4602.402954025149
    self.TMR = 1.1829030593531298
    self.eta = 0.3
    self.J_she = 500000000000.0
    self.t_pulse = 1e-08
    self.t_relax = 1.5e-08
    # self.t_pulse = 1e-08
    # self.t_relax = 1e-08
    self.d = 3e-09
    self.tf = 1.1e-09
    
    # Get initial config score
    chi2, bitstream, energy_avg, countData, bitData = mtj_run(self.alpha, self.Ki, self.Ms, self.Rp, self.TMR, self.d, self.tf, self.eta, self.J_she, self.t_pulse, self.t_relax, samples=self.dev_samples)
    self.current_config_score = self.get_config_score(chi2, bitstream, energy_avg, countData, bitData)
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
    
    return self.obs
  


############################################## Helper Functions ##############################################
def normalize(x, min, max):
  return (x-min) / (max-min)


def unnormalize(n, min, max):
  return n*(max-min) + min


def Train_Model(model_name, training_timesteps:int, eval:bool=True):
  env = MTJ_Env()
  log_path = os.path.join('Training', 'Logs')
  # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
  model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path)
  model.learn(total_timesteps=training_timesteps)

  # Save model
  model_path = os.path.join('Training', 'Saved_Models', model_name)
  model.save(model_path)

  if eval == True:
    evaluation = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(evaluation)


def Test_Model(model_path:str, episodes:int):
  model = PPO.load(model_path)
  env = MTJ_Env()

  for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    infos = []
    config_scores = []
    
    while not done:
      action, _state = model.predict(obs)
      obs, reward, done, info = env.step(action)
      score += reward

      infos.append(info)
      config_scores.append(env.current_config_score)

      print(f"Action: {action}")
      print(f"Obs   : {obs}")
      print(f"Reward: {reward}")

    best_episode_config = infos[np.argmin(config_scores)]
    print(f"\nEpisode: {episode} Score: {score}")
    print(f"Best Episode Config: {best_episode_config}")
    print("**************************************************************************\n")
  
  print(f"Best Config: {env.best_config}")

  env.close()


def Test_Env(episodes=5):
  env = MTJ_Env()

  for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0 
    
    while not done:
      # env.render()
      action = env.action_space.sample()
      obs, reward, done, info = env.step(action)
      score += reward

      print(f"Action: {action}")
      print(f"Obs   : {obs}")
      print(f"Reward: {reward}")

    print(f"\nEpisode: {episode} Score: {score}")
    print("**************************************************************************\n")
  
  print(f"Best Config: {env.best_config}")

  env.close()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--activity", "-a", required=True, type=str, choices=["TrainModel", "TestModel", "TestEnv"], help="activity to perform")
  args = parser.parse_args()

  if (args.activity == "TrainModel"):
    model_name = "mtj_model_energyTest3"
    training_timesteps = 500000
    eval = True
    Train_Model(model_name, training_timesteps, eval)


  if (args.activity == "TestModel"):
    model_path = "Training/Saved_Models/mtj_model_energyTest2"
    Test_Model(model_path, episodes=150)


  if (args.activity == "TestEnv"):
    Test_Env(episodes=5)