import os
import csv
import argparse
import numpy as np
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold

from SOT_Env import SOT_Env


# Custom callback for plotting additional values in tensorboard.
class TensorboardCallback_SingleProcess(BaseCallback):
  def __init__(self, env, verbose=0):
    super().__init__(verbose)
    self.env = env
    self.score = 0

  def _on_step(self) -> bool:
    self.score += self.env.reward
    self.logger.record("score", self.score)
    self.logger.record("kl_div_score", self.env.kl_div_score)
    self.logger.record("energy", self.env.energy)
    self.logger.record("current_config_score", self.env.current_config_score)
    self.logger.record("best_config_score", self.env.best_config_score)
    if (self.num_timesteps % 1 == 0):
      self.logger.dump(self.num_timesteps)
    return True
    
class TensorboardCallback_MultiProcess(BaseCallback):
  def __init__(self, num_envs, verbose=0):
    super().__init__(verbose)
    self.num_envs = num_envs
    self.score = [0 for i in range(self.num_envs)]

  def _on_step(self) -> bool:
    for i in range(self.num_envs):
      self.score[i] += self.locals["rewards"][i]
      self.logger.record(f"score_{i}", self.score[i])
      self.logger.record(f"kl_div_score_{i}", self.locals["infos"][i]["kl_div_score"])
      self.logger.record(f"energy_{i}", self.locals["infos"][i]["energy"])
      self.logger.record(f"current_config_score_{i}", self.locals["infos"][i]["current_config_score"])
      self.logger.record(f"best_config_score_{i}", self.locals["infos"][i]["best_config_score"])
      if (self.num_timesteps % 500 == 0):
        self.logger.dump(self.num_timesteps)
      return True


def Train_Model(pdf_type, model_dir, log_dir, num_envs:int=1, training_timesteps:int=1000, log_window:int=100, eval:bool=False):
  if num_envs == 1:
    env = SOT_Env(pdf_type)
    callback = TensorboardCallback_SingleProcess(env)
  else:
    # Distributes each environment to its own process; better for complex environments
    make_env = lambda : SOT_Env(pdf_type)
    env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=SubprocVecEnv)
    callback = TensorboardCallback_MultiProcess(num_envs)
  
  model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

  for i in range(1, training_timesteps//log_window):
    # Train model for given time window
    model.learn(total_timesteps=log_window, reset_num_timesteps=False, callback=callback)

    # Save model for given time window
    model_path = f"{model_dir}/timestep-{log_window*i}"
    model.save(model_path)
  
  if eval == True:
    evaluation = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(evaluation)


def Test_Model(pdf_type, model_path:str, csvFile:str, episodes:int):
  model = PPO.load(model_path)
  env = SOT_Env(pdf_type)

  f = open(csvFile, "w")
  writeFile = csv.writer(f)
  writeFile.writerow(["Episode", "Reward", "alpha", "Ki", "Ms", "Rp", "TMR", "eta", "J_she", "t_pulse", "t_relax", "d", "tf", "kl_div_score", "energy"])

  for episode in range(1, episodes+1):
    obs, _ = env.reset()
    done = False
    score = 0
    infos = []
    config_scores = []
    
    while not done:
      action, _state = model.predict(obs)
      obs, reward, terminated, truncated, info = env.step(action)
      score += reward
      done = terminated or truncated

      infos.append(info)
      config_scores.append(env.current_config_score)

      if reward != -1:
        writeFile.writerow([episode, reward, info["alpha"], info["Ki"], info["Ms"], info["Rp"], info["TMR"], info["eta"], info["J_she"], info["t_pulse"], info["t_relax"], info["d"], info["tf"], info["kl_div_score"], info["energy"]])
        f.flush()

      # print(f"Action: {action}")
      # print(f"Obs   : {obs}")
      print(f"Reward: {reward}")

    best_episode_config = infos[np.argmin(config_scores)]
    print(f"\nEpisode: {episode} Score: {score}")
    print(f"Best Episode Config: {best_episode_config}")
    print("**************************************************************************\n")
  
  print(f"Best Config: {env.best_config}")
  f.close()
  env.close()


def Test_Env(pdf_type, episodes=5):
  env = SOT_Env(pdf_type)

  for episode in range(1, episodes+1):
    obs, _ = env.reset()
    done = False
    score = 0 
    
    while not done:
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
      score += reward
      done = terminated or truncated

      # print(f"Action: {action}")
      # print(f"Obs   : {obs}")
      print(f"Reward: {reward}")

    print(f"\nEpisode: {episode} Score: {score}")
    print("**************************************************************************\n")
  
  print(f"Best Config: {env.best_config}")
  env.close()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--activity", "-a", required=True, type=str, choices=["TrainModel", "TestModel", "TestEnv"], help="activity to perform")
  parser.add_argument("--pdf_type", "-pdf", required=True, type=str, choices=["exp", "gamma"], help="pdf type")
  args = parser.parse_args()


  if (args.activity == "TrainModel"):
    model_name = "mtj_model_1"
    model_dir = os.path.join(f"SOT_{args.pdf_type}_Training", "Saved_Models", model_name)
    log_dir = os.path.join(f"SOT_{args.pdf_type}_Training", "Logs", model_name)
    training_timesteps = 500000
    log_window = training_timesteps//1000
    num_envs = 1
    eval = True

    Train_Model(args.pdf_type, model_dir, log_dir, num_envs, training_timesteps, log_window, eval)


  if (args.activity == "TestModel"):
    model_path = "JETCAS_gamma_Training/Saved_Models/mtj_model_1/timestep-6000.zip"
    model_version = model_path.split("/")[-1][:-4]
    csvFile = f"SOT_{args.pdf_type.capitalize()}_Model-{model_version}_Results.csv"
    Test_Model(args.pdf_type, model_path, csvFile, episodes=150)


  if (args.activity == "TestEnv"):
    Test_Env(args.pdf_type, episodes=5)