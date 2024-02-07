import os
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
class TensorboardCallback(BaseCallback):
  def __init__(self, envs, verbose=0):
    super().__init__(verbose)
    self.envs = envs
    self.score = [0 for i in range(self.envs.num_envs)]

  def _on_step(self) -> bool:
    for i in range(self.envs.num_envs):
      self.score[i] += self.envs.envs[i].reward
      self.logger.record(f"score_{i}", self.score[i])
      self.logger.record(f"kl_div_score_{i}", self.envs.envs[i].kl_div_score)
      self.logger.record(f"energy_{i}", self.envs.envs[i].energy)
      self.logger.record(f"current_config_score_{i}", self.envs.envs[i].current_config_score)
      self.logger.record(f"best_config_score_{i}", self.envs.envs[i].best_config_score)
      if (self.num_timesteps % 500 == 0):
        self.logger.dump(self.num_timesteps)
      return True


def Train_Model(model_dir, log_dir, num_envs:int=1, training_timesteps:int=1000, log_window:int=100, eval:bool=False):
  if num_envs == 1:
    # Calls each environment in sequence on the current Python process; better for simple environments
    env = make_vec_env(SOT_Env, n_envs=num_envs)
  else:
    # Distributes each environment to its own process; better for complex environments
    env = make_vec_env(SOT_Env, n_envs=num_envs, vec_env_cls=SubprocVecEnv)
  
  model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

  for i in range(1, training_timesteps//log_window):
    # Train model for given time window
    model.learn(total_timesteps=log_window, reset_num_timesteps=False, callback=TensorboardCallback(env))

    # Save model for given time window
    model_path = f"{model_dir}/timestep-{log_window*i}"
    model.save(model_path)
  
  if eval == True:
    evaluation = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(evaluation)


def Test_Model(model_path:str, episodes:int):
  model = PPO.load(model_path)
  env = SOT_Env()

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
  env = SOT_Env()

  for episode in range(1, episodes+1):
    obs, _ = env.reset()
    done = False
    score = 0 
    
    while not done:
      # env.render()
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
      score += reward
      done = terminated or truncated

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
    model_name = "mtj_model_1"
    model_dir = os.path.join('JETCAS_Training', 'Saved_Models', model_name)
    log_dir = os.path.join('JETCAS_Training', 'Logs', model_name)
    training_timesteps = 500000
    log_window = training_timesteps//1000
    num_envs = 4
    eval = True

    Train_Model(model_dir, log_dir, num_envs, training_timesteps, log_window, eval)


  if (args.activity == "TestModel"):
    model_path = "JETCAS_Training/Saved_Models/mtj_model_1"
    Test_Model(model_path, episodes=150)


  if (args.activity == "TestEnv"):
    Test_Env(episodes=5)