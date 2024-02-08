import csv
import glob
import os
import pickle
import pandas as pd
import numpy as np


def seed_parameters():
  df = pd.read_csv("valid_param.csv")
  df = df[["alpha","Ki","Ms","Rp","TMR","d","tf","eta","J_she","t_pulse","t_relax"]]

  seed_params = []
  for i, row in df.iterrows():
    seed_params.append(row.to_dict())
  
  with open("seed_params.pkl", "wb") as file:
    pickle.dump(seed_params, file)



if __name__ == "__main__":
  seed_parameters()