import csv
import glob
import os
import sys


def main():
  ID = 19999

  path = "results/bitData/"
  nums = []
  for file in glob.glob(os.path.join(path, '*')):
    fnum = file.split("/")[-1].split(".")[0].split("_")[-1]
    fnum = int(fnum)
    nums.append(fnum)

  nums.sort()

  x = 0
  for i in nums:
    if i != x:
      print("fnum: {}   num: {}".format(i, x))
      exit()
    x += 1
    

if __name__ == "__main__":
  main()