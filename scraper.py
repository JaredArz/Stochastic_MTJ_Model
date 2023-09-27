import csv
import glob
import os
import sys


def scraper():
  csvFile = "MTJ_Param_Results.csv"
  f = open(csvFile, "w")
  writeFile = csv.writer(f)
  writeFile.writerow(["alpha", "Ki", "Ms", "Rp", "TMR", "d", "tf", "eta", "J_she", "t_pulse", "t_relax", 
                      "distribution_plot_path",
                      "magnetization_plot_path",
                      "bitstream_path",
                      "energy_path",
                      "countData_path",
                      "bitData_path",
                      "chi2Data_path"])

  path = "results/parameter_files"
  for file in glob.glob(os.path.join(path, '*.txt')):
    alpha = None
    Ki = None
    Ms = None
    Rp = None
    TMR = None
    d = None
    tf = None
    eta = None
    J_she = None
    t_pulse = None
    t_relax = None
    distribution_plot_path = None
    magnetization_plot_path = None
    bitstream_path = None
    energy_path = None
    countData_path = None
    bitData_path = None
    chi2Data_path = None

    with open(file) as f:
      for line in f:
        line = line.rstrip()
        if "alpha:" in line:
          alpha = float(line.split(":")[-1].strip())
        elif "Ki:" in line:
          Ki = float(line.split(":")[-1].strip())
        elif "Ms:" in line:
          Ms = float(line.split(":")[-1].strip())
        elif "Rp:" in line:
          Rp = float(line.split(":")[-1].strip())
        elif "TMR:" in line:
          TMR = float(line.split(":")[-1].strip())
        elif "d:" in line:
          d = float(line.split(":")[-1].strip())
        elif "tf:" in line:
          tf = float(line.split(":")[-1].strip())
        elif "eta:" in line:
          eta = float(line.split(":")[-1].strip())
        elif "J_she:" in line:
          J_she = float(line.split(":")[-1].strip())
        elif "t_pulse:" in line:
          t_pulse = float(line.split(":")[-1].strip())
        elif "t_relax:" in line:
          t_relax = float(line.split(":")[-1].strip())
        elif "distribution_plot_path:" in line:
          distribution_plot_path = line.split(":")[-1].strip()
        elif "magnetization_plot_path:" in line:
          magnetization_plot_path = line.split(":")[-1].strip()
        elif "bitstream_path:" in line:
          bitstream_path = line.split(":")[-1].strip()
        elif "energy_path:" in line:
          energy_path = line.split(":")[-1].strip()
        elif "countData_path:" in line:
          countData_path = line.split(":")[-1].strip()
        elif "bitData_path:" in line:
          bitData_path = line.split(":")[-1].strip()
        elif "chi2Data_path:" in line:
          chi2Data_path = line.split(":")[-1].strip()

    writeFile.writerow([alpha, Ki, Ms, Rp, TMR, d, tf, eta, J_she, t_pulse, t_relax, 
                      distribution_plot_path,
                      magnetization_plot_path,
                      bitstream_path,
                      energy_path,
                      countData_path,
                      bitData_path,
                      chi2Data_path])

  f.close()
      

if __name__ == "__main__":
  scraper()