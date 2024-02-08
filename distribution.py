import numpy as np
import math

def gamma_pdf(g1, g2, nrange) -> list: 
  # Build an analytical gamma probability density function (PDF)

  # g1 corresponds to alpha in gamma distribution definitino
  # g2 corresponds to beta in gamma distribution, or lmda in previous work here
  # g1 must be an integer for this formula to work. if non-integer g1 are desired, factorial function should become gamma function
  xxis = []
  pdf = []
  for j in range(nrange):
    gval = pow(j,g1-1)*pow(g2,g1)*np.exp(-g2*j)/factorial(g1-1)
    xxis.append(j)
    pdf.append(gval)

  # Normalize exponential distribution
  pdfsum = 0
  for j in range(nrange):
    pdfsum += pdf[j]

  pdf = pdf/pdfsum
  
  return pdf


