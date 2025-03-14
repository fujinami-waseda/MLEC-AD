import numpy as np
import pandas as pd
import sys
import os

os.makedirs('ens', exist_ok=True)

outpath = "/home/haru/github/MLEC-AD/Sample/Jagging/"
mol = "1"
sets = "out"
typ = 2
print(mol)

data = np.loadtxt(outpath+"/1/"+sets+"/"+mol+"_result.csv", delimiter=",")
data2 = data.reshape(-1, 1)

for i in range(1,15):
  i += 1
  print(i)
  data = np.loadtxt(outpath+"/"+str(i)+"/"+sets+"/"+mol+"_result.csv", delimiter=",")
  data = data.reshape(-1, 1)
  data2 = np.hstack([data2, data])

data3 = data2.std(axis=1, ddof=1)

np.savetxt("./ens/"+mol+"_std.csv", data3, delimiter=",")
np.savetxt("./ens/"+mol+"_ens.csv", data2, delimiter=",")


