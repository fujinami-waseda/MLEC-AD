import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd

from train_torch_gpu import NN

def model_setting_test(ndes, unit, layer_num, actf):
    if (layer_num == 1):
        model = NN(ndes, unit, 0, 0, 0, 0, 0, 0, 0, 0, 0, layer_num, actf)
    if (layer_num == 2):
        model = NN(ndes, unit, unit, 0, 0, 0, 0, 0, 0, 0, 0, layer_num, actf)
    if (layer_num == 3):
        model = NN(ndes, unit, unit, unit, 0, 0, 0, 0, 0, 0, 0, layer_num, actf)
    if (layer_num == 4):
        model = NN(ndes, unit, unit, unit, unit, 0, 0, 0, 0, 0, 0, layer_num, actf)
    if (layer_num == 5):
        model = NN(ndes, unit, unit, unit, unit, unit, 0, 0, 0, 0, 0, layer_num, actf)
    if (layer_num == 6):
        model = NN(ndes, unit, unit, unit, unit, unit, unit, 0, 0, 0, 0, layer_num, actf)
    if (layer_num == 7):
        model = NN(ndes, unit, unit, unit, unit, unit, unit, unit, 0, 0, 0, layer_num, actf)
    if (layer_num == 8):
        model = NN(ndes, unit, unit, unit, unit, unit, unit, unit, unit, 0, 0, layer_num, actf)
    if (layer_num == 9):
        model = NN(ndes, unit, unit, unit, unit, unit, unit, unit, unit, unit, 0, layer_num, actf)
    if (layer_num == 10):
        model = NN(ndes, unit, unit, unit, unit, unit, unit, unit, unit, unit, unit, layer_num, actf)
    
    return model

def prepro(x,d,b):
    x = x + d
    x = np.log(x)/np.log(b)
    
    return x

def reading_csv_test(csvdir_x, mol):
    test_x = np.loadtxt(csvdir_x + '/' + mol + '.csv', delimiter=",", skiprows=1)

    return test_x

def cutoff(x, pre_y, thr):
    for i in range(len(x)):
        if x[i] < thr:
            pre_y[i] = 0.0

    return pre_y

def inv_trans(y, d, b):
    y = b**y
    y = d - y

    return y

def procdata(d, desType, delta, base, msdir):
    x = d[:, [4, 5, 6, 7]]  # rho, grad, tau, exchange

    x[:,3] = -x[:,3]

    # for jagging 
    if desType == "MLEN_1":
        x   = x[:,0]
        ndes = 1
    if desType == "MLEN_2":
        x   = x[:,1]
        ndes = 1
    if desType == "MLEN_3":
        x   = x[:,2]
        ndes = 1
    if desType == "MLEN_4":
        x   = x[:,3]
        ndes = 1
    if desType == "MLEN_5":
        x   = x[:,[0, 1]]
        ndes = 2
    if desType == "MLEN_6":
        x   = x[:,[0, 2]]
        ndes = 2
    if desType == "MLEN_7":
        x   = x[:,[0, 3]]
        ndes = 2
    if desType == "MLEN_8":
        x   = x[:,[1, 2]]
        ndes = 2
    if desType == "MLEN_9":
        x   = x[:,[1, 3]]
        ndes = 2
    if desType == "MLEN_10":
        x   = x[:,[2, 3]]
        ndes = 2
    if desType == "MLEN_11":
        x   = x[:,[0, 1, 2]]
        ndes = 3
    if desType == "MLEN_12":
        x   = x[:,[0, 1, 3]]
        ndes = 3
    if desType == "MLEN_13":
        x   = x[:,[0, 2, 3]]
        ndes = 3
    if desType == "MLEN_14":
        x   = x[:,[1, 2, 3]]
        ndes = 3
    if desType == "MLEN_15":
        x   = x[:,[0, 1, 2, 3]]
        ndes = 4

    x = prepro(x, delta, base)
    x = x.astype(np.float64)

    w = d[:, [3]] 
    rho = d[:, [4]]

    with open(msdir + "/ms.npy", "rb") as fms:
        ms = np.load(msdir + '/ms.npy')

    if ndes == 1:
        x_mu = ms[0]
        x_sigma = ms[1]
    else:
        x_mu = ms[:,0]
        x_sigma = ms[:,1]

    x_std = (x - x_mu) / x_sigma

    return x_std, w, rho, ndes

def main():
    pwddir = "."
    savedir = pwddir + '/save'
    testdir_x = "../data"
    msdir  = pwddir + "/ms"
    outdir = pwddir + "/out"
    desType = "MLEN_1"
    layer_num = 4
    unit = 300
    actf = "ReLU"
    delta = 0.001
    base = 5
    thr = 1.0e-8
    Model_name = "nn_model1"

    os.makedirs(outdir, exist_ok=True)

    mols = []
    mols.append('1')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    for mol in mols:
        print("test: " + mol)

        test_x = reading_csv_test(testdir_x, mol)
        test_x_std, w, rho, ndes = procdata(test_x, desType, delta, base, msdir)
        test_x_std = test_x_std.reshape(len(test_x_std), ndes)
        test_x_std = torch.from_numpy(test_x_std).float().to(device)

        model = model_setting_test(ndes, unit, layer_num, actf)

        model = model.to(device)
        model.load_state_dict(torch.load(savedir + '/' + Model_name + ".pth", map_location=device))

        with torch.no_grad():
            pred = model(test_x_std)

        pred = pred.to("cpu").detach().numpy()
        pred = inv_trans(pred, delta, base)
        pred = cutoff(rho, pred, thr)
        np.savetxt(outdir + '/' + mol + '_result.csv', pred, delimiter=',')

        pred_w = w * pred
        np.savetxt(outdir + '/' + mol + '_w.csv', pred_w, delimiter=',')

        w = w.T
        toto = w.dot(pred)
        print("### ", toto)

if __name__ == "__main__":
    main()

