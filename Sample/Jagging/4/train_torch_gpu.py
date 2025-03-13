import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import os

"""
setting class or function
"""

class NN(nn.Module):
    def __init__(self,ndes,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,layer_num,actf):
        super(NN, self).__init__()

        self.lan = layer_num
        self.actf = actf

        self.layer1 = nn.Linear(ndes,u1)

        if (self.lan<2):
            self.out_layer = nn.Linear(u1,1)
        else:
            self.layer2 = nn.Linear(u1,u2)
            if (self.lan<3):
                self.out_layer = nn.Linear(u2,1)
            else:
                self.layer3 = nn.Linear(u2,u3)
                if (self.lan<4):
                    self.out_layer = nn.Linear(u3,1)
                else:
                    self.layer4 = nn.Linear(u3,u4)
                    if (self.lan<5):
                        self.out_layer = nn.Linear(u4,1)
                    else:
                        self.layer5 = nn.Linear(u4,u5)
                        if (self.lan<6):
                            self.out_layer = nn.Linear(u5,1)
                        else:
                            self.layer6 = nn.Linear(u5,u6)
                            if (self.lan<7):
                                self.out_layer = nn.Linear(u6,1)
                            else:
                                self.layer7 = nn.Linear(u6,u7)
                                if (self.lan<8):
                                    self.out_layer = nn.Linear(u7,1)
                                else:
                                    self.layer8 = nn.Linear(u7,u8)
                                    if (self.lan<9):
                                        self.out_layer = nn.Linear(u8,1)
                                    else:
                                        self.layer9 = nn.Linear(u8,u9)
                                        if (self.lan<10):
                                            self.out_layer = nn.Linear(u9,1)
                                        else:
                                            self.layer10 = nn.Linear(u9,u10)
                                            if (self.lan == 10):
                                                self.out_layer = nn.Linear(u10,1)

        if actf == "ELU":
            self.act = nn.ELU()
        elif actf == "Hardshrink":
            self.act = nn.Hardshrink()
        elif actf == "Hardtanh":
            self.act = nn.Hardtanh()
        elif actf == "LeakyReLU":
            self.act = nn.LeakyReLU()
        elif actf == "LogSigmoid":
            self.act = nn.LogSigmoid()
        elif actf == "MultiheadAttention":
            self.act = nn.MultiheadAttention()
        elif actf == "PReLU":
            self.act = nn.PReLU()
        elif actf == "ReLU":
            self.act = nn.ReLU()
        elif actf == "ReLU6":
            self.act = nn.ReLU6()
        elif actf == "RReLU":
            self.act = nn.RReLU()
        elif actf == "SELU":
            self.act = nn.SELU()
        elif actf == "CELU":
            self.act = nn.CELU()
        elif actf == "GELU":
            self.act = nn.GELU()
        elif actf == "Sigmoid":
            self.act = nn.Sigmoid()
        elif actf == "Softplus":
            self.act = nn.Softplus()
        elif actf == "Softshrink":
            self.act = nn.Softshrink()
        elif actf == "Softsign":
            self.act = nn.Softsign()
        elif actf == "Tanh":
            self.act = nn.Tanh()
        elif actf == "Tanhshrink":
            self.act = nn.Tanhshrink()
        elif actf == "Threshold":
            self.act = nn.Threshold()
        elif actf == "Softmin":
            self.act = nn.Softmin()
        elif actf == "Softmax":
            self.act = nn.Softmax()
        elif actf == "Softmax2d":
            self.act = nn.Softmax2d()
        elif actf == "LogSoftmax":
            self.act = nn.LogSoftmax()
        elif actf == "AdaptiveLogSoftmaxWithLoss":
            self.act = nn.AdaptiveLogSoftmaxWithLoss()

    def forward(self, x):
        h1 = self.act(self.layer1(x))
        if (self.lan<2):
            h_out = self.out_layer(h1)
        else:
            h2 = self.act(self.layer2(h1))
            if (self.lan<3):
                h_out = self.out_layer(h2)
            else:
                h3 = self.act(self.layer3(h2))
                if (self.lan<4):
                    h_out = self.out_layer(h3)
                else:
                    h4 = self.act(self.layer4(h3))
                    if (self.lan<5):
                        h_out = self.out_layer(h4)
                    else:
                        h5 = self.act(self.layer5(h4))
                        if (self.lan<6):
                            h_out = self.out_layer(h5)
                        else:
                            h6 = self.act(self.layer6(h5))
                            if (self.lan<7):
                                h_out = self.out_layer(h6)
                            else:
                                h7 = self.act(self.layer7(h6))
                                if (self.lan<8):
                                    h_out = self.out_layer(h7)
                                else:
                                    h8 = self.act(self.layer8(h7))
                                    if (self.lan<9):
                                        h_out = self.out_layer(h8)
                                    else:
                                        h9 = self.act(self.layer9(h8))
                                        if (self.lan<10):
                                            h_out = self.out_layer(h9)
                                        else:
                                            h10 = self.layer10(h9)
                                            if (self.lan==10):
                                                h_out = self.out_layer(h10)

        return h_out

def train_step(model, optimizer, criterion, train_x, train_y):
    model.train()

    pred_y = model(train_x)
    optimizer.zero_grad()
    loss = criterion(pred_y, train_y)
    loss.backward()

    optimizer.step()

    return loss.item()

def valid_step(model, criterion, valid_x, valid_y):
    model.eval()

    pred_y = model(valid_x)
    loss = criterion(pred_y, valid_y)

    return loss.item()

def model_setting(ndes, unit, layer_num, actf, opt, lr, loss_function):
    model = None
    optimizer = None
    criterion = None

    if (layer_num == 1):
        model = NN(ndes,unit,0,0,0,0,0,0,0,0,0,layer_num,actf)
    if (layer_num == 2):
        model = NN(ndes,unit,unit,0,0,0,0,0,0,0,0,layer_num,actf)
    if (layer_num == 3):
        model = NN(ndes,unit,unit,unit,0,0,0,0,0,0,0,layer_num,actf)
    if (layer_num == 4):
        model = NN(ndes,unit,unit,unit,unit,0,0,0,0,0,0,layer_num,actf)
    if (layer_num == 5):
        model = NN(ndes,unit,unit,unit,unit,unit,0,0,0,0,0,layer_num,actf)
    if (layer_num == 6):
        model = NN(ndes,unit,unit,unit,unit,unit,unit,0,0,0,0,layer_num,actf)
    if (layer_num == 7):
        model = NN(ndes,unit,unit,unit,unit,unit,unit,unit,0,0,0,layer_num,actf)
    if (layer_num == 8):
        model = NN(ndes,unit,unit,unit,unit,unit,unit,unit,unit,0,0,layer_num,actf)
    if (layer_num == 9):
        model = NN(ndes,unit,unit,unit,unit,unit,unit,unit,unit,unit,0,layer_num,actf)
    if (layer_num == 10):
        model = NN(ndes,unit,unit,unit,unit,unit,unit,unit,unit,unit,unit,layer_num,actf)

    lr = lr
    if (opt == "Adadelta"):
        optimizer = optim.Adadelta(model.parameters(),lr=lr)
    if (opt == "Adagrad"):
        optimizer = optim.Adagrad(model.parameters(),lr=lr)
    if (opt == "Adam"):
        optimizer = optim.Adam(model.parameters(),lr=lr)
    if (opt == "AdamW"):
        optimizer = optim.AdamW(model.parameters(),lr=lr)
    if (opt == "SparseAdam"):
        optimizer = optim.SparseAdam(model.parameters(),lr=lr)
    if (opt == "Adamax"):
        optimizer = optim.Adamax(model.parameters(),lr=lr)
    if (opt == "ASGD"):
        optimizer = optim.ASGD(model.parameters(),lr=lr)
    if (opt == "LBFGS"):
        optimizer = optim.LBFGS(model.parameters(),lr=lr)
    if (opt == "RMSprop"):
        optimizer = optim.RMSprop(model.parameters(),lr=lr)
    if (opt == "Rprop"):
        optimizer = optim.Rprop(model.parameters(),lr=lr)
    if (opt == "SGD"):
        optimizer = optim.SGD(model.parameters(),lr=lr)

    if (loss_function == "L1Loss"):
        criterion = nn.L1Loss()
    if (loss_function == "MSELoss"):
        criterion = nn.MSELoss()
    if (loss_function == "CrossEntropyLoss"):
        criterion = nn.CrossEntropyLoss()
    if (loss_function == "CTCLoss"):
        criterion = nn.CTCLoss()
    if (loss_function == "NLLLoss"):
        criterion = nn.NLLLoss()
    if (loss_function == "PoissonNLLLoss"):
        criterion = nn.PoissonNLLLoss()
    if (loss_function == "KLDivLoss"):
        criterion = nn.KLDivLoss()
    if (loss_function == "BCELoss"):
        criterion = nn.BCELoss()
    if (loss_function == "BCEWithLogitsLoss"):
        criterion = nn.BCEWithLogitsLoss()
    if (loss_function == "MarginRankingLoss"):
        criterion = nn.MarginRankingLoss()
    if (loss_function == "HingeEmbeddingLoss"):
        criterion = nn.HingeEmbeddingLoss()
    if (loss_function == "MultiLabelMarginLoss"):
        criterion = nn.MultiLabelMarginLoss()
    if (loss_function == "SmoothL1Loss"):
        criterion = nn.SmoothL1Loss()
    if (loss_function == "SoftMarginLoss"):
        criterion = nn.SoftMarginLoss()
    if (loss_function == "MultiLabelSoftMarginLoss"):
        criterion = nn.MultiLabelSoftMarginLoss()
    if (loss_function == "CosineEmbeddingLoss"):
        criterion = nn.CosineEmbeddingLoss()
    if (loss_function == "MultiMarginLoss"):
        criterion = nn.MultiMarginLoss()
    if (loss_function == "TripletMarginLoss"):
        criterion = nn.TripletMarginLoss()

    return model, optimizer, criterion

def reading_csv(csvdir, typ='train,valid', name=''):
    result = {}

    if "train" in typ:
        data = np.loadtxt(csvdir+'/train.csv', delimiter=",", skiprows=1)
        result["train_x"] = data[:, [4, 5, 6, 7]]  # rho, grad, tau, exchange
        result["train_y"] = data[:, 8]  # reference
    if "valid" in typ:
        data = np.loadtxt(csvdir+'/valid.csv', delimiter=",", skiprows=1)
        result["valid_x"] = data[:, [4, 5, 6, 7]]  # rho, grad, tau, exchange
        result["valid_y"] = data[:, 8]  # reference

    return result

def prepro(x,y,d,b):

    x = x + d
    x = np.log(x)/np.log(b)
    y = d - y
    y = np.log(y)/np.log(b)

    return x, y

def procdata(d, y, desType, delta, base, trteflg, msdir):
    d[:,3] = -d[:,3]

    # for jagging 
    if desType == "MLEN_1":
        x   = d[:,0]
        ndes = 1
    if desType == "MLEN_2":
        x   = d[:,1]
        ndes = 1
    if desType == "MLEN_3":
        x   = d[:,2]
        ndes = 1
    if desType == "MLEN_4":
        x   = d[:,3]
        ndes = 1
    if desType == "MLEN_5":
        x   = d[:,[0, 1]]
        ndes = 2
    if desType == "MLEN_6":
        x   = d[:,[0, 2]]
        ndes = 2
    if desType == "MLEN_7":
        x   = d[:,[0, 3]]
        ndes = 2
    if desType == "MLEN_8":
        x   = d[:,[1, 2]]
        ndes = 2
    if desType == "MLEN_9":
        x   = d[:,[1, 3]]
        ndes = 2
    if desType == "MLEN_40":
        x   = d[:,[2, 3]]
        ndes = 2
    if desType == "MLEN_41":
        x   = d[:,[0, 1, 2]]
        ndes = 3
    if desType == "MLEN_42":
        x   = d[:,[0, 1, 3]]
        ndes = 3
    if desType == "MLEN_43":
        x   = d[:,[0, 2, 3]]
        ndes = 3
    if desType == "MLEN_44":
        x   = d[:,[1, 2, 3]]
        ndes = 3
    if desType == "MLEN_45":
        x   = d[:,[0, 1, 2, 3]]
        ndes = 4

    x, y = prepro(x, y, delta, base)
    x, y = x.astype(np.float64), y.astype(np.float64)
    
    if trteflg == "train":
        x_mu = np.mean(x, axis=0)
        x_sigma = np.std(x, axis=0)
        x_std = (x-x_mu)/x_sigma

        if ndes == 1:
            ms = np.array([x_mu, x_sigma])

        else:
            ms = np.concatenate((x_mu.reshape((len(x_mu),1)),
                                 x_sigma.reshape((len(x_sigma),1))),axis=1)
        
        np.save(msdir+'/ms.npy',ms)
        
    if trteflg == "test":
        with open(msdir+"/ms.npy","rb") as fms:
            ms = np.load(msdir+'/ms.npy')
            
        if ndes == 1:
            x_mu = ms[0]
            x_sigma = ms[1]
        else:
            x_mu = ms[:,0]
            x_sigma = ms[:,1]
        x_std = (x-x_mu)/x_sigma
        
    return x_std, y, ndes

def train_model(model, optimizer, criterion, loader_train, loader_valid, epochs, device, prev_epochs=0, model_name=None, save_path=None):
    train_history = []
    valid_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_val_loss = 0.0
        total_train = 0
        total_valid = 0
        
        for train_x, train_y in loader_train:
            loss = train_step(model, optimizer, criterion, train_x, train_y)
            
            total_loss += loss
            total_train += len(train_y)
            
        for valid_x, valid_y in loader_valid:
            val_loss = valid_step(model, criterion, valid_x, valid_y)
            
            total_val_loss += val_loss
            total_valid += len(valid_y)
            
        n = epoch + 1 + prev_epochs
        avg_loss = total_loss / n
        avg_val_loss = total_val_loss / n
        
        train_history.append(avg_loss)
        valid_history.append(avg_val_loss)
        
        print(f'[Epoch {epoch+1:4d}/{epochs:4d}]'\
              f'loss: {avg_loss:.8f}, val_loss: {avg_val_loss:.8f}')
    
    if model_name and save_path:
        torch.save(model.to("cpu").state_dict(), save_path+'/'+model_name+'.pth')
        model = model.to(device)
    
    return model, train_history, valid_history

def main():
    os.makedirs('ms', exist_ok=True)
    os.makedirs('save', exist_ok=True)
    os.makedirs('out', exist_ok=True)

    """
    param setting
    """

    pwddir = "/home/haru/github/MLEC-AD/Sample/Jagging/4/"
    savedir = pwddir + '/save'
    csvdir = "/home/haru/github/MLEC-AD/Sample/Jagging/data"
    msdir  = pwddir + "/ms"
    desType = "MLEN_4"
    layer_num = 4
    unit = 300
    actf = "ReLU"
    optim_first = "Adam"
    optim_second = "Adagrad"
    epoch_first = 17000
    epoch_second = 3000
    epoch_first = 1
    epoch_second = 1
    loss_function = "MSELoss"
    delta = 0.001
    base = 5
    batch_size = 1024
    lr = 0.001
    Model_name = "nn_model1"

    """
    preparation
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    csv_data = reading_csv(csvdir, typ="train,valid")
    train_x, train_y = csv_data["train_x"], csv_data["train_y"]
    valid_x, valid_y = csv_data["valid_x"], csv_data["valid_y"]

    train_x, train_y, ndes = procdata(train_x, train_y, desType, delta, base, "train", msdir)
    valid_x, valid_y, ndes = procdata(valid_x, valid_y, desType, delta, base, "test", msdir)

    train_x, valid_x = train_x.reshape(len(train_x),ndes), valid_x.reshape(len(valid_x),ndes)
    train_y, valid_y = train_y.reshape(len(train_y),1), valid_y.reshape(len(valid_y),1)
    N_tr,N_te = len(train_y), len(valid_y)

    print("Trainig Data : ",N_tr)
    print("Test Data : ",N_te)

    """
    data loader
    """

    train_x = torch.from_numpy(train_x).float().to(device)
    train_y = torch.from_numpy(train_y).float().to(device)
    valid_x = torch.from_numpy(valid_x).float().to(device)
    valid_y = torch.from_numpy(valid_y).float().to(device)

    dataset_train = TensorDataset(train_x, train_y)
    dataset_valid = TensorDataset(valid_x, valid_y)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size)

    """
    model setting first
    """

    model1, optimizer1, criterion1 = model_setting(ndes, unit, layer_num, actf, optim_first, lr, loss_function)
    model1 = model1.to(device)

    """
    start learning first
    """
    print('Starting First Training')
    model1, train_history1, valid_history1 = train_model(
        model1, optimizer1, criterion1, 
        loader_train, loader_valid, 
        epoch_first, 
        device,
        model_name=Model_name+'_first', 
        save_path=savedir
    )
    print('Finished First Training')

    """
    model setting second
    """

    model2, optimizer2, criterion2 = model_setting(ndes, unit, layer_num, actf, optim_second, lr, loss_function)
    model2 = model2.to(device)
    model2.load_state_dict(torch.load(savedir+'/'+Model_name+'_first.pth', map_location=device))
    for state in optimizer2.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    """
    start learning second
    """
    print('Starting Second Training')
    model2, train_history2, valid_history2 = train_model(
        model2, optimizer2, criterion2, 
        loader_train, loader_valid, 
        epoch_second, 
        device,
        prev_epochs=epoch_first,
        model_name=Model_name+'_second', 
        save_path=savedir
    )
    print('Finished Second Training')

if __name__ == "__main__":
    main() 
