import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tqdm

batch_size=8
num_epoch=20
learning_rate=0.001
num_class=2
train_path="C:\\Users\\Lenovo\\Desktop\\1\\for_DF\\1\\train_dataset.xlsx"
valid_path="C:\\Users\\Lenovo\\Desktop\\1\\for_DF\\1\\valid_dataset.csv"
report_steps=20
output_model_path="C:\\Users\\Lenovo\\Desktop\\1\\srt\\FS-Net\\models\\df.params"
alpha=0.01

dP = torch.zeros((batch_size, 30, 8))
dX = torch.arange(30, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, 8, 2, dtype=torch.float32) / 8)
for i in range(0,batch_size):
    dP[i, :, 0::2] = torch.sin(dX)
    dP[i, :, 1::2] = torch.cos(dX)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dP=dP.to(device)

class DF_Network(nn.Module):
    def __init__(self):
        super(DF_Network, self).__init__()
        self.emb=nn.Embedding(256,8)
        self.enc = nn.GRU(8,8,2,False,True,0.2,True)
        self.dec = nn.GRU(8,8,2,False,True,0.2,True)
        self.recon=nn.Linear(64, 30)
        self.out = nn.Sequential(
            nn.ELU(),                     
            nn.Flatten(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_class)
        )
    def forward(self, x):
        x=self.emb(x)
        x=x+dP
        enc_out,dec_input = self.enc(x)
        dec_input=dec_input.transpose(0,1)
        dec_out,dec_output = self.dec(dec_input)
        dec_output=dec_output.transpose(0,1)
        x_recon=self.recon(dec_out.reshape(batch_size,-1))
        product=dec_input*dec_output
        absolute=torch.abs(dec_input-dec_output)
        final=torch.cat([dec_input,dec_output,product,absolute],dim=1).reshape(batch_size,-1)
        output = self.out(final)
        return nn.LogSoftmax(dim=-1)(output),x_recon

def batch_loader(data,label):
    instances_num = data.size()[0]
    for i in range(instances_num // batch_size):
        data_batch = data[i * batch_size : (i + 1) * batch_size, :]
        label_batch = label[i * batch_size : (i + 1) * batch_size]
        yield data_batch,label_batch
    if instances_num > instances_num // batch_size * batch_size+1:
        data_batch = data[instances_num // batch_size * batch_size :, :]
        label_batch = label[instances_num // batch_size * batch_size :]
        yield data_batch,label_batch

def train_model(device, model, optimizer, data_batch, label_batch):
    model.zero_grad()
    data_batch=data_batch
    data_batch = data_batch.to(device)
    label_batch = label_batch.to(device)
    pre,x_pre= model(data_batch)
    loss = nn.NLLLoss()(pre, label_batch.view(-1).long())
    loss_x=nn.MSELoss()(data_batch.reshape(-1), x_pre.reshape(-1))
    loss=loss+loss_x*alpha
    loss.backward() 
    optimizer.step()
    return loss

def evaluate(device,model,data_va,label_va):
    correct = 0
    model.eval()
    for i, (data_va_batch,label_va_batch) in enumerate(batch_loader(data_va,label_va)):
        data_va_batch=data_va_batch
        data_va_batch = data_va_batch.to(device)
        label_va_batch = label_va_batch.to(device)
        with torch.no_grad():
            pre,x_pre = model(data_va_batch)
        pred = torch.argmax(pre, dim=1)
        gold = label_va_batch 
        correct += torch.sum(pred == gold).item()
    return correct / len(label_va)

def main():
    model=DF_Network()

    for n, p in list(model.named_parameters()):
        if "gamma" not in n and "beta" not in n:
            p.data.normal_(0, 0.02)  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    data = pd.read_excel(train_path)
    #data = pd.read_csv(train_path,error_bad_lines=False)
    data=np.array(data.fillna(0))
    label=torch.tensor(data[:,0],dtype=torch.int)
    data=torch.Tensor(data[:,1:31]//10).int()

    data_va = pd.read_csv(valid_path,error_bad_lines=False)
    data_va=np.array(data_va.fillna(0))
    label_va=torch.tensor(data_va[:,0],dtype=torch.int)
    data_va=torch.Tensor(data_va[:,1:31]//10).int()

    optimizer=torch.optim.RMSprop(model.parameters(),lr=learning_rate)

    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")

    for epoch in tqdm.tqdm(range(1, num_epoch + 1)):

        model.train()

        for i, (data_batch, label_batch) in enumerate(batch_loader(data, label)):
            loss = train_model(device, model, optimizer, data_batch, label_batch)
            total_loss += loss.item()
            if (i + 1) % report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / report_steps))
                total_loss = 0.0

        result = evaluate(device,model,data_va,label_va)
        if result > best_result:
            best_result = result
            torch.save(model.state_dict(), output_model_path)

if __name__ == "__main__":
    main()