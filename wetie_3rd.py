
import wandb
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import pandas as pd
import sys, os
sys.path.append("C:/Users/USER")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epoch", type=int, default=10)
parser.add_argument("--hidden1", type=int, default=64)
parser.add_argument("--hidden2", type=int, default=128)
parser.add_argument("--hidden3", type=int, default=64)

args = parser.parse_args()

learning_rate = args.learning_rate
num_epoch = args.num_epoch
hidden1 = args.hidden1
hidden2 = args.hidden2
hidden3 = args.hidden3
batch_size=100

os.chdir("C:/Users/USER")
train_data = pd.read_csv('log.csv')
test_data = train_data.iloc[8001:, :] # 시험 데이터
train_data = train_data.iloc[:8001, :] # 훈련 데이터

train_data = torch.tensor(train_data.values, dtype=torch.float32)
test_data = torch.tensor(test_data.values, dtype=torch.float32)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False
)

model=nn.Sequential(
    nn.Linear(5,hidden1),
    nn.ReLU(),
    nn.Linear(hidden1,hidden2),
    nn.ReLU(),
    nn.Linear(hidden2,hidden3),
    nn.ReLU(),
    nn.Linear(hidden3,1)
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)
model.apply(init_weights)

loss_func=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
epoch_train_loss = []
epoch_test_loss = []

wandb.init(project="wetie_3rd", config=vars(args))
wandb.watch(model, log="all", log_freq=10)

for epoch in range(1,2):
    model.train()

    for batch in train_loader:
        optimizer.zero_grad()
        x, y = batch[:,:-1], batch[:,-1:]
        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[:,:-1], batch[:,-1:]
            output = model(x)


for epoch in range(2, num_epoch + 1):
    model.train()
    train_loss_sum = 0

    for batch in train_loader:
        optimizer.zero_grad()
        x, y = batch[:,:-1], batch[:,-1:]
        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()

    model.eval()
    test_loss_sum = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[:,:-1], batch[:,-1:]
            output = model(x)
            test_loss_sum += loss_func(output, y).item()

    epoch_train_loss.append(train_loss_sum / (batch.shape[0]*len(train_loader)))
    epoch_test_loss.append(test_loss_sum / (batch.shape[0]*len(test_loader)))

    current_train_loss = train_loss_sum / (batch.shape[0] * len(train_loader))
    wandb.log({
        "train_loss": current_train_loss,
    })

import matplotlib.pyplot as plt
plt.plot( range(2, num_epoch+1), epoch_train_loss, label='Train Loss')
plt.plot( range(2, num_epoch+1), epoch_test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss per Epoch')
plt.legend()
plt.grid()
plt.show()
