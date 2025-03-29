#1. module import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("C:/Users/USER")
from mlp import MLP

# 2. data setting
os.chdir("C:/Users/USER")
train_data = pd.read_csv('log.csv')
test_data = train_data.iloc[8001:, :] # 시험 데이터
train_data = train_data.iloc[:8001, :] # 훈련 데이터

x_train = train_data.iloc[:, 0:-1].to_numpy()
t_train = train_data.iloc[:, -1].to_numpy().astype(np.float64).reshape(-1, 1)
x_test = test_data.iloc[:, 0:-1].to_numpy()
t_test = test_data.iloc[:, -1].to_numpy().astype(np.float64).reshape(-1, 1)

# 하이퍼파라미터 설정
iters_num = 10000
learning_rate = 0.035
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = min(100, train_size)
network = MLP(input_size=5, hidden_size1=25, hidden_size2=35, output_size=1)

# 5. optimization function
class Adam:
    def __init__(self, lr=0.035, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

optimizer = Adam(lr=0.035)

# 6. training
train_loss_list = []
test_loss_list = []
epoch_list1= []
epoch_list2= []
iter_per_epoch = max(train_size // batch_size, 1)
iter_per_epoch2 = max(test_size // batch_size, 1)

for i in range(iters_num):
    # 미니배치 학습
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grad)
    loss = network.loss(x_batch, t_batch)

    #7. eveluation
    if i % iter_per_epoch == 0:
        epoch1= i // iter_per_epoch
        train_loss = network.loss(x_batch, t_batch)
        train_loss_list.append((train_loss))
        epoch_list1.append(epoch1)
        print(f"epoch:{epoch1}, train_loss:{train_loss}")

for i in range(2000):
    batch_mask2= np.random.choice(test_size, batch_size)
    x_batch2 = x_test[batch_mask2]
    t_batch2 = t_test[batch_mask2]

    grad = network.gradient(x_batch2, t_batch2)
    optimizer.update(network.params, grad)
    loss = network.loss(x_batch2, t_batch2)

    #7. eveluation
    if i % iter_per_epoch2 == 0:
        test_loss = network.loss(x_batch2, t_batch2)
        epoch2 = i // iter_per_epoch2
        test_loss_list.append((test_loss))
        epoch_list2.append(epoch2)
        print(f"epoch:{epoch2}, test_loss:{test_loss}")


#8. test result
plt.plot(epoch_list1, train_loss_list, label='Train Loss')
plt.plot(epoch_list2, test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss per Epoch')
plt.legend()
plt.grid()
plt.show()

plt.plot(epoch_lis2t, test_loss_list, label='Test Loss')

