import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum(t-y)/100

def relu(x):
    return np.maximum(0, x)

class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=1):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size1) * np.sqrt(2 / input_size)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2 / hidden_size1)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = np.random.randn(hidden_size2, output_size) *np.sqrt(2 / hidden_size2)
        self.params['b3'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'],self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'],self.params['b3']
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2  
        z2 = relu(a2)
        y = np.dot(z2, W3) + b3  
        return y

    def loss(self, x, t): 
        y = self.predict(x)
        t = t.reshape(-1, 1)
        mse_loss = mean_squared_error(y, t)
        l2_loss = 0.5 * 0.01 * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2)+np.sum(self.params['W3']**2))
        return mse_loss + l2_loss

    def gradient(self, x, t):
        #순전파
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = relu(a2)
        y = np.dot(z2, W3) + b3
        #역전파
        dy = (y - t.reshape(-1, 1)) / x.shape[0] 
        grads = {}
        grads['W3'] = np.dot(z2.T, dy) + 0.01 * W3  
        grads['b3'] = np.sum(dy, axis=0)
        dz2 = np.dot(dy, W3.T)
        dz2[a2 <= 0] = 0  
        grads['W2'] = np.dot(z1.T, dz2) + 0.01 * W2  
        grads['b2'] = np.sum(dz2, axis=0)
        dz1 = np.dot(dz2, W2.T)  
        dz1[a1 <= 0] = 0  
        grads['W1'] = np.dot(x.T, dz1) + 0.01 * W1
        grads['b1'] = np.sum(dz1, axis=0)

        return grads  