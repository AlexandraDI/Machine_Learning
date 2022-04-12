import numpy as np
from typing import Optional, Union
from functions import Sigmoid

class Autoencoder:
    def __init__(self):
        self._w1 = np.random.rand(3,9)
        self._w2 = np.random.rand(8,4)
        self._h = np.random.rand(4, 1)
        self._o = np.random.rand(8, 1)
        self._g = Sigmoid()
        
        
    def train(self, x, y, a, l, max_iter = 1000000, err = 0.01, a_reduction = 0, reduction_steps=100):      
        js = [0 for i in range(max_iter)]
        x = np.vstack([[1 for i in range(x.shape[1])], x])
        
        mask1 = np.ones(self._w1.shape)
        mask1[:,0] = 0
        mask2 = np.ones(self._w2.shape)
        mask2[:,0] = 0
        
        for i in range(max_iter):         
            d1 = np.zeros(self._w1.shape)
            d2 = np.zeros(self._w2.shape)
            for k in range(x.shape[1]):
                xk = x[:, k].reshape(-1, 1)
                yk = y[:, k].reshape(-1, 1)
                #print("xk:", xk.shape)
                tmp = self._backward(xk, yk)
                d1 += tmp[0]
                d2 += tmp[1]
            d1 = (d1 + l * mask1 * self._w1) / xk.shape[1] 
            d2 = (d2 + l * mask2 * self._w2) / xk.shape[1]
            
            self._w1 -= a * d1
            self._w2 -= a * d2
            
            j = np.linalg.norm(self._o - yk)
            js[i] = j
            
            if j < err:
                print("Exit at iteration:", i)
                break 
            #if ((i + 1) % 1000) == 0:
            #    print(f"Iteration {i+1}: {j}")
            
            if ((i+1) % reduction_steps) == 0:
                a -= a * a_reduction
        return js, i
        
    def _forward(self, x):
        #print("x-forward:", x.shape)
        #print(np.matmul(self._w1, x).shape)
        tmp = self._g(np.matmul(self._w1, x)).reshape(3, 1)
        #print(tmp.shape)
        self._h = np.vstack([1, tmp])
        self._o = self._g(np.matmul(self._w2, self._h)).reshape(8, 1)   
    
    def _backward(self, x, y):
        d3 = np.ndarray((8,1))
        d2 = np.ndarray((4,1))
        
        self._forward(x)

        d3 = (self._o * (1 - self._o) * (self._o - y)).reshape(8,1)
        d2 = self._h * (1 - self._h) * np.matmul(self._w2.T, d3).reshape(4,1)
        
        delta1 = np.matmul(d2[1:, :], x.T)
        delta2 = np.matmul(d3, self._h.T)
        #compute Delta (1 example)
        return delta1, delta2
    
    def predict(self, x):
        self._forward(np.vstack([1, x.reshape(8,1)]))
        return self._o