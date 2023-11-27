from main_h import *
from NeuralNetwork import *

class Optimizer(object):
    def __init__(self,lr: float, net: NeuralNetwork) -> None:
        self.lr = lr
        self.net = net
    def step(self):
        raise NotImplementedError
class NormalGD(Optimizer):
    def __init__(self, lr: float,net: NeuralNetwork) -> None:
        super().__init__(lr,net)
    def step(self):
        for layer in self.net.layers:
            if type(layer)!=MaxPoolingLayer and type(layer)!=Flatten:
                for operation in layer.operations:
                    if type(operation)==WeightSum or type(operation)==AddBias:
                        operation.param-=self.lr*operation.param_grad
class MomentumGD(Optimizer):
    def __init__(self, lr: float, net: NeuralNetwork) -> None:
        super().__init__(lr, net)
        self.first = True
    def step(self):
        if self.first == True:
            self.v = []
            for layer in self.net.layers:
                if type(layer)!=MaxPoolingLayer and type(layer)!=Flatten:
                    for operation in layer.operations:
                        if type(operation)==WeightSum or type(operation)==AddBias:
                            self.v.append(np.zeros_like(operation.param))
            self.first = False
        count = 0
        for layer in self.net.layers:
            if type(layer)!=MaxPoolingLayer and type(layer)!=Flatten:
                for operation in layer.operations:
                    if type(operation)==WeightSum or type(operation)==AddBias:
                        self.v[count] = self.v[count]*0.9+self.lr*operation.param_grad
                        operation.param -= self.v[count]
                        count+=1
                    
class AdaGrad(Optimizer):
    def __init__(self, lr: float, net: NeuralNetwork) -> None:
        super().__init__(lr, net)
        self.first = True
    def step(self):
        if self.first==True:
            self.G = []
            for layer in self.net.layers:
                if type(layer)!=MaxPoolingLayer and type(layer)!=Flatten:
                    for operation in layer.operations:
                        if type(operation)==WeightSum or type(operation)==AddBias:
                            self.G.append(np.zeros_like(operation.param))
            self.first = False
        count = 0
        for layer in self.net.layers:
            if type(layer)!=MaxPoolingLayer and type(layer)!=Flatten:
                for operation in layer.operations:
                    if type(operation)==WeightSum or type(operation)==AddBias:
                        self.G[count]+=operation.param_grad
                        operation.param -= self.lr*operation.param_grad*(1/np.sqrt((self.G[count])**2+1e-6))
                        count+=1
class RMSProp(Optimizer):
    def __init__(self, lr: float, net: NeuralNetwork) -> None:
        super().__init__(lr, net)
        self.first = True
    def step(self):
        if self.first==True:
            self.G = []
            for layer in self.net.layers:
                if type(layer)!=MaxPoolingLayer and type(layer)!=Flatten:
                    for operation in layer.operations:
                        if type(operation)==WeightSum or type(operation)==AddBias:
                            self.G.append(np.zeros_like(operation.param))
            self.first = False
        count = 0
        for layer in self.net.layers:
            if type(layer)!=MaxPoolingLayer and type(layer)!=Flatten:
                for operation in layer.operations:
                    if type(operation)==WeightSum or type(operation)==AddBias:
                        self.G[count]= 0.9*self.G[count]+0.1*(operation.param_grad)**2
                        operation.param -= self.lr*operation.param_grad*(1/np.sqrt(self.G[count]+1e-6))
                        count+=1
        