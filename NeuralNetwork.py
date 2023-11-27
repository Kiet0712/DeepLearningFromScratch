
from NeuralLayer import *
from LossFunction import *

class NeuralNetwork(object):
    def __init__(self, layers: list[Layer], loss: Loss) -> None:
        self.layers = layers
        self.loss = loss
    def forward(self,x_batch: np.ndarray)->np.ndarray:
        for layer in self.layers:
            x_batch = layer.forward(x_batch)
        return x_batch
    def backward(self,loss_grad:np.ndarray):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
    def train_batch(self,x_batch: np.ndarray,y_batch:np.ndarray):
        predict = self.forward(x_batch)
        loss_value = self.loss.forward(predict,y_batch)
        grad_loss = self.loss._input_grad()
        self.backward(grad_loss)
        return loss_value
class CNN(NeuralNetwork):
    def __init__(self, layers: list[Layer], loss: Loss) -> None:
        super().__init__(layers, loss)
    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        return super().forward(x_batch)
    def backward(self, loss_grad: np.ndarray):
        return super().backward(loss_grad)
    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray):
        return super().train_batch(x_batch, y_batch)
