from NeuralNetwork import *
from OptimizerGD import *


class Trainer(object):
    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        self.net = net
        self.optimizer = optim
    def fit(self):
        raise NotImplementedError
class Fullbatch(Trainer):
    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        super().__init__(net, optim)
    def fit(self,x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, iters: int, eval_every: int):
        for iter in range(iters):
            loss = self.net.train_batch(x_train,y_train)
            self.optimizer.step()
            if (iter+1)%eval_every==0:
                test_pred = self.net.forward(x_test)
                loss = self.net.loss.forward(test_pred,y_test)
                print('Iter ' + str(iter+1) + ':')
                print('Validation test: ' + str(loss))
class SGD_mini_batch(Trainer):
    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        super().__init__(net, optim)
    def fit(self,x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,batch_size: int, iters: int, eval_every: int):
        for iter in range(iters):
            k = int(x_train.shape[0]/batch_size)
            m = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
            for batch in range(k):
                x_batch = x_train[m[batch*batch_size:(batch+1)*batch_size]]
                y_batch = y_train[m[batch*batch_size:(batch+1)*batch_size]]
                loss = self.net.train_batch(x_batch,y_batch)
                self.optimizer.step()
            if (iter+1)%eval_every==0:
                test_pred = self.net.forward(x_test)
                loss = self.net.loss.forward(test_pred,y_test)
                print('Iter ' + str(iter+1) + ':')
                print('Validation test: ' + str(loss))

        
    
            