from main_h import *

def softmax(x: np.ndarray) -> np.ndarray:
    c = np.max(x,axis=1)
    e_z_temp = np.exp(x-c.reshape(x.shape[0],1))
    sum_vec = np.sum(e_z_temp,axis=1) 
    return e_z_temp/sum_vec.reshape(x.shape[0],1)
class Loss(object):
    def __init__(self) -> None:
        pass
    def forward(self,predict:np.ndarray,real: np.ndarray):
        self.predict = predict
        self.real = real
        return self._output()
    def backward(self):
        return self._input_grad()
    def _output(self):
        raise NotImplementedError
    def _input_grad(self):
        raise NotImplementedError
class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()
    def _output(self):
        N = self.predict.shape[0]
        return (np.linalg.norm(self.predict-self.real))**2/N
    def _input_grad(self):
        N = self.predict.shape[0]
        return (2*self.predict-2*self.real)/N
class SoftMax(Loss):
    def __init__(self) -> None:
        super().__init__()
    def _output(self):
        self.softmax_predict = softmax(self.predict)
        entropy = -self.real*np.log(self.softmax_predict+1e-6)
        return np.sum(entropy)
    def _input_grad(self):
        return self.softmax_predict-self.real

    
