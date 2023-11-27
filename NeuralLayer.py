from Operation import *



class Layer(object):
    def __init__(self, neurons: int) -> None:
        self.neurons = neurons
        self.operations: list[Operation] = []
        self.first = True
    def forward(self,input: np.ndarray) -> np.ndarray:
        self.input = input
        if self.first==True:
            self.set_up()
            self.first = False
        for operation in self.operations:
            self.input = operation.forward(self.input)
        return self.input
    def backward(self,output_grad: np.ndarray)-> np.ndarray:
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        return output_grad
    def set_up(self)->None:
        raise NotImplementedError
class Dense(Layer):
    def __init__(self, neurons: int, activation: Operation) -> None:
        super().__init__(neurons)
        self.activation = activation
    def set_up(self) -> None:
        self.operations.append(WeightSum(np.random.rand(self.input.shape[1],self.neurons)))
        self.operations.append(AddBias(np.random.randn(self.neurons,1)))
        self.operations.append(self.activation)
class ConvolutionLayer(Layer):
    def __init__(self, number_kernel: int,kernel_size: tuple,activation: Operation, stride: int = 1, padding: int = 0) -> None:
        super().__init__(number_kernel)
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.height, self.width = kernel_size
    def set_up(self) -> None:
        self.operations.append(Convolutional(np.random.rand(self.input.shape[1],self.neurons,self.height,self.width),self.stride,self.padding))
        output_height = int((self.input.shape[2]-self.height+2*self.padding)/self.stride)+1
        output_width = int((self.input.shape[3]-self.width+2*self.padding)/self.stride)+1
        self.operations.append(ConvolutionalAddBias(np.random.rand(1,self.neurons,output_height,output_width)))
        self.operations.append(self.activation)
class Flatten(Layer):
    def __init__(self) -> None:
        pass
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return input.reshape(input.shape[0],input.shape[1]*input.shape[2]*input.shape[3])
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad.reshape(self.input.shape)
class MaxPoolingLayer(Layer):
    def __init__(self,size: tuple,stride: int = 2) -> None:
        self.operation = MaxPooling(size,stride)
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return self.operation.forward(input)
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        return self.operation.backward(output_grad)


