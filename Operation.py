
from main_h import *
from scipy.signal import correlate2d
from numpy.lib.stride_tricks import as_strided
import torch
class Operation(object):
    def __init__(self) -> None:
        pass
    def forward(self,input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = self._output()
        return self.output
    def backward(self,output_grad: np.ndarray) -> np.ndarray:
        self.input_grad = self._input_grad(output_grad)
        return self.input_grad
    def _output(self)-> np.ndarray:
        raise NotImplementedError
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
#ParameterOperation
class ParamOperation(Operation):
    def __init__(self, param: np.ndarray) -> None:
        super().__init__()
        self.param = param
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        return self.input_grad
    def _param_grad(self,output_grad: np.ndarray)->np.ndarray:
        raise NotImplementedError
class WeightSum(ParamOperation):
    def __init__(self, w: np.ndarray) -> None:
        super().__init__(w)
    def _output(self) -> np.ndarray:
        return self.input@self.param
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad@self.param.T
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return self.input.T@output_grad
class AddBias(ParamOperation):
    def __init__(self, b: np.ndarray) -> None:
        super().__init__(b)
    def _output(self) -> np.ndarray:
        return self.input+np.ones((self.input.shape[0],1))@self.param.T
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.input)*output_grad
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad.T@np.ones((self.input.shape[0],1))
#Activation
class Linear(Operation):
    def __init__(self) -> None:
        super().__init__()
    def _output(self) -> np.ndarray:
        return self.input
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad
class Sigmoid(Operation):
    def __init__(self) -> None:
        super().__init__()
    def _output(self) -> np.ndarray:
        return 1/(1+np.exp(-self.input))
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return self.output*(1-self.output)*output_grad
class Relu(Operation):
    def __init__(self) -> None:
        super().__init__()
    def _output(self) -> np.ndarray:
        return np.maximum(0,self.input)
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(self.input)
        grad[self.input>=0] = 1
        return grad*output_grad
class tanh(Operation):
    def __init__(self) -> None:
        super().__init__()
    def _output(self) -> np.ndarray:
        return (np.exp(2*self.input)-1)/(np.exp(2*self.input)+1)
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return (4*(np.exp(self.input)**2)/(np.exp(2*self.input)+1))*output_grad
class LeakyRelu(Operation):
    def __init__(self) -> None:
        super().__init__()
    def _output(self) -> np.ndarray:
        return np.maximum(0.1*self.input,self.input)
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        grad = np.ones_like(self.input)
        grad[0.1*self.input>=self.input] = 0.1
        return grad*output_grad
    


#convolutional neuralnetwork
def convo2D(input: np.ndarray,kernel: np.ndarray,stride: int = 1, padding_size: int = 0,check=False):
    number_image_in, depth_in, height_in, width_in = input.shape
    depth_kernel, number_kernel, height_kernel, width_kernel = kernel.shape
    add_padd = np.zeros((number_image_in,depth_in,height_in+2*padding_size,width_in+2*padding_size))
    if padding_size !=0:
        add_padd[:,:,padding_size:-padding_size,padding_size:-padding_size] = input[:,:]
    else:
        add_padd = input
    # output_width = int((width_in-width_kernel+2*padding_size)/stride)+1
    # output_height = int((height_in-height_kernel+2*padding_size)/stride)+1
    # output_depth = number_kernel
    # output  = np.zeros((number_image_in,output_depth,output_height,output_width))
    # for i in range(number_image_in):
    #     for j in range(number_kernel):
    #         for depth in range(depth_in):
    #             # for y in range(add_padd.shape[2]):
    #             #     if y > add_padd.shape[2]-height_kernel:
    #             #         break
    #             #     if y%stride == 0:
    #             #         for x in range(add_padd.shape[3]):
    #             #             if x > add_padd.shape[3]-width_kernel:
    #             #                 break
    #             #             if x%stride == 0:
    #             #                 output[i,j,int(y/stride),int(x/stride)] += np.sum(kernel[depth,j]*add_padd[i,depth,y:y+height_kernel,x:x+width_kernel])
    #             output[i,j] += correlate(add_padd[i,depth],kernel[depth,j],'valid')[::stride,::stride]
    add_padd_ = add_padd.copy()
    kernel_ = kernel.copy()
    output = torch.nn.functional.conv2d(torch.from_numpy(add_padd_).float(),torch.from_numpy(np.transpose(kernel_,[1,0,2,3])).float(),None,(int(stride),int(stride)),'valid')
    if check == True:
      return add_padd,output.numpy()
    else:
      return output.numpy()
def rotate180(input: np.ndarray):
    return np.rot90(input,2,[-2,-1])
def addCenter(arr, stride):
    stride = stride-1
    dim1, dim2, rows, cols = arr.shape
    new_rows = rows + (rows - 1) * stride
    new_cols = cols + (cols - 1) * stride
    result = np.zeros((dim1, dim2, new_rows, new_cols),dtype=arr.dtype)
    result[:, :, ::stride+1, ::stride+1] = arr
    return result
def transpose(input: np.ndarray):
    return np.transpose(input,[1,0,2,3])
def pool2D(A: np.ndarray,kernel_size: tuple,stride: int):
    output_shape = ((A.shape[0] - kernel_size[0]) // stride + 1,
                    (A.shape[1] - kernel_size[1]) // stride + 1)
    shape_w = (output_shape[0], output_shape[1], kernel_size[0], kernel_size[1])
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
    
    A_w = as_strided(A, shape_w, strides_w)
    return A_w.max(axis=(2,3)),A_w
class MaxPooling(Operation):
    def __init__(self, size: tuple,stride: int = 2) -> None:
        super().__init__()
        self.height_kernel, self.width_kernel = size
        self.stride = stride
    def _output(self) -> np.ndarray:
        # output = np.zeros((self.input.shape[0],self.input.shape[1],output_height,output_width))
        # for i in range(self.input.shape[0]):
        #     for j in range(self.input.shape[1]):
        #         # for y in range(self.input.shape[2]):
        #         #     if y > self.input.shape[2]-self.height_kernel:
        #         #         break
        #         #     if y % self.stride==0:
        #         #         for x in range(self.input.shape[3]):
        #         #             if x > self.input.shape[3]-self.width_kernel:
        #         #                 break
        #         #             if x%self.stride==0:
        #         #                 output[i,j,int(y/self.stride),int(x/self.stride)] = np.max(self.input[i,j,y:y+self.height_kernel,x:x+self.width_kernel])
        #         output[i,j] = pool2D(self.input[i,j],(self.height_kernel,self.width_kernel),self.stride)
        pool = torch.nn.MaxPool2d(kernel_size=self.height_kernel,stride = self.stride)
        output = pool(torch.from_numpy(self.input).float())
        return output.numpy()
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        output = np.zeros_like(self.input)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for y in range(self.input.shape[2]):
                    if y > self.input.shape[2]-self.height_kernel:
                        break
                    if y % self.stride==0:
                        for x in range(self.input.shape[3]):
                            if x > self.input.shape[3]-self.width_kernel:
                                break
                            if x % self.stride==0:
                                y_max_index, x_max_index = np.where(self.input[i,j,y:y+self.height_kernel,x:x+self.width_kernel]==np.max(self.input[i,j,y:y+self.height_kernel,x:x+self.width_kernel]))
                                y_max_index, x_max_index = y_max_index[0], x_max_index[0]
                                output[i,j,y_max_index,x_max_index] += output_grad[i,j,int(y/self.stride),int(x/self.stride)]
        return output
class Convolutional(ParamOperation):
    def __init__(self, kernel: np.ndarray, stride: int = 1,padding_size: int = 0) -> None:
        super().__init__(kernel)
        self.stride = stride
        self.padding_size = padding_size
    def _output(self) -> np.ndarray:
        self.add_padd, output = convo2D(self.input,self.param,self.stride,self.padding_size,True)
        return output
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        add_intervae = addCenter(output_grad,self.stride)
        add_pad = np.zeros((add_intervae.shape[0],add_intervae.shape[1],add_intervae.shape[2]+2*self.param.shape[2]-2,add_intervae.shape[3]+2*self.param.shape[3]-2))
        add_pad[:,:,self.param.shape[2]-1:1-self.param.shape[2],self.param.shape[3]-1:1-self.param.shape[3]] = add_intervae[:,:]
        param_rotate = rotate180(self.param)
        return convo2D(add_pad,transpose(param_rotate))
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        add_intervae = addCenter(output_grad,self.stride)
        return convo2D(transpose(self.add_padd),add_intervae)
class ConvolutionalAddBias(ParamOperation):
    def __init__(self, b: np.ndarray) -> None:
        super().__init__(b)
    def _output(self) -> np.ndarray:
        return self.input+np.ones((self.input.shape[0],1,self.input.shape[2],self.input.shape[3]))[:,:]*self.param[:,:]
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return np.ones_like(self.input)*output_grad
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        transpose_output_grad = transpose(output_grad)
        output = transpose_output_grad.sum(axis=1)
        output = output.reshape(output.shape[0],1,output.shape[1],output.shape[2])
        return output
