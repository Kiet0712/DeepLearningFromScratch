from TrainerGD import *
from main_h import *
from keras.datasets import cifar10
import time
# (train_X,train_y), (test_X, test_y) = cifar10.load_data()
# number_class = np.max(train_y)+1
# one_hot_train, one_hot_test = np.zeros((train_y.shape[0],number_class)), np.zeros((test_y.shape[0],number_class))
# for i in range(test_y.shape[0]):
#     one_hot_test[i,test_y[i]] = 1
# for j in range(train_y.shape[0]):
#     one_hot_train[j,train_y[j]] = 1
# train_X = train_X.reshape(train_X.shape[0],3,train_X.shape[1],train_X.shape[2])
# test_X = test_X.reshape(test_X.shape[0],3,test_X.shape[1],test_X.shape[2])

# start = time.time()
# net = CNN(
#     layers=[
#         ConvolutionLayer(32,(3,3),Relu()),
#         MaxPoolingLayer((3,3),2),
#         ConvolutionLayer(28,(3,3),Relu()),
#         MaxPoolingLayer((2,2),2),
#         Flatten(),
#         Dense(100,Relu()),
#         Dense(10,Relu())
#     ],
#     loss=SoftMax()
# )
# optim = RMSProp(0.01,net)
# trainer = SGD_mini_batch(net,optim)
# trainer.fit(train_X[0:45000],train_y[0:45000],train_X[45000:50000],train_y[45000:50000],2500,25,1)
# predict = trainer.net.forward(test_X)
# predict = trainer.net.loss.forward(predict,one_hot_test)
# print(predict)
# predict =trainer.net.loss.softmax_predict
# predict = np.argmax(predict,axis=1)
# print(accuracy_score(predict,test_y)*100)
# end = time.time()
# print(end-start)
a = np.random.randint(1,4,(1,3,2,2))
b = np.random.randint(1,5,(3,1,2,2))
print(a)
print(b)
print(convo2D(a,b))