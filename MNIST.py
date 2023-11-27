from TrainerGD import *
from main_h import *
from keras.datasets import mnist
import time
(train_X, train_y), (test_X, test_y) = mnist.load_data()
number_class = np.max(train_y)+1
one_hot_train, one_hot_test = np.zeros((train_y.shape[0],number_class)), np.zeros((test_y.shape[0],number_class))
for i in range(test_y.shape[0]):
    one_hot_test[i,test_y[i]] = 1
for j in range(train_y.shape[0]):
    one_hot_train[j,train_y[j]] = 1
train_X = train_X.reshape(train_X.shape[0],1,train_X.shape[1],train_X.shape[2])
test_X = test_X.reshape(test_X.shape[0],1,test_X.shape[1],test_X.shape[2])
start = time.time()
net = CNN(
    layers=[
        ConvolutionLayer(32,(3,3),Relu()),
        # MaxPooling((2,2)),
        Flatten(),
        Dense(100,Relu()),
        Dense(10,Relu())
    ],
    loss=SoftMax()
)
optim = RMSProp(0.01,net)
trainer = SGD_mini_batch(net,optim)
trainer.fit(train_X[0:30000],one_hot_train[0:30000],train_X[50000:55000],one_hot_train[50000:55000],5000,25,1)
predict = trainer.net.forward(test_X)
predict = trainer.net.loss.forward(predict,one_hot_test)
print(predict)
predict =trainer.net.loss.softmax_predict
predict = np.argmax(predict,axis=1)
print(accuracy_score(predict,test_y)*100)
end = time.time()
print(end-start)
m = np.random.randint(0,10001,1)
plt.imshow(test_X[m[0]][0],cmap='gray')
plt.show()
k = test_X[m[0]].reshape(1,1,test_X.shape[2],test_X.shape[3])
pre = net.forward(k)
pre = net.loss.forward(pre,one_hot_test[m[0]].reshape(1,10))
print('Loss' + str(pre))
pre = net.loss.softmax_predict
print(np.argmax(pre,axis=1))
