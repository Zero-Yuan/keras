import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
num_classes = 10
epochs = 20

#加载数据
mnist = input_data.read_data_sets("MNIST_data/")
x_train,y_train = mnist.train.images,mnist.train.labels
x_test,y_test = mnist.test.images,mnist.train.labels
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

x_train = x_train.reshape(55000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

#数据的标准化
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#将Y 标签转化为分类
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#对模型进行总结，可以看到模型参数个数，每层的shape等信息
model.summary()

#模型编译，loss是损失函数，optimizer是优化方法
model.compile(loss='categorical_crossentropy',
			  optimizer=RMSprop(),
			  metrics=['accuracy'])

#给模型喂数据
history = model.fit(x_train, y_train,
					batch_size=batch_size,
					epochs=epochs,
					verbose=1,
					validation_data=(x_test,y_test))

#对模型进行评估
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])