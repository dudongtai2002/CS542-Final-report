# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:26:06 2016

@author: Dongtai Du

Main function
"""

# -*- coding: utf-8 -*-

#%% Load Data
import os.path
import numpy as np
import theano
from theano import tensor as T

from Font_DN import *
from utility import *
from NeuralNets import *

basis_size = 36
font_dir = 'Fonts'

input_letter = ['B','A','S','Q']
output_letter = ['R']

lamb1 = 0.01        # neural network parameter cost, regularization
lamb2 = 0.01

n_train_batches = 20
n_epochs = 1       #original:1500
batch_size = 1

learning_rate = 1   # learning rate, when using 0.02, less than 200000 epoches will not work.
output_num = 3      # test font output number
total_layer = 4    # writing def in loop is complicated, this parameter is not used


Fonts = Font(basis_size, font_dir, input_letter, output_letter )

[trainInput, trainOutput, testInput, testOutput] = Fonts.getLetterSets(n_train_batches * batch_size, output_num * batch_size)

trainInput = 1 - trainInput
testInput = 1 - testInput

n_train = trainInput.shape[0]
n_test = testInput.shape[0]
input_size = len(input_letter) * basis_size * basis_size
image_size = basis_size * basis_size

trainInput = trainInput.reshape((n_train,image_size*(len(input_letter)+len(output_letter))))
trainOutput = trainOutput.reshape((n_train,1))
testInput = testInput.reshape((n_test,image_size*(len(input_letter)+len(output_letter))))
testOutput = testOutput.reshape((n_test,1))

[trainInput,trainOutput] = shared_dataset(trainInput,trainOutput)
###trainInput:BASQR
###trainOutput:0,1,1,0,.....
#%% building neural networks

rng1 = np.random.RandomState(1234)
rng2 = np.random.RandomState(2345)
rng3 = np.random.RandomState(1567)
rng4 = np.random.RandomState(1124)

nkerns = [2, 2]
f1=3;
f2=4;
max1=(2,2)
max2=(2,2)
f2_size=(36-f1+1)/max1[0]
f3_size=(f2_size-f2+1)/max2[0]


# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')
y = T.imatrix('y')

print('...building the model')

layer00_input = x[:,0 * image_size:1 * image_size].reshape((batch_size, 1, basis_size, basis_size))
layer01_input = x[:,1 * image_size:2 * image_size].reshape((batch_size, 1, basis_size, basis_size))
layer02_input = x[:,2 * image_size:3 * image_size].reshape((batch_size, 1, basis_size, basis_size))
layer03_input = x[:,3 * image_size:4 * image_size].reshape((batch_size, 1, basis_size, basis_size))
layer04_input = x[:,4 * image_size:5 * image_size].reshape((batch_size, 1, basis_size, basis_size))
# first convolutional layer
# image original size 36*36, filter size 3X3, filter number nkerns[0]
# after filtering, image size reduced to (36 - 3 + 1) = 34
# after max pooling, image size reduced to 34 / 2 = 17
layer00 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer00_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, f1,f1),
        poolsize=max1
    )
layer01 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer01_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, f1, f1),
        poolsize=max1
    )
layer02 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer02_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, f1, f1),
        poolsize=max1
    )
layer03 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer03_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, f1, f1),
        poolsize=max1
    )
layer04 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer04_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, f1, f1),
        poolsize=max1
    )

# second convolutional layer
# input image size 17X17, filter size 4X4, filter number nkerns[1]
# after filtering, image size (17 - 4 + 1) = 14
# after max pooling, image size reduced to 14 / 2 = 7
layer10 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer00.output,
        image_shape=(batch_size, nkerns[0], f2_size, f2_size),
        filter_shape=(nkerns[1], nkerns[0], f2, f2),
        poolsize=max2
    )
layer11 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer01.output,
        image_shape=(batch_size, nkerns[0], f2_size, f2_size),
        filter_shape=(nkerns[1], nkerns[0], f2, f2),
        poolsize=max2
    )
layer12 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer02.output,
        image_shape=(batch_size, nkerns[0], f2_size, f2_size),
        filter_shape=(nkerns[1], nkerns[0], f2, f2),
        poolsize=max2
    )
layer13 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer03.output,
        image_shape=(batch_size, nkerns[0], f2_size, f2_size),
        filter_shape=(nkerns[1], nkerns[0], f2, f2),
        poolsize=max2
    )
layer14 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer04.output,
        image_shape=(batch_size, nkerns[0], f2_size, f2_size),
        filter_shape=(nkerns[1], nkerns[0], f2, f2),
        poolsize=max2
    )

# layer 2 input size = 2 * 4 * 7 * 7 =  392
layer2_input = T.concatenate([layer10.output.flatten(2), layer11.output.flatten(2), layer12.output.flatten(2), layer13.output.flatten(2)
                              ,layer14.output.flatten(2)],
                              axis = 1)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer2_input,
        n_in=nkerns[1] * (len(input_letter)+1) * 7 * 7,
        n_out=50,
        activation=T.nnet.sigmoid
    )

layer3 = HiddenLayer(
    np.random.RandomState(np.random.randint(10000)),
    input=layer2.output,
    n_in=50,
    n_out=50,
    activation=T.nnet.sigmoid
)


layer4 = BinaryLogisticRegression(
        np.random.RandomState(np.random.randint(10000)),
        input=layer3.output,
        n_in=50,
        n_out=1
    )

params = (
          layer3.params
          + layer2.params
          + layer4.params
          + layer10.params + layer11.params + layer12.params + layer13.params+layer14.params
          + layer00.params + layer01.params + layer02.params + layer03.params+layer04.params)

#cost function with regularization term
cost = layer4.negative_log_likelihood(y)+ lamb1 * ((params[0])**2).sum() + lamb2 * ((params[1])**2).sum()

error = ((y - layer4.y_pred)**2).sum()

#compute the gradient and apply gradient descent
grads = T.grad(cost, params)
updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates=updates,
        givens={
            x: trainInput[index * batch_size: (index + 1) * batch_size],
            y:trainOutput[index * batch_size: (index + 1) * batch_size]
        }
    )

#%% training the model

epoch = 0
costlist = []


while (epoch < n_epochs):
    epoch = epoch + 1
    total = 0
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        total += minibatch_avg_cost
        iter = (epoch - 1) * n_train_batches + minibatch_index
    if (epoch % 100 == 0):
        print(('   epoch %i') % (epoch))
        print(total)
        costlist += [total]


#%% predict output & print(error)

predict_model = theano.function(
        inputs = [x,y],
        outputs = [layer4.y_pred,cost],
        on_unused_input='ignore',
        allow_input_downcast=True
    )
# create two txt files to store the result
f1=open('output.txt','w+')
f1.close()
f2=open('real.txt','w+')
f2.close()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for testindex in range(output_num):
    predicted_values = predict_model(testInput[testindex * batch_size:(testindex + 1) * batch_size],
                                    testOutput[testindex * batch_size:(testindex + 1) * batch_size])
    output_img = predicted_values[0]
    output_img = output_img.reshape((batch_size))
    with open('output.txt','a') as myfile:
        myfile.write(np.array_str(output_img))
    with open('real.txt','a') as myfile:
        myfile.write(np.array_str(testOutput[testindex * batch_size:(testindex + 1) * batch_size]))
    test_cost = predicted_values[1]
    print(test_cost)
"""

    plt.figure(1)

    le = len(input_letter)
    siz = 10*le + 200

    for x in range(len(input_letter)):

        plt.subplot(siz + x + 1)
        plt.imshow(testInput[testindex,x * image_size:  (x+1)*image_size].reshape((basis_size,basis_size)),interpolation="nearest",cmap='Greys')
    plt.subplot(siz + le + 2)

    plt.imshow(output_img[0,:,:],interpolation="nearest",cmap='Greys')
    plt.subplot(siz + le + 1)
    plt.imshow(testOutput[testindex,:].reshape((basis_size,basis_size)),interpolation="nearest",cmap='Greys')
    x = 0
    st = 'test/c6lasfil-'+ str(learning_rate) + '-' + str(lamb1) + '-' + str(lamb2) + '-'+ str(n_train_batches) + '-' + str(n_epochs) +'-'+ str(batch_size)
    while os.path.exists(st + '-' + str(x) + '.png'):
        x += 1
    plt.savefig(st + '-' + str(x) +'.png')
    plt.show()


# c: Chinese test training
# 6l: 6 layers in total
# as: autosave, f: font.py changes, i: image display change (L -> 1)
# l: lambda
# name style: surffi -d -a -b -c -num
# -d learning rate
# -a n_train_batches
# -b n_epochs      #original: 1500
# -c batch_size    #original: 50
# -num in the end: the name for same parameter images.


fig, ax = plt.subplots( nrows=1, ncols=1 )
ax.plot(costlist)
fig.savefig(st + '-' + str(x) + 'cost_graph.png')
plt.close(fig)

textfile = open('paramrecord', 'a')
textfile.write(st + '-' + str(x) + '\n'
               + "learning rate :" + str(learning_rate) + '\n'
               + 'test number: ' + str(output_num) +'\n'
               + 'lambda: ' + str(lamb1) + '/' + str(lamb2) + '\n'
               + str(total) +'\n \n')
textfile.close()
"""

