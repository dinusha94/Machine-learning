import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


cwd = os.getcwd()
loadpath = cwd + "/npz/train.npz"
l = np.load(loadpath)

# See what's in here
print (l.files)

# Parse data
train_img = l['train']
train_label = l['train_labels']

#shuffle the input data
c = list(zip(train_img , train_label))
random.shuffle(c)
train_img, train_label = zip(*c)
train_label = np.array(train_label)

#im = train_img[0]
#plt.imshow(im.reshape(240, 640), cmap=plt.cm.gray)
#plt.show()


# reconstruct images
image_data_list=[]
frame = 0
for im in train_img:
    input_img = im.reshape(120, 320)
    #cv2.imwrite("training_data/"+str(frame)+".jpg", input_img)
    image_data_list.append(input_img)
    frame += 1

#plt.imshow(image_data_list[0], cmap=plt.cm.gray)
#plt.show()


input_img_data_list=[]
for im in image_data_list:
    input_img_resize = cv2.resize(im,(120,120))
    input_img_data_list.append(input_img_resize)


#plt.imshow(input_img_data_list[0], cmap=plt.cm.gray)
#plt.show()

img_data = np.array(input_img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
EPOCHS = 200
num_classes = 4
num_channel=1

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)

# Defining the model
input_shape=img_data[0].shape	
print (	input_shape)
			
model = Sequential()
model.add(Convolution2D(60, (3,3), border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Convolution2D(120, (3,3), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(240, (3,3), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

# Viewing model_configuration

model.summary()

#Train the model
history = model.fit(img_data, train_label, epochs=EPOCHS,validation_split=0.2, verbose=1)
#save the model
model.save('my_model_1.h5')























