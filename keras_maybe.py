import os
from glob import glob

import numpy as np
from keras import backend as keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import (Conv2D, Cropping2D, Dropout, Input, MaxPooling2D,
                          UpSampling2D, concatenate)
from keras.models import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

inputList_gPb = sorted(glob("./cut/*/cv_norm/layer_*.mat"))
inputList_npy = sorted(glob("./cut/*/cv_norm/layer_*.npy"))
outputList_npy = sorted(glob("./cut/*/laendo/layer_*.npy"))

# inputs_gPb = []
# for inputgPb in inputList_gPb:
#     mat_contents = sio.loadmat(inputgPb)
#     gPb_orient = mat_contents["gPb_orient"]
#     max_gPb_orient = np.max(gPb_orient, axis = 2)
#     inputs_gPb.append(max_gPb_orient)
# inputs_gPb = np.array(inputs_gPb)
# np.save("cut_gPb", inputs_gPb)

# inputs_cut = []
# for inputcut in inputList_npy:
#     inputs_cut.append(np.load(inputcut))
# inputs_cut = np.array(inputs_cut)
# np.save("cut", inputs_cut)

# masks = []
# for masknpy in outputList_npy:
#     mask = np.load(masknpy)/255
#     mask = mask.astype(np.int)
#     masks.append(mask)
# masks = np.array(masks)
# np.save("cut_laendo", masks)

inputs_gPb = np.load("cut_gPb.npy")
inputs_cut = np.load("cut.npy")
masks = np.load("cut_laendo.npy")

print(inputs_gPb.shape)
print(masks.shape)
print(inputs_cut.shape)
print(np.unique(masks[0]))

#X_train, X_test, Y_train, Y_test = train_test_split(inputs_gPb, masks, test_size=0.15)
X_train, X_test, Y_train, Y_test = train_test_split(inputs_cut, masks, test_size=0.15)


num_train, height, width = X_train.shape
num_test = X_test.shape[0]

X_train = X_train.reshape(num_train, height, width, 1)
Y_train = Y_train.reshape(num_train, height, width, 1)
X_test = X_test.reshape(num_test, height, width, 1)
Y_test = Y_test.reshape(num_test, height, width, 1)

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

np.save("xtrain", X_train)
np.save("xtest", X_test)
np.save("ytrain", Y_train)
np.save("ytest", Y_test)

inp = Input(shape=(height, width, 1))
conv1 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(inp)
print("conv1 shape:", conv1.shape)
conv1 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv1)
print("conv1 shape:", conv1.shape)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
print("pool1 shape:", pool1.shape)

conv2 = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool1)
print("conv2 shape:", conv2.shape)
conv2 = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv2)
print("conv2 shape:", conv2.shape)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
print("pool2 shape:", pool2.shape)

conv3 = Conv2D(256, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool2)
print("conv3 shape:", conv3.shape)
conv3 = Conv2D(256, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv3)
print("conv3 shape:", conv3.shape)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
print("pool3 shape:", pool3.shape)

conv4 = Conv2D(512, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool3)
print("conv4 shape:", conv4.shape)
conv4 = Conv2D(512, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(conv4)
print("conv4 shape:", conv4.shape)
drop4 = Dropout(0.5)(conv4)
print("drop4 shape:", drop4.shape)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
print("pool4 shape:", pool4.shape)

conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(pool4)
print("conv5 shape:", conv5.shape)
conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(conv5)
print("conv5 shape:", conv5.shape)
drop5 = Dropout(0.5)(conv5)
print("drop5 shape:", drop5.shape)

up6 = Conv2D(512, 2, activation='relu', padding='same',
                kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(512, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(merge6)
conv6 = Conv2D(512, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(conv6)

up7 = Conv2D(256, 2, activation='relu', padding='same',
                kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(256, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(merge7)
conv7 = Conv2D(256, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(conv7)

up8 = Conv2D(128, 2, activation='relu', padding='same',
                kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(128, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(merge8)
conv8 = Conv2D(128, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(conv8)

up9 = Conv2D(64, 2, activation='relu', padding='same',
                kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(64, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(merge9)
conv9 = Conv2D(64, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same',
                kernel_initializer='he_normal')(conv9)
conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

model = Model(input=inp, output=conv10)

model.compile(optimizer=Adam(lr=1e-4),
                loss='binary_crossentropy', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
print('Fitting model...')
model.fit(X_train, Y_train, 
          batch_size=4, epochs=10, verbose=1,
          validation_split=0.25, shuffle=True, callbacks=[model_checkpoint])
