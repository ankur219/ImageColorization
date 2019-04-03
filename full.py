"""Extraction of keyframes using FFMPEG"""

import os

source_video = input("Enter the path of the source video")
import numpy as np
import cv2

cap = cv2.VideoCapture(source_video)

X = []
while(cap.isOpened()):
    ret, frame = cap.read()
    #if ret == True:
        #X.append(mse(np.array(cv2.resize(frame,(256, 256)), dtype = float)))
    if ret == False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
    #frame = cv2.resize(frame, (256, 256))
    
    X.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

frame_shape = (256, 256)

cap.release()
cv2.destroyAllWindows()

sample_image = np.ones((frame_shape[0], frame_shape[1], 3))

sample_hist = cv2.calcHist([np.float32(sample_image)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

histograms = []

sample_hist = cv2.normalize(sample_hist, None).flatten()

for image in X:
    
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None).flatten()
    histograms.append(hist)

results = []

for hist in histograms:
    results.append(cv2.compareHist(hist, sample_hist, cv2.HISTCMP_BHATTACHARYYA)*10000)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

X1 = np.array(list(zip(results,np.zeros(len(results)))), dtype=np.int)
bandwidth = estimate_bandwidth(X1, quantile=0.07)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X1)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

for k in range(n_clusters_):
    my_members = labels == k
    #print("cluster {0}: {1}".format(k, X1[my_members, 0]))

label_indices = [[] for i in range(len(labels_unique))]
for i in range(len(labels)):
    label_indices[labels[i]].append(i)

imp_indices = []
for i in label_indices:
    if len(i) >= 30:
        for x in i[::30]:
            imp_indices.append(x)
    else:
        imp_indices.append(i[0])

Xtrain = []
for i in imp_indices:
        Xtrain.append(X[i])

"""Training a Convolutional Neural Network model on keyframes"""
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import random
import tensorflow as tf

# Get images
# Change to '/data/images/Train/' to use all the 10k images
#X = []
#for filename in os.listdir(output_directory):
    #X.append(img_to_array(load_img(output_directory + '/' + filename, target_size = (256, 256))))
Xtrain = np.array(Xtrain, dtype=float)

# Set up train and test data
Xtrain = 1.0/255*Xtrain

model = Sequential()
model.add(InputLayer(input_shape=(frame_shape[0], frame_shape[1], 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='adam', loss='mse')


# Image transformer
datagen = ImageDataGenerator()

# Generate training data
batch_size = 4
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Train model      

model.fit_generator(image_a_b_gen(batch_size), epochs=50, steps_per_epoch=len(Xtrain)//batch_size)

model.save('colorize.h5')


"""Creating a list of grayscale frames of the video"""
color_me = list(X)

#Deleting variables that will not be used again
del(X)
del(results)
del(Xtrain)
del(histograms)

"""Colorizing grayscale frames of the video using the model that was saved earlier"""

from keras.models import load_model
model = load_model('colorize.h5')

"""color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i, j in range(len(output)):
    cur = np.zeros((frame_shape[0], frame_shape[1], 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("./Output/"+j, lab2rgb(cur))"""


for count in range(0, len(color_me), 500):
    try:
        X = np.array(color_me[count:count + 500])
        X = rgb2lab(1.0/255*X)[:,:,:,0]
        X = X.reshape(X.shape+(1,))
        output = model.predict(X)
        output = output * 128

        for i, j in zip(range(count, count+500), range(0, 500)):
            cur = np.zeros((frame_shape[0], frame_shape[1], 3))
            cur[:,:,0] = X[j][:,:,0]
            cur[:,:,1:] = output[j]
            imsave("./Output/"+str(i)+'.jpg', lab2rgb(cur))


    except:
        X = np.array(color_me[count:])
        X = rgb2lab(1.0/255*X)[:,:,:,0]
        X = X.reshape(X.shape+(1,))
        output = model.predict(X)
        output = output * 128

        for i, j in zip(range(count, len(color_me)), range(0, X.shape[0])):
            cur = np.zeros((frame_shape[0], frame_shape[1], 3))
            cur[:,:,0] = X[j][:,:,0]
            cur[:,:,1:] = output[j]
            imsave("./Output/"+str(i)+'.jpg', lab2rgb(cur))




