# Exp-3: Convolutional Deep Neural Network for Image Classification

## AIM:

To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY: 

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model:

<img width="816" height="590" alt="image" src="https://github.com/user-attachments/assets/f12f2799-3475-4427-995c-b00d5ad2a95b" />


## DESIGN STEPS:

**STEP 1:**
Import tensorflow and preprocessing libraries.

**STEP 2:**
Download and load the dataset

**STEP 3:**
Scale the dataset between it's min and max values

**STEP 4:**
Using one hot encode, encode the categorical values

**STEP 5:**
Split the data into train and test

**STEP 6:**
Build the convolutional neural network model

**STEP 7:**
Train the model with the training data

**STEP 8:**
Plot the performance plot

**STEP 9:**
Evaluate the model with the testing data

**STEP 10:**
Fit the model and predict the single input

## PROGRAM:

**Name:** P VENKATA BALAJI

**Register Number:** 2305001024

```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape

single_image= X_train[0]
single_image.shape

plt.imshow(single_image,cmap='gray')
y_train.shape

X_train.min()
X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()
X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=10,
          batch_size=128,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('7.png')

type(img)

img = image.load_img('7.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

```

## OUTPUT:

### Training Loss, Validation Loss Vs Iteration Plot:

<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/341f83ca-da35-43c7-b773-d36d2793515f" />

<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/07dde5af-6f7f-48ef-8fe5-01a52da0b664" />

### Confusion Matrix:

<img width="659" height="286" alt="image" src="https://github.com/user-attachments/assets/4d72714a-9761-418f-a581-f4ad3baf8c33" />

### Classification Report:

<img width="744" height="436" alt="image" src="https://github.com/user-attachments/assets/33cb9649-9abe-46c1-82dc-0ac7ae16b845" />

### Accuracy:

<img width="951" height="68" alt="image" src="https://github.com/user-attachments/assets/32242c9d-ef81-4b20-adef-672d3536e792" />

### New Sample Data Prediction:

<img width="738" height="576" alt="image" src="https://github.com/user-attachments/assets/1f595693-d557-4792-800c-3782e64db225" />

### Predicted Output:

<img width="635" height="73" alt="image" src="https://github.com/user-attachments/assets/9e62afb2-5000-40df-a717-f48e70802f05" />


## RESULT:

Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
