import numpy as np
import os

from zipfile import ZipFile
dataset = '/content/brain-tumor-mri-dataset.zip'
with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('done')

base_dir ="/content/Training/"

pituitary = base_dir+"pituitary/"
notumor =base_dir+ "notumor/"
meningioma = base_dir+"meningioma/"
glioma = base_dir+"glioma/"

def find_path(path):
    file_list = os.listdir(path)
    # print(f"length is: {len(file_list)}")
    # print(file_list[0:5])
    return file_list

data_path = [ pituitary,notumor,meningioma,glioma]
for path in data_path:
    find_path(path)

notumor_files = find_path(notumor)
pituitary_files = find_path(pituitary)
meningioma_files = find_path(meningioma)
glioma_files = find_path(glioma)

print(len(notumor_files))
print(len(pituitary_files))
print(len(meningioma_files))
print(len(glioma_files))

print(notumor_files[0:5])
print(pituitary_files[0:5])
print(meningioma_files[0:5])
print(glioma_files[0:5])

print(notumor_files[-5:])
print(pituitary_files[-5:])
print(meningioma_files[-5:])
print(glioma_files[-5:])

notumor_labels = [0]*1595;
pituitary_labels = [1]*1457;
meningioma_labels = [2]*1339;
glioma_labels = [3]*1321;

print(len(notumor_labels))
print(len(pituitary_labels))
print(len(meningioma_labels))
print(len(glioma_labels))

print(notumor_labels[0:5])
print(pituitary_labels[0:5])
print(meningioma_labels[0:5])
print(glioma_labels[0:5])

labels = notumor_labels + pituitary_labels + meningioma_labels + glioma_labels

print(len(labels))

from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread( pituitary +'Tr-pi_0532.jpg')
imgplot= plt.imshow(img)

image_data = []
for img_file in notumor_files:
  image = Image.open(notumor+img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  image_data.append(image)

for img_file in pituitary_files:
  image = Image.open(pituitary+img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  image_data.append(image)

for img_file in meningioma_files:
  image = Image.open(meningioma+img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  image_data.append(image)

for img_file in glioma_files:
  image = Image.open(glioma+img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  image_data.append(image)

len(image_data)

type(image_data)

image_data[0]

type(image_data[0])

X= np.array(image_data)
Y= np.array(labels)

X[0]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

X_train_scaled = X_train/255
X_test_scaled = X_test/255

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

num_of_classes = 4

model = Sequential()

model.add(Conv2D(32,3,3,input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_of_classes,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

history = model.fit(X_train_scaled,Y_train,epochs=10,validation_split=0.1)

loss , accuracy = model.evaluate(X_test_scaled,Y_test)
print("loss: ",loss)
print("accuracy: ",accuracy)

h = history
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

# predictive system
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

def prediction_model(image_path):
  input_image = cv2.imread(image_path)
  input_image_resize = cv2.resize(input_image,(128,128))
  input_image_scaled = input_image_resize/255

  input_image_reshaped = np.reshape(input_image_scaled,[1,128,128,3])

  input_prediction = model.predict(input_image_reshaped)


  input_pred_labels = np.argmax(input_prediction)

  if input_pred_labels == 0:
   print("The MRI scan has no tumor")
  elif input_pred_labels == 1:
   print("The MRI scan has pituitary tumor")
  elif input_pred_labels == 2:
   print("The MRI scan has meningioma tumor")
  elif input_pred_labels == 3:
   print("The MRI scan has glioma tumor")

prediction_model("/content/Testing/meningioma/Te-me_0134.jpg")

# Save the entire model
model.save("newbrain_tumor_model.h5")

