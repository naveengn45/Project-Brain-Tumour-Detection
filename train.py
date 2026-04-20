import os
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(42)

from PIL import Image, ImageOps
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
img1 = Image.open('./split_data/train/any/ID_001faa58f.jpg')
print(np.array(img1).shape)
display(img1)
img2=Image.open('./split_data/val/Any/ID_001cc58e9.jpg')
print(np.array(img2).shape)
display(img2)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)

train_data = train_datagen.flow_from_directory(
    directory='./split_data/train',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=64,
    seed=42
)

#first_batch = train_data.next()
#print(first_batch[0].shape), print(first_batch[1].shape)

train_data = train_datagen.flow_from_directory(
    directory='./split_data/train/',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=64,
    seed=42
)

valid_data = valid_datagen.flow_from_directory(
    directory='./split_data/val/',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=64,
    seed=42
)
model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model_1.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
)

history_1 = model_1.fit(
    train_data,
    epochs=3

)

    
num_of_test_samples=len(valid_data.classes)
y_test=valid_data.classes
Y_pred = model_1.predict(valid_data)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
print("confusion_matrix",confusion_matrix(valid_data.classes, y_pred))
print("Recall_score",recall_score(valid_data.classes, y_pred, pos_label='positive',average='micro'))
print("F1_score",f1_score(valid_data.classes, y_pred, pos_label='positive',average='micro'))
print("Precision_score",precision_score(valid_data.classes, y_pred, pos_label='positive',average='micro'))

import h5py
model_1.save('model.h5')
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history_1.history['accuracy'])
#plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(history_1.history['loss'])
#plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

