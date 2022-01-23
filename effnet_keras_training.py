import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
import numpy as np
import cv2

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.efficientnet.preprocess_input(img_array_expanded_dims)

# img = cv2.imread('D:/Study/VOC2012/JPEGImages/2007_000033.jpg')
# cv2.imshow("Test",img)
# cv2.waitKey(0)

# preprocessed_image = prepare_image('D:/Study/VOC2012/JPEGImages/2007_000033.jpg')
# predictions = mobile.predict(preprocessed_image)
# results = imagenet_utils.decode_predictions(predictions)
# print(results)


base_model=EfficientNetB0(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(14,activation='softmax')(x) #final layer with softmax activation


model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture

# for i,layer in enumerate(model.layers):
#     print(i, layer.name)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory("D:/Study/VOC2012/Classification/train",
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

val_generator=val_datagen.flow_from_directory("D:/Study/VOC2012/Classification/val",
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

# Learning rate decay scheduler
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    decay_rate=0.9)

# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
opt = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
step_size_val=val_generator.n//val_generator.batch_size
model.fit(train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data=val_generator,
                   validation_steps=step_size_val,
                   epochs=25)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('effnet_model.tflite', 'wb') as f:
  f.write(tflite_model)