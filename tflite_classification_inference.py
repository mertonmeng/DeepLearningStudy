import numpy as np
import cv2
import os
import tensorflow as tf
import tensorflow.keras as keras
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.efficientnet import preprocess_input

test_data_folder = "D:/Study/VOC2012/JPEGImages"

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.efficientnet.preprocess_input(img_array_expanded_dims)

val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
val_generator=val_datagen.flow_from_directory("D:/Study/VOC2012/Classification/val",
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

class_indices = val_generator.class_indices
class_list = list(class_indices.items())
print(class_list)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="effnet_model.tflite")
interpreter.allocate_tensors()

image_files = os.listdir(test_data_folder)

for file_name in image_files:

    full_path = os.path.join(test_data_folder, file_name)
    input_data = prepare_image(full_path)
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']

    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    end = time.time()
    print("Inference Time: {}".format(end - start))
    max_conf = np.max(output_data)
    if max_conf < 0.5:
        print("Image Class Unknown")
    
    class_id = np.argmax(output_data)
    label, _ = class_list[class_id]
    print("Image Class: {}".format(label))
    print("Max Confidence: {}".format(max_conf))
    img = cv2.imread(full_path)

    # Put Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (0, 25)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    img = cv2.putText(img, label, pos, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow(full_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()