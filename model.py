import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras


def load_model(model_path):
    """
    Loads a saved model from a saved path
    """
    print(f"Loading saved model from {model_path}")
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={"KerasLayer": hub.KerasLayer})
    return model


loaded_full_model = load_model("model/full-image-set.h5")


labels_csv = pd.read_csv("model/labels.csv")
labels = labels_csv["breed"]
labels = np.array(labels)
unique_breeds = np.unique(labels)

# Define image size
IMG_SIZE = 224

# Define the preprocess function


def process_image(image_path, img_size=IMG_SIZE):
    """
    Takes an image file path and turns it into a Tensor
    """
    # Read an image
    image = tf.io.read_file(image_path)
    # Turn the image into Tensor with rgb
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the color channel values from 0-255 to 0-1 values
    # Normalization
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the shape to (224, 224)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


# Define the batch size = 32
BATCH_SIZE = 32

# Function to create data into batches


def create_data_batches(X, y=None, batch_size=BATCH_SIZE):
    """
    Creates batches of data out of images (X) and labels (y) pairs.
    Shuffles if trainning data but not of validation data
    Also accepts test data (they don't have labels)
    """
    # test data
    data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X)))  # Only filepath no labels
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch

# Turn prediction probabilities into their respectiv labels


def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label
    """
    return unique_breeds[np.argmax(prediction_probabilities)]


def get_pred(file_path):
    print(f"Getting file from {file_path}")
    custom_image_paths = [file_path]
    custom_data = create_data_batches(custom_image_paths)
    custom_preds = loaded_full_model.predict(custom_data)
    print([get_pred_label(custom_preds[i])
           for i in range(len(custom_preds))][0])
    return [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))][0]


def clear_uploads(file_path):
    os.remove(file_path)
