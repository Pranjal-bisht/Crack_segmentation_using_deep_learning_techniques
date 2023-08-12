import os
import numpy as np
import tensorflow as tf
import cv2
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
import glob

# Load functions from other files
from data_preprocessing import img_resize, normalize_array, load_data
from model_definitions import create_segmentation_model
from model_definitions import define_model_unet, define_model_fpn, define_model_pspnet, define_model_deeplabv3plus
from evaluation_metrics import dice_coefficient, iou, precision, recall, f1_score, mean_absolute_error, mean_squared_error
from losses import focal_loss

random.seed(23)

# Define data folder path
data_dir = "./data_cracks/"

# Define image size
image_size = 128

# Define paths
image_path = "./data_cracks/images/*.png"
mask_path = "./data_cracks/masks/*.png"

image_names = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
mask_names = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])

# Define data folder path
data_dir = "./data_cracks/"

# Define adjustable parameters
batch_size = 16
epochs = 40
image_size = 128
test_size = 0.2


# Define optimizer
optimizer = Adam()

# import model , for adjusting the backbone change it in model_train file
model_type = 'unet'
model = create_segmentation_model(model_type) # call the deeplab model using deeplab model function

# import data
X_train, X_test, y_train, y_test = load_data(data_dir,image_size,test_size)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',  # or loss=focal_loss()
    metrics=['accuracy', iou, dice_coefficient, f1_score, mean_absolute_error, mean_squared_error, precision, recall],
)

# Fit model
history = model.fit(
    x=X_train,
    y=y_train[..., None],
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test[..., None]),
)

# Save model
model.save("crack_detection_model.h5")
