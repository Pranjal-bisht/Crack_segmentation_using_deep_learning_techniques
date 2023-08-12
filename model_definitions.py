import segmentation_models as sm
import tensorflow as tf
from evaluation_metrics import iou, dice_coefficient, f1_score, mean_absolute_error, mean_squared_error,precision, recall

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf


# Set the framework for segmentation_models
sm.set_framework('tf.keras')
sm.framework()

# Common configuration
BACKBONE = 'mobilenet'
loss = 'binary_crossentropy'
num_classes = 1
activation = 'sigmoid'

# Get the preprocessing function for the specified backbone
preprocess_input = sm.get_preprocessing(BACKBONE)

# Define the backbone-based model
def create_segmentation_model(model_type):
    if model_type == 'unet':
        model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=num_classes, activation=activation)
    elif model_type == 'fpn':
        model = sm.FPN(BACKBONE, encoder_weights='imagenet', classes=num_classes, activation=activation)
    elif model_type == 'pspnet':
        model = sm.PSPNet(BACKBONE, encoder_weights='imagenet', classes=num_classes, activation=activation)
    else:
        raise ValueError("Invalid model type. Supported types: 'unet', 'fpn', 'pspnet'")
    
    # Compile the model
    model.compile(
        optimizer='Adam',
        loss=loss,
        metrics=['accuracy', iou, dice_coefficient, f1_score, mean_absolute_error, mean_squared_error, precision, recall],
    )
    return model


# deeplab model
def SqueezeAndExcite(inputs, ratio=8):
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = init * se
    return x

def ASPP(inputs):
    """ Image Pooling """
    shape = inputs.shape
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y1 = Conv2D(256, 1, padding="same", use_bias=False)(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

    """ 1x1 conv """
    y2 = Conv2D(256, 1, padding="same", use_bias=False)(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    """ 3x3 conv rate=6 """
    y3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    """ 3x3 conv rate=12 """
    y4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    """ 3x3 conv rate=18 """
    y5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(inputs)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(256, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def deeplabv3_plus(shape):
    """ Input """
    inputs = Input(shape)

    """ Encoder """
    encoder = keras.applications.Resnet50(weights="imagenet", include_top=False, input_tensor=inputs)

    image_features = encoder.get_layer("block13_sepconv2_bn").output
    x_a = ASPP(image_features)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    x_b = encoder.get_layer("block4_sepconv2_bn").output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)

    x = Concatenate()([x_a, x_b])
    x = SqueezeAndExcite(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SqueezeAndExcite(x)

    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(1, 1)(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs, x)
    return model


