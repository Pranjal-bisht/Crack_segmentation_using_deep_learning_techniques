import segmentation_models as sm
import tensorflow as tf
from evaluation_metrics import iou, dice_coefficient, f1_score, mean_absolute_error, mean_squared_error,precision, recall

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
