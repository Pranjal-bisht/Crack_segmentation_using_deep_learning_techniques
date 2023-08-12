from tensorflow.keras.optimizers import Adam
from model_definitions import create_segmentation_model
from evaluation_metrics import iou, dice_coefficient, f1_score, mean_absolute_error, mean_squared_error,precision, recall
from data_preprocessing import load_data
from losses import focal_loss

# Define data folder path
data_dir = "./data_cracks/"

# Define adjustable parameters
batch_size = 16
epochs = 40
image_size = 128
test_size = 0.2


# Define optimizer
optimizer = Adam()


# import model 
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
