
import os
import numpy as np
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm


###########  1. Load the training data

from deepd3.model import DeepD3_Model
from deepd3.training.stream import DataGeneratorStream
from DataGeneratorStreamSiamese import DataGeneratorStreamSiamese

TRAINING_DATA_PATH = r"DeepD3_Training.d3set"
VALIDATION_DATA_PATH = r"DeepD3_Validation.d3set"

# dg_training = DataGeneratorStreamSiamese(TRAINING_DATA_PATH, #TRAINING_DATA_PATH, 
#                                   batch_size=32, # Data processed at once, depends on your GPU
#                                   target_resolution=0.094, # fixed to 94 nm, can be None for mixed resolution training
#                                   augment=False,
#                                   samples_per_epoch=5000, # number of samples per epoch
#                                   shuffle=False,
#                                   min_content=50) # images need to have at least 50 segmented px

dg_training = DataGeneratorStream(TRAINING_DATA_PATH, #TRAINING_DATA_PATH,
                                  batch_size=32, # Data processed at once, depends on your GPU
                                  target_resolution=0.094, # fixed to 94 nm, can be None for mixed resolution training
                                  augment=False,
                                  samples_per_epoch=5000, # number of samples per epoch
                                  shuffle=False,
                                  min_content=50)

dg_validation = DataGeneratorStream(VALIDATION_DATA_PATH,
                                    batch_size=32,
                                    target_resolution=0.094,
                                    min_content=50,
                                    augment=False,
                                    shuffle=False)



###########  2. Create the DeepD3 model
# Create a naive DeepD3 model with a given base filter count (e.g. 32)
m = DeepD3_Model(filters=32)
# m.summary()

# Define losses
from TripletLoses import Triplet_loss_function_dendrite, Triplet_loss_function_spine
from tensorflow.keras.losses import MeanSquaredError

#loss_1 = Triplet_loss_function_dendrite(l=0,margin=0)
#loss_2 = Triplet_loss_function_spine(l=0,margin=0)

loss_1 = sm.losses.DiceLoss() #sm.losses.dice_loss
loss_2 = MeanSquaredError()

from tensorflow.keras.optimizers import Adam

# Set appropriate training settings
m.compile(Adam(learning_rate=0.0001), # optimizer, good default setting, can be tuned 
          [loss_1, loss_2], # triplet custom losses for dendrite and spine
          metrics=[sm.metrics.iou_score, sm.metrics.iou_score]) # Metrics for monitoring progress


#########  3. Fitting  the model

# Loading some training callbacks, such as adjusting the learning rate across time, saving training progress and intermediate models
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.math.exp(-0.025)

# Number of epochs to train the network
EPOCHS = 100

# Save best model automatically during training
mc = ModelCheckpoint("DeepD3_model.h5.keras", save_best_only=True)
        
# Save metrics  
csv = CSVLogger("DeepD3_model.csv")

# Adjust learning rate during training to allow for better convergence
lrs = LearningRateScheduler(schedule)

# Actually train the network
h = m.fit(dg_training, 
        epochs=EPOCHS, 
        validation_data=dg_validation, 
        callbacks=[mc, csv, lrs]
        )


# ## Save model for use in GUI or batch processing
# This is for saving the neural network manually. The best model is automatically saved during training.
m.save("deepd3_custom_trained_model.h5")

#########  4. Plotting the training progress
import pandas as pd
import matplotlib.pyplot as plt

# Load the training history
df = pd.read_csv("DeepD3_model.csv")

# Plot the "val_spines_iou_score" column
plt.figure(figsize=(10, 6))
plt.plot(df["val_spines_iou_score"])
plt.title("Validation Spines IoU Score Over Epochs (5.000 examples)")
plt.xlabel("Epoch")
plt.ylabel("IoU Score")
plt.grid(True)
#plt.legend(["Original original lr", "Original", "Tripled loss (ours)"])
plt.savefig("Spines.png")
plt.show()
plt.pause(1)
