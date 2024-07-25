
import os
import numpy as np
os.environ["SM_FRAMEWORK"] = "tf.keras"
import argparse

import segmentation_models as sm
from deepd3.model import DeepD3_Model
from deepd3.training.stream import DataGeneratorStream
from DataGeneratorStreamSiamese import DataGeneratorStreamSiamese
from TripletLoses import Triplet_loss_function_dendrite, Triplet_loss_function_spine
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

#import tensorflow as tf
#tf.config.run_functions_eagerly(True)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Get parameters for the DeepD3 model training.")

    # Add the arguments
    parser.add_argument('--GeneratorData', default='DataGeneratorStreamSiamese', type=str, required=False, help="Kind of data generator to use")
    parser.add_argument('--samples_per_epoch', default=5000, required=False, type=int, help="Number of samples per epoch")
    parser.add_argument('--batchsize', default=32, type=int, required=False, help="Batch Size")
    parser.add_argument('--epochs', default=100, type=int, required=False, help="Number epochs for training")
    parser.add_argument('--loss1', default='TripletLoss', choices=['DiceLoss', 'TripletLoss'], type=str, required=False, help="DiceLoss or TripletLoss")
    parser.add_argument('--l1', default=0, type=float, required=False, help="Lambda for the loss1 function")
    parser.add_argument('--margin1', default=0, type=float, required=False, help="margin for the loss1 function")
    parser.add_argument('--loss2', default='TripletLoss', choices=['MSE', 'TripletLoss'], type=str, required=False, help="MSE or TripletLoss")   
    parser.add_argument('--l2', default=0.01, type=float, required=False, help="Lambda for the loss2 function")
    parser.add_argument('--margin2', default=0.1, type=float, required=False, help="margin for the loss2 function")
    parser.add_argument('--lr', default=0.001, type=float, required=False, help="initial learning rate for the optimizer")
    parser.add_argument('--epochdecay', default=15, type=int, required=False, help="An integer parameter: When to decay the learning rate")
    parser.add_argument('--valuedecay', default=0.025, type=float, required=False, help="Value for exponential decay of the learning rate")
    parser.add_argument('--number_runs', default=1, type=int, required=False, help="An integer parameter: Number of runs of conf")
    parser.add_argument('--data_path', default="./", type=str, required=False,  help="Path to the data")
    parser.add_argument('--data_results', default="./", type=str, required=False,  help="Path to the results")
    parser.add_argument('--augmentation', default=False, type=str, required=False,  help="Augmentation of the data")

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    generator_data = args.GeneratorData
    samples_per_epoch = args.samples_per_epoch
    batchsize = args.batchsize
    EPOCHS = args.epochs
    loss1 = args.loss1
    loss2 = args.loss2
    margin1 = args.margin1
    margin2 = args.margin2
    l1 = args.l1
    l2 = args.l2
    lr = args.lr
    epoch_decay = args.epochdecay
    value_decay = args.valuedecay
    number_runs = args.number_runs
    data_path = args.data_path
    data_results = args.data_results
    augmentation = args.augmentation
    augmentation = (augmentation == "True")

    # Print the arguments
    print(f"GeneratorData: {generator_data}")
    print(f"Samples per epoch: {samples_per_epoch}")
    print(f"Batch Size: {batchsize}")
    print(f"Number of epochs: {EPOCHS}")
    print(f"Loss1: {loss1}")
    print(f"Lambda1: {l1}")
    print(f"Margin1: {margin1}")
    print(f"Loss2: {loss2}")
    print(f"Lambda2: {l2}")
    print(f"Margin2: {margin2}")    
    print(f"Learning rate flag: {lr}")
    print(f"Epoch Decay: {epoch_decay}")
    print(f"Value Decay: {value_decay}")
    print(f"Number of Runs: {number_runs}")
    print(f"Data Path: {data_path}")
    print(f"Data Results: {data_results}")
    print(f"Augmentation: {augmentation}")

    
    ###########  1. Load the training data

    TRAINING_DATA_PATH = data_path+r"DeepD3_Training.d3set"
    VALIDATION_DATA_PATH = data_path+r"DeepD3_Validation.d3set"

    for i in range(number_runs):
        print(f"Run number: {i}")

        if generator_data == "DataGeneratorStream":
            dg_training = DataGeneratorStream(TRAINING_DATA_PATH, #TRAINING_DATA_PATH,
                                            batch_size=batchsize, # Data processed at once, depends on your GPU
                                            target_resolution=0.094, # fixed to 94 nm, can be None for mixed resolution training
                                            augment=augmentation,
                                            samples_per_epoch=samples_per_epoch, # number of samples per epoch
                                            shuffle=False,
                                            min_content=50)
            X_training = []
            Y1_training = []
            Y2_training = []
            voltes = int(samples_per_epoch/batchsize)
            for i in range(voltes):
                print(i,end=' ')
                Xbatch, (Y1batch, Y2batch) = dg_training[i]
                X_training.extend(Xbatch)
                Y1_training.extend(Y1batch)
                Y2_training.extend(Y2batch)
            X_training = np.array(X_training)
            Y1_training = np.array(Y1_training)
            Y2_training = np.array(Y2_training)

            dg_validation = DataGeneratorStream(VALIDATION_DATA_PATH,
                                                batch_size=batchsize,
                                                target_resolution=0.094,
                                                min_content=50,
                                                augment=False,
                                                samples_per_epoch=samples_per_epoch, # number of samples per epoch
                                                shuffle=False)
            X_validation = []
            Y1_validation = []
            Y2_validation = []
            voltes = int(1000/batchsize)
            for i in range(voltes):
                print(i,end=' ')
                Xbatch, (Y1batch, Y2batch) = dg_validation[i]
                X_validation.extend(Xbatch)
                Y1_validation.extend(Y1batch)
                Y2_validation.extend(Y2batch)
            X_validation = np.array(X_validation)
            Y1_validation = np.array(Y1_validation)
            Y2_validation = np.array(Y2_validation)            

        else:
            dg_training = DataGeneratorStreamSiamese(TRAINING_DATA_PATH, #TRAINING_DATA_PATH, 
                                            batch_size=batchsize, # Data processed at once, depends on your GPU
                                            target_resolution=0.094, # fixed to 94 nm, can be None for mixed resolution training
                                            augment=augmentation,
                                            samples_per_epoch=samples_per_epoch, # number of samples per epoch
                                            shuffle=False,
                                            min_content=50) # images need to have at least 50 segmented px
                        
            X_training = []
            Y1_training = []
            Y2_training = []
            voltes = int(samples_per_epoch/batchsize)
            for i in range(voltes):
                print(i,end=' ')
                Xbatch, (Y1batch, Y2batch) = dg_training[i]
                X_training.extend(Xbatch)
                Y1_training.extend(Y1batch)
                Y2_training.extend(Y2batch)
            X_training = np.array(X_training)
            Y1_training = np.array(Y1_training)
            Y2_training = np.array(Y2_training)


            dg_validation = DataGeneratorStreamSiamese(VALIDATION_DATA_PATH,
                                                batch_size=batchsize,
                                                target_resolution=0.094,
                                                min_content=50,
                                                augment=False,
                                                samples_per_epoch=samples_per_epoch, # number of samples per epoch
                                                shuffle=False)

            X_validation = []
            Y1_validation = []
            Y2_validation = []
            voltes = int(samples_per_epoch/batchsize)
            for i in range(voltes):
                print(i,end=' ')
                Xbatch, (Y1batch, Y2batch) = dg_validation[i]
                X_validation.extend(Xbatch)
                Y1_validation.extend(Y1batch)
                Y2_validation.extend(Y2batch)
            X_validation = np.array(X_validation)
            Y1_validation = np.array(Y1_validation)
            Y2_validation = np.array(Y2_validation)   

        ###########  2. Create the DeepD3 model
        # Create a naive DeepD3 model with a given base filter count (e.g. 32)
        m = DeepD3_Model(filters=32)
        # m.summary()

        # Define losses
        if loss1 == "DiceLoss":
            loss_1 = sm.losses.DiceLoss()  #sm.losses.dice_loss
        else:
            loss_1 = Triplet_loss_function_dendrite(l=l1,margin=margin1)

        if loss2 == "MSE":
            loss_2 = MeanSquaredError()
        else:
            loss_2 = Triplet_loss_function_spine(l=l2,margin=margin2)


        # Set appropriate training settings
        m.compile(Adam(learning_rate=lr), # optimizer, good default setting, can be tuned 
                [loss_1, loss_2], # triplet custom losses for dendrite and spine
                loss_weights=[10, 1], # weights for the losses
                metrics=[[sm.metrics.iou_score,loss_1], [sm.metrics.iou_score,loss_2]]) # Metrics for monitoring progress
#                metrics=[[sm.metrics.iou_score], [sm.metrics.iou_score]]) # Metrics for monitoring progress

        #########  3. Fitting  the model

        # Loading some training callbacks, such as adjusting the learning rate across time, saving training progress and intermediate models

        def schedule(epoch, lr):
            if epoch < epoch_decay:
                return lr
            else:
                return lr * np.exp(-value_decay)

        # Save best model automatically during training
        mc = ModelCheckpoint(data_results+"DeepD3_model "+str(i)+".h5.keras", save_best_only=True)
                
        # Save metrics  
        csv = CSVLogger(data_results+"DeepD3_model "+str(i)+".csv")

        # Adjust learning rate during training to allow for better convergence
        lrs = LearningRateScheduler(schedule)

        # Actually train the network
        h = m.fit(x=X_training,y=(Y1_training, Y2_training), 
                epochs=EPOCHS, 
                batch_size=int(3*batchsize),
                validation_data=(X_validation, (Y1_validation, Y2_validation)),
                callbacks=[mc, csv, lrs],
                shuffle = False
                )

        # h = m.fit(dg_training, 
        #         epochs=EPOCHS, 
        #         validation_data=dg_validation, 
        #         callbacks=[mc, csv, lrs],
        #         )


        # ## Save model for use in GUI or batch processing
        # This is for saving the neural network manually. The best model is automatically saved during training.
        m.save(data_results+"deepd3_custom_trained_model "+str(i)+".h5")

        #########  4. Plotting the training progress
        import pandas as pd
        import matplotlib.pyplot as plt

        # Load the training history
        df = pd.read_csv(data_results+"DeepD3_model "+str(i)+".csv")

        # Plot the "val_spines_iou_score" column
        plt.figure(figsize=(10, 6))
        plt.plot(df["val_spines_iou_score"])
        plt.title("Validation Spines IoU Score Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("IoU Score")
        plt.grid(True)
        #plt.legend(["Original original lr", "Original", "Tripled loss (ours)"])
        plt.savefig(data_results+"Spines "+str(i)+".png")
#        plt.show()
#        plt.pause(1)

        plt.figure(figsize=(10, 6))
        plt.plot(df["val_dendrites_iou_score"])
        plt.title("Validation Dendrites IoU Score Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("IoU Score")
        plt.grid(True)
        #plt.legend(["Original original lr", "Original", "Tripled loss (ours)"])
        plt.savefig(data_results+"Dendrites "+str(i)+".png")
#        plt.show()
#        plt.pause(1)

if __name__ == "__main__":
    main()