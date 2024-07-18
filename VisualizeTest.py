from deepd3.model import DeepD3_Model
from deepd3.training.stream import DataGeneratorStream
from DataGeneratorStreamSiamese import DataGeneratorStreamSiamese
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from TripletLoses import Triplet_loss_function_dendrite, Triplet_loss_function_spine   
from tensorflow.keras.losses import MeanSquaredError

import segmentation_models as sm

VALIDATION_DATA_PATH = "./DeepD3_Validation.d3set"
dg_validation = DataGeneratorStreamSiamese(VALIDATION_DATA_PATH,
                                                batch_size=32,
                                                target_resolution=0.094,
                                                min_content=50,
                                                augment=False,
                                                shuffle=False)
dg_validation2 = DataGeneratorStream(VALIDATION_DATA_PATH,
                                                batch_size=32,
                                                target_resolution=0.094,
                                                min_content=50,
                                                augment=False,
                                                shuffle=False)

model = DeepD3_Model(filters=32)

#model.load_weights("DeepD3_model 0.h5.keras")
model.load_weights("deepd3_custom_trained_model 0.h5")
#print(model.summary())



#####   VISUALIZE A SINGLE BATCH OF VALIDATION DATA
X, Y = dg_validation.__getitem__(0)
Yt = tf.convert_to_tensor(Y[0])
Yp = model(X)

plt.ion()

for i in range(0,X.shape[0],3):
    print(i)
    plt.figure(figsize=(15,10))
    plt.subplot(331)
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.colorbar()

    plt.subplot(332)
    plt.imshow(Y[0][i].squeeze(), cmap='gray')
    plt.colorbar()

    plt.subplot(333)
    plt.imshow(Y[1][i].squeeze(), cmap='gray')
    plt.colorbar()

    plt.subplot(335)
    plt.imshow(tf.squeeze(Yp[0][i]), cmap='gray')
    plt.colorbar()

    plt.subplot(336)
    plt.imshow(tf.squeeze(Yp[1][i]), cmap='gray')
    plt.colorbar()

    # Visualize the binarized prediction
    # plt.subplot(339)
    # threshold = 0.5
    # binarized_Yp = tf.cast(Yp[1][i] > threshold, tf.float32)
    # plt.imshow(tf.squeeze(binarized_Yp), cmap='gray')
    # plt.colorbar()

    plt.tight_layout()
    plt.pause(0.1)


####### COMPUTE IOU SCORES FOR VALIDATION SET (in nbatches batches so that it fits in memory)
## Done fot triplets generator one comproved that both generators match the same IOU scores

l1 =Triplet_loss_function_dendrite(l=0, margin=0)
l2 = Triplet_loss_function_spine(l=0, margin=0)
loss_2 = MeanSquaredError()
loss_1 = sm.losses.DiceLoss()  #sm.losses.dice_loss

lval1=[]
lval2=[]
ts1=[]
ts2=[]
os1=[]
os2=[]
nbatches = 15
nbatch = int(len(dg_validation)/nbatches)
for batch in range(nbatches):
    X_batches = []
    Y1_batches = []
    Y2_batches = []
    for i in range(nbatch*batch,nbatch*(batch+1)):
        print(i,end=' ')
        Xbatch, (Y1batch, Y2batch) = dg_validation[i]
        X_batches.extend(Xbatch)
        Y1_batches.extend(Y1batch)
        Y2_batches.extend(Y2batch)
    # Convert lists to numpy arrays for vectorized operations
    X_batches = np.array(X_batches)
    Y1_batches = np.array(Y1_batches)
    Y2_batches = np.array(Y2_batches)
    res = model.predict(X_batches)
    lval1.append(sm.metrics.iou_score(Y1_batches, res[0]).numpy())
    lval2.append(sm.metrics.iou_score(Y2_batches, res[1]).numpy())
    ts1.append(l1(Y1_batches, res[0]).numpy())
    ts2.append(l2(Y2_batches, res[1]).numpy())
    os1.append(loss_1(Y1_batches, res[0]).numpy())
    os2.append(loss_2(Y2_batches, res[1]).numpy())

# Calculate and print mean IoU scores
print("Validation dendrite IoU: ", np.mean(lval1))
print("Validation spine IoU: ", np.mean(lval2))
print()
print("Triplet loss dendrite: ", np.mean(ts1))
print("Triplet loss spine: ", np.mean(ts2))
print()
print("Original loss dendrite: ", np.mean(os1))
print("Original loss spine: ", np.mean(os2))
print()


####### COMPUTE IOU SCORES FOR VALIDATION SET (in nbatches batches so that it fits in memory)
# For original generator. Checked that is the outputs very similar IOU scores as the triplet generator

# lval1=[]
# lval2=[]
# os1=[]
# os2=[]
# nbatches = 5
# nbatch = int(len(dg_validation2)/nbatches)
# for batch in range(nbatches):
#     X_batches = []
#     Y1_batches = []
#     Y2_batches = []
#     for i in range(nbatch*batch,nbatch*(batch+1)):
#         print(i,end=' ')
#         Xbatch, (Y1batch, Y2batch) = dg_validation2[i]
#         X_batches.extend(Xbatch)
#         Y1_batches.extend(Y1batch)
#         Y2_batches.extend(Y2batch)
#     # Convert lists to numpy arrays for vectorized operations
#     X_batches = np.array(X_batches)
#     Y1_batches = np.array(Y1_batches)
#     Y2_batches = np.array(Y2_batches)
#     res = model.predict(X_batches)
#     lval1.append(sm.metrics.iou_score(Y1_batches, res[0]).numpy())
#     lval2.append(sm.metrics.iou_score(Y2_batches, res[1]).numpy())
#     os1.append(loss_1(Y1_batches, res[0]).numpy())
#     os2.append(loss_2(Y2_batches, res[1]).numpy())

# # Calculate and print mean IoU scores
# print("Validation dendrite IoU: ", np.mean(lval1))
# print("Validation spine IoU: ", np.mean(lval2))
# print()
# print("Original loss dendrite: ", np.mean(os1))
# print("Original loss spine: ", np.mean(os2))


###############

X, Y = dg_validation.__getitem__(0)
Yt = tf.convert_to_tensor(Y[0])
Yp = model(X)

l1=0.2
l2=0.2
margin1=0.2
margin2=0.2

loss1 = Triplet_loss_function_dendrite(l=l1, margin=margin1)
loss2 = Triplet_loss_function_spine(l=l2, margin=margin2)

print('Dendrite')
print(" Triplet Loss", loss1(Yt,Yp[0])) # whole batch of 96
print(" Dice Loss", sm.losses.dice_loss(Yt,Yp[0]))
print(" IoU score", sm.metrics.iou_score(Yt,Yp[0]))
print('Spine')
print(" Triplet Loss", loss2(Yt,Yp[1])) # whole batch of 96
print(" MSE Loss", MeanSquaredError(Yt,Yp[1]))
print(" IoU score", sm.metrics.iou_score(Yt,Yp[1]))
print()


##
images = X_batches
for i in range(images.shape[0]):
    first_image = images[i]
    matches = tf.equal(images, first_image)
    full_matches = tf.reduce_all(matches, axis=[1, 2, 3])
    count = tf.reduce_sum(tf.cast(full_matches, tf.int32))
    if count > 2:
        result = count.numpy()
        print(f"The {i} image appears {result} times in the batch.")