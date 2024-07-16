
import tensorflow as tf
from tensorflow.keras.losses import Loss, MSE
import segmentation_models as sm

# l1=0.2
# l2=0.2
# margin1=0
# margin2=0.2

# l1=0
# l2=0
# margin1=0
# margin2=0

class Triplet_loss_function_dendrite(Loss):
    def __init__(self, l = 0, margin = 0):
        super(Triplet_loss_function_dendrite, self).__init__()
        self.l = l
        self.margin = margin

    def call(self, y_true, y_pred):
        original_loss = sm.losses.dice_loss(y_true, y_pred)
        y_p_reshaped = tf.reshape(y_pred, (-1, 3, 128, 128, 1))
        triplet_loss=tf.math.maximum(tf.convert_to_tensor(0,dtype='float32'), tf.reduce_mean(MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,1,:,:])-MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,2,:,:])+self.margin))
        return original_loss+ self.l*triplet_loss


class Triplet_loss_function_spine(Loss):
    def __init__(self, l = 0, margin = 0):
        super(Triplet_loss_function_spine, self).__init__()
        self.l = l
        self.margin = margin

    def call(self, y_true, y_pred):
        original_loss = MSE(y_true, y_pred)
        y_p_reshaped = tf.reshape(y_pred, (-1, 3, 128, 128, 1))
        triplet_loss=tf.math.maximum(tf.convert_to_tensor(0,dtype='float32'), tf.reduce_mean(MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,1,:,:])-MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,2,:,:])+self.margin))
        return tf.reduce_mean(original_loss, axis=-1)+self.l*triplet_loss
