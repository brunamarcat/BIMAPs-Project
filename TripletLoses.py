
import tensorflow as tf
from tensorflow.keras.losses import Loss, MSE, MeanSquaredError
import segmentation_models as sm


class Triplet_loss_function_dendrite(Loss):
    def __init__(self, l = 0, margin = 0):
        super(Triplet_loss_function_dendrite, self).__init__()
        self.l = l
        self.margin = margin

    def call(self, y_true, y_pred):
        y_p_reshaped = tf.reshape(y_pred, (-1, 3, 128, 128, 1))
        y_t_reshaped = tf.reshape(y_true, (-1, 3, 128, 128, 1))
        original_loss = sm.losses.dice_loss(y_t_reshaped[:,0,:,:,:], y_p_reshaped[:,0,:,:,:])
        mask = tf.not_equal(tf.squeeze(y_t_reshaped[:,0,:,:]),tf.squeeze(y_t_reshaped[:,2,:,:]))
        Aux = sm.losses.dice_loss(y_p_reshaped[:,0,:,:], y_p_reshaped[:,2,:,:])
        Aux =tf.where(mask, Aux, tf.zeros_like(Aux))
        triplet_loss=tf.math.maximum(tf.convert_to_tensor(0,dtype='float32'), 
                    tf.reduce_mean(MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,1,:,:])) -
                    tf.reduce_mean(Aux)
                    +self.margin)
        return original_loss+ self.l*triplet_loss


class Triplet_loss_function_spine(Loss):
    def __init__(self, l = 0, margin = 0):
        super(Triplet_loss_function_spine, self).__init__()
        self.l = l
        self.margin = margin

    def call(self, y_true, y_pred):
        y_t_reshaped = tf.reshape(y_true, (-1, 3, 128, 128, 1))
        y_p_reshaped = tf.reshape(y_pred, (-1, 3, 128, 128, 1))
        original_loss = MSE(y_t_reshaped[:,0,:,:,:], y_p_reshaped[:,0,:,:,:])
        mask = tf.not_equal(tf.squeeze(y_t_reshaped[:,0,:,:]),tf.squeeze(y_t_reshaped[:,2,:,:]))
        Aux = MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,2,:,:])
        Aux =tf.where(mask, Aux, tf.zeros_like(Aux))
        triplet_loss=tf.math.maximum(tf.convert_to_tensor(0,dtype='float32'), 
                    tf.reduce_mean(MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,1,:,:])) -
                    tf.reduce_mean(Aux)
                    +self.margin)
        return tf.reduce_mean(original_loss)+self.l*triplet_loss        
