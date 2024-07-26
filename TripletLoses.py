
import tensorflow as tf
from tensorflow.keras.losses import Loss, MSE, MeanSquaredError
import segmentation_models as sm


class Triplet_loss_function_dendrite(Loss):
    def __init__(self, l = 0, margin = 0):
        super(Triplet_loss_function_dendrite, self).__init__()
        self.l = l
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        Computes the Triplet Loss for dendrites
        
        :param y_true: Ground truth labels for dendrites
        :param y_pred:  Predicted segmentation of dendrites
    
        """
        # Reshape ground truth and prediction in triplet format
        y_p_reshaped = tf.reshape(y_pred, (-1, 3, 128, 128, 1)) # batch, triplet (anchor, positive, negative), x, y, channel
        y_t_reshaped = tf.reshape(y_true, (-1, 3, 128, 128, 1))
        original_loss = sm.losses.dice_loss(y_t_reshaped[:,0,:,:,:], y_p_reshaped[:,0,:,:,:]) # Compute Dice Loss for the anchor
        mask = tf.not_equal(tf.squeeze(y_t_reshaped[:,0,:,:]),tf.squeeze(y_t_reshaped[:,2,:,:])) # Check which pixels in the labels are different in the anchor and the negative
        Aux = sm.losses.dice_loss(y_p_reshaped[:,0,:,:], y_p_reshaped[:,2,:,:]) # Calculate the Dice Loss between the prediction for anchor and negative
        Aux =tf.where(mask, Aux, tf.zeros_like(Aux)) # if pixels are different, write in that position the Dice Loss, if they are the same, write 0.
        triplet_loss=tf.math.maximum(tf.convert_to_tensor(0,dtype='float32'), 
                    tf.reduce_mean(MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,1,:,:])) - # Mean MSE between anchor and positive
                    tf.reduce_mean(Aux) # Mean of dice loss for pixels labeled different in anchor and negative
                    +self.margin)
        return original_loss+ self.l*triplet_loss


class Triplet_loss_function_spine(Loss):
    def __init__(self, l = 0, margin = 0):
        super(Triplet_loss_function_spine, self).__init__()
        self.l = l
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        Computes the Triplet Loss for spines
        
        :param y_true: Ground truth labels for spines
        :param y_pred:  Predicted segmentation of spines
    
        """
        # Reshape output in triplet format for ground truth and prediction
        y_t_reshaped = tf.reshape(y_true, (-1, 3, 128, 128, 1)) #batch, triplet (anchor, positive, negative), x, y, channel
        y_p_reshaped = tf.reshape(y_pred, (-1, 3, 128, 128, 1))
        original_loss = MSE(y_t_reshaped[:,0,:,:,:], y_p_reshaped[:,0,:,:,:]) # Compute Mean Square Error for the anchor
        mask = tf.not_equal(tf.squeeze(y_t_reshaped[:,0,:,:]),tf.squeeze(y_t_reshaped[:,2,:,:])) # Check which pixels in the labels are different in the anchor and the negative
        Aux = MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,2,:,:]) # Calculate the Mean Square Error between the prediction for anchor and negative
        Aux =tf.where(mask, Aux, tf.zeros_like(Aux)) # if pixels are different, write in that position the MSE, if they are the same, write 0. 
        triplet_loss=tf.math.maximum(tf.convert_to_tensor(0,dtype='float32'), 
                    tf.reduce_mean(MSE(y_p_reshaped[:,0,:,:], y_p_reshaped[:,1,:,:])) - # Mean MSE between anchor and positive
                    tf.reduce_mean(Aux) # Mean of MSE for pixels labeled different in anchor and negative
                    +self.margin)
        return tf.reduce_mean(original_loss)+self.l*triplet_loss        
