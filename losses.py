import tensorflow as tf
import keras.backend as K

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        pt = tf.where(tf.equal(y_true, 1), pt_1, pt_0 + K.epsilon())
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    
    return focal_loss_fixed
