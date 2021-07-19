import tensorflow as tf
#from keras.backend.tensorflow_backend import tf

def layerWeightsNorm (layer):
    def weightsNorm (y_true, y_pred):
        return tf.reduce_sum(tf.square(layer))
    return weightsNorm