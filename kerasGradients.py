# Taken from https://stackoverflow.com/questions/61887944/how-to-get-gradients-in-tf-2-2-eager/#answer-61952452


import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.mixed_precision.experimental import (
    loss_scale_optimizer as lso)


def _get_grads_graph(model, x, y, params, sample_weight=None, learning_phase=0):
    sample_weight = sample_weight or np.ones(len(x))

    outputs = model.optimizer.get_gradients(model.total_loss, params)
    inputs  = (model.inputs + model._feed_targets + model._feed_sample_weights
               + [K.learning_phase()])

    grads_fn = K.function(inputs, outputs)
    gradients = grads_fn([x, y, sample_weight, learning_phase])
    return gradients

def _get_grads_eager(model, x, y, params, sample_weight=None, learning_phase=0):
    def _process_input_data(x, y, sample_weight, model):
        iterator = data_adapter.single_batch_iterator(model.distribute_strategy,
                                                      x, y, sample_weight,
                                                      class_weight=None)
        data = next(iterator)
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        return x, y, sample_weight

    def _clip_scale_grads(strategy, tape, optimizer, loss, params):
        with tape:
            if isinstance(optimizer, lso.LossScaleOptimizer):
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, params)

        aggregate_grads_outside_optimizer = (
            optimizer._HAS_AGGREGATE_GRAD and not isinstance(
                strategy.extended,
                parameter_server_strategy.ParameterServerStrategyExtended))

        if aggregate_grads_outside_optimizer:
            gradients = optimizer._aggregate_gradients(zip(gradients, params))
        if isinstance(optimizer, lso.LossScaleOptimizer):
            gradients = optimizer.get_unscaled_gradients(gradients)

        gradients = optimizer._clip_gradients(gradients)
        return gradients

    x, y, sample_weight = _process_input_data(x, y, sample_weight, model)

    with tf.GradientTape() as tape:
        y_pred = model(x, training=bool(learning_phase))
        loss = model.compiled_loss(y, y_pred, sample_weight,
                                   regularization_losses=model.losses)

    gradients = _clip_scale_grads(model.distribute_strategy, tape,
                                  model.optimizer, loss, params)
    gradients = K.batch_get_value(gradients)
    return gradients

def get_gradients(model, x, y, params, sample_weight=None, learning_phase=0):
    if tf.executing_eagerly():
        return _get_grads_eager(model, x, y, params, sample_weight,
                                learning_phase)
    else:
        return _get_grads_graph(model, x, y, params, sample_weight,
                                learning_phase)