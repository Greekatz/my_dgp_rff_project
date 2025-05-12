import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def log_norm_pdf(x, m=0.0, log_v=0.0):
    return -0.5 * tf.log(2 * np.pi) - 0.5 * log_v - 0.5 * tf.square(x - m) / tf.exp(log_v)


def DKL_gaussian(mq, log_vq, mp, log_vp):
    log_vp = tf.reshape(log_vp, (-1, 1))
    return 0.5 * tf.reduce_sum(
        log_vp - log_vq +
        tf.square(mq - mp) / tf.exp(log_vp) +
        tf.exp(log_vq - log_vp) - 1
    )


def get_normal_samples(ns, din, dout):
    dx = np.amax(din)
    dy = np.amax(dout)
    return tf.random.normal(shape=[ns, dx, dy], dtype=tf.float32)


def logsumexp(vals, dim=None):
    m = tf.reduce_max(vals, axis=dim)
    if dim is None:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - m)))
    else:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - tf.expand_dims(m, dim)), axis=dim))


def get_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 50, 'Batch size.')
    flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
    flags.DEFINE_integer('n_iterations', 2000, 'Number of training iterations')
    flags.DEFINE_integer('display_step', 100, 'Print every X iterations')
    flags.DEFINE_integer('mc_train', 30, 'MC samples for gradients')
    flags.DEFINE_integer('mc_test', 30, 'MC samples for prediction')
    flags.DEFINE_integer('n_rff', 10, 'Random Fourier Features per layer')
    flags.DEFINE_integer('df', 1, 'Number of GPs per layer')
    flags.DEFINE_integer('nl', 1, 'Number of DGP layers')
    flags.DEFINE_string('optimizer', "adagrad", 'Optimizer to use')
    flags.DEFINE_string('kernel_type', "RBF", 'Kernel: RBF or arccosine')
    flags.DEFINE_integer('kernel_arccosine_degree', 1, 'Degree for arccosine kernel')
    flags.DEFINE_boolean('is_ard', False, 'Use ARD kernel')
    flags.DEFINE_boolean('local_reparam', False, 'Use local reparameterization')
    flags.DEFINE_boolean('feed_forward', False, 'Feed inputs into all layers')
    flags.DEFINE_integer('q_Omega_fixed', 0, 'Iterations to freeze Omega')
    flags.DEFINE_integer('theta_fixed', 0, 'Iterations to freeze theta')
    flags.DEFINE_string('learn_Omega', 'prior_fixed', 'Omega learning mode')
    flags.DEFINE_integer('duration', 10000000, 'Training time limit')

    # Optional for cluster training (can ignore for Kaggle)
    flags.DEFINE_string("dataset", "", "Dataset name")
    flags.DEFINE_string("fold", "1", "Dataset fold")
    flags.DEFINE_integer("seed", 0, "Seed value")
    flags.DEFINE_boolean("less_prints", False, "Quiet mode")
    flags.DEFINE_string("ps_hosts", "", "")
    flags.DEFINE_string("worker_hosts", "", "")
    flags.DEFINE_string("job_name", "", "")
    flags.DEFINE_integer("task_index", 0, "")
    return FLAGS


def get_optimizer(opt_name, learning_rate):
    optimizers = {
        "adagrad": tf.train.AdagradOptimizer(learning_rate),
        "sgd": tf.train.GradientDescentOptimizer(learning_rate),
        "adam": tf.train.AdamOptimizer(learning_rate),
        "adadelta": tf.train.AdadeltaOptimizer(learning_rate)
    }
    return optimizers.get(opt_name)
