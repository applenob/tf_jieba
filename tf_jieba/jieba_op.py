''' python custom ops '''

import tensorflow as tf
from absl import logging

path = tf.compat.v1.resource_loader.get_path_to_datafile('x_ops.so')
logging.debug(f'x_ops.so path: {path}')
gen_x_ops = tf.load_op_library(path)

jieba_cut = gen_x_ops.jieba_cut

