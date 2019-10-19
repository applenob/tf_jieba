''' python custom ops '''

import tensorflow as tf
from absl import logging
from glob import glob
import os

logging.set_verbosity(logging.INFO)
this_directory = os.path.abspath(os.path.dirname(__file__))
path = glob(os.path.join(this_directory, 'x_ops.*.so'))[0]
logging.info(f'x_ops.so path: {path}')
gen_x_ops = tf.load_op_library(path)

jieba_cut = gen_x_ops.jieba_cut

