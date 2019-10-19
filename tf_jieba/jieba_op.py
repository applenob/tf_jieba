''' python custom ops '''

import tensorflow as tf
from absl import logging
from glob import glob
import os

logging.set_verbosity(logging.INFO)
this_directory = os.path.abspath(os.path.dirname(__file__))
path = glob(os.path.join(this_directory, 'x_ops.*so'))[0]
logging.info(f'x_ops.so path: {path}')
gen_x_ops = tf.load_op_library(path)


def jieba_cut(input_sentence,
              use_file=True,
              dict_path="",
              hmm_path="",
              user_dict_path="",
              idf_path="",
              stop_word_path="",
              dict_lines=[""],
              model_lines=[""],
              user_dict_lines=[""],
              idf_lines=[""],
              stop_word_lines=[""],
              hmm=True):
  if use_file:
    if not os.path.exists(dict_path):
      raise FileNotFoundError(f"Dict file not found: {dict_path}!")

    if not os.path.exists(hmm_path):
      raise FileNotFoundError(f"HMM Model file not found: {hmm_path}!")

    if not os.path.exists(user_dict_path):
      raise FileNotFoundError(f"User dict file not found: {user_dict_path}!")

    if not os.path.exists(idf_path):
      raise FileNotFoundError(f"IDF file not found: {idf_path}!")

    if not os.path.exists(stop_word_path):
      raise FileNotFoundError(f"Stop words file not found: {stop_word_path}!")

    output_sentence = gen_x_ops.jieba_cut(
      input_sentence,
      use_file=use_file,
      hmm=hmm,
      dict_path=dict_path,
      hmm_path=hmm_path,
      user_dict_path=user_dict_path,
      idf_path=idf_path,
      stop_word_path=stop_word_path)
  else:
    output_sentence = gen_x_ops.jieba_cut(
      input_sentence,
      use_file=use_file,
      hmm=hmm,
      dict_lines=dict_lines,
      model_lines=model_lines,
      user_dict_lines=user_dict_lines,
      idf_lines=idf_lines,
      stop_word_lines=stop_word_lines)

  return output_sentence

