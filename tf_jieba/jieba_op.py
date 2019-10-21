''' python custom ops '''

import tensorflow as tf
from absl import logging
from glob import glob
import os

logging.set_verbosity(logging.INFO)
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
  path = glob(os.path.join(this_directory, 'x_ops.*so'))[0]
except IndexError:
  raise FileNotFoundError('No x_ops.*so found in {}'.format(this_directory))
logging.info(f'x_ops.so path: {path}')
gen_x_ops = tf.load_op_library(path)


def read_lines_from_text_file(file_path):
  """Read lines from a text file."""
  with open(file_path) as f:
    lines = [line.strip() for line in f.readlines()]
    return lines


def jieba_cut(input_sentence,
              use_file=True,
              hmm=True):

  dict_path = os.path.join(this_directory,
                           "./cppjieba_dict/jieba.dict.utf8")
  hmm_path = os.path.join(this_directory,
                          "./cppjieba_dict/hmm_model.utf8")
  user_dict_path = os.path.join(this_directory,
                                "./cppjieba_dict/user.dict.utf8")
  idf_path = os.path.join(this_directory,
                          "./cppjieba_dict/idf.utf8")
  stop_word_path = os.path.join(this_directory,
                                "./cppjieba_dict/stop_words.utf8")

  if use_file:
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
    dict_lines = read_lines_from_text_file(dict_path)
    model_lines = read_lines_from_text_file(hmm_path)
    user_dict_lines = read_lines_from_text_file(user_dict_path)
    idf_lines = read_lines_from_text_file(idf_path)
    stop_word_lines = read_lines_from_text_file(stop_word_path)

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

