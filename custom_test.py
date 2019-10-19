import tensorflow as tf
from tf_jieba.jieba_op import jieba_cut

dict_path = "./cppjieba/dict/jieba.dict.utf8"
hmm_path = "./cppjieba/dict/hmm_model.utf8"
user_dict_path = "./cppjieba/dict/user.dict.utf8"
idf_path = "./cppjieba/dict/idf.utf8"
stop_word_path = "./cppjieba/dict/stop_words.utf8"

sentence_t = tf.placeholder(dtype=tf.string)
jieba_cut_t = jieba_cut(
    sentence_t,
    use_file=True,
    hmm=True,
    dict_path=dict_path,
    hmm_path=hmm_path,
    user_dict_path=user_dict_path,
    idf_path=idf_path,
    stop_word_path=stop_word_path)

sess = tf.Session()
while True:
    sentence_in = input("Input: ").strip()
    if sentence_in == "q":
        break
    res = sess.run(jieba_cut_t, feed_dict={sentence_t: sentence_in})
    print(f"Raw output: {res}")
    print(f"Output: {res.decode('utf-8')}")
sess.close()
