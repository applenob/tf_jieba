import tensorflow as tf
from tf_jieba.jieba_op import jieba_cut

sentence_t = tf.placeholder(dtype=tf.string)
jieba_cut_t = jieba_cut(sentence_t)

sess = tf.Session()
while True:
    sentence_in = input("Input: ").strip()
    if sentence_in == "q":
        break
    res = sess.run(jieba_cut_t, feed_dict={sentence_t: sentence_in})
    print("Raw output: {}".format(res))
    print("Output: {}".format(res.decode('utf-8')))
sess.close()
