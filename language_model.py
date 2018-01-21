from os.path import join

from utils import logger
from constants import *


class LanguageModel(object):
    """
    Pre-trained language model with large volume of data
    """
    def __init__(self, params, word2vec, alpha):
        """
        LanguageModel constructor
        :param params: parameters (weights, biases) defined the language model
        :param word2vec: numpy.array, embedding vector
        :param alpha: int: forgetting factor
        :param params2: used in dual alpha model
        :param alpha2: used in dual alpha model
        :return: LanguageModel object
        """
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.left_context = tf.placeholder(tf.int32, [None, None, None], name="LeftContext")
            self.right_context = tf.placeholder(tf.int32, [None, None, None], name="RightContext")
            self.alpha = tf.constant(
                np.tile(np.float32(alpha) ** np.arange(1024), [1024, 1]),
                dtype=tf.float32, name="Alpha"
            )

            # Embedding vector non-trainable
            self.word2vec = tf.constant(word2vec, name="Embedding")

            batch_size, context_size, ngram = tf.unstack(tf.shape(self.left_context))

            right_alpha = tf.reshape(self.alpha[:batch_size, :context_size], [batch_size, 1, -1])
            left_alpha = tf.reshape(right_alpha[:, :, ::-1], [batch_size, 1, -1])

            left_context = tf.squeeze(
                tf.matmul(
                    left_alpha, 
                    tf.reshape(
                        tf.gather(self.word2vec, self.left_context),
                        [batch_size, context_size, -1]
                    )
                ),
                squeeze_dims=[1], 
                name="LeftContextFOFE"
            )

            right_context = tf.squeeze(
                tf.matmul(
                    right_alpha,
                    tf.reshape(
                        tf.gather(self.word2vec, self.right_context),
                        [batch_size, context_size, -1]
                    )
                ),
                squeeze_dims=[1],
                name="RightContextFOFE"
            )
            projection = tf.concat([left_context, right_context], 1, name="Projection")
            current_out = projection

            for i, param in enumerate(params):
                W = tf.constant(param[0])
                b = tf.constant(param[1])
                current_out = tf.nn.relu(
                    tf.add(tf.matmul(current_out, W), b, name="Linear_%d" % i),
                    name="Relu_%d" % i
                )

            self.bottom_neck = current_out
            init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def infer(self, left_context, right_context):
        """
        Given left_context and right_context of samples, return the predict target
        idx
        Let N be batch size, C be context size, O be ngram size
        :param left_context: numpy.array(N, C, O)
        :param right_context: numpy.array(N, C, O)
        :return: numpy.array(N, M), M is the bottom neck layer size.
        """
        predicted = self.session.run(
            self.bottom_neck,
            feed_dict={
                self.left_context: left_context,
                self.right_context: right_context,
            }
        )
        return predicted


def load_word_list(path):
    word_list = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            word_list[line.split()[0]] = i
    length = len(word_list)
    word_list[pad] = length
    return word_list


def load_embedding(path):
    rst = np.loadtxt(join(path, 'embedding.txt'), delimiter=' ')
    rst = np.append(rst, np.zeros(rst.shape[1]).reshape(1,-1), axis=0).astype(np.float32) # add padding vector
    logger.info("Embedding dim: %d, %d" % (rst.shape[0], rst.shape[1]))
    return rst


def load_params(path):
    params = []
    hardcode = [2, 4, 6, 8]
    for i in hardcode:
        weight = np.transpose(np.loadtxt(join(path, 'weight_%d.txt' % i), delimiter=' ').astype(np.float32))
        bias = np.loadtxt(join(path, 'bias_%d.txt' % i)).astype(np.float32)
        params.append((weight, bias))
        logger.info("Layer dim: %d" % bias.shape[0])
    return params


def load_language_model(language_model_path, alpha):
    embedding = load_embedding(language_model_path)
    params = load_params(language_model_path)
    return LanguageModel(params, embedding, alpha)
