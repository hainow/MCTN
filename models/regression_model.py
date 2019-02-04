from keras import backend as K
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers import Permute
from keras.layers import RepeatVector
from keras.layers.merge import multiply
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2


def create_regression_model(n_hidden, input, l2_factor):
  """
  RNN, single layer, for regression score, with Attention 
  
  Args:
    n_hidden: number of hidden neurons for this RNN 
    input: taken from Seq2Seq model 
    l2_factor: regularization factor 

  Returns: regression float score 

  """
  activations = LSTM(n_hidden,
                     name='lstm_layer',
                     trainable=True,
                     return_sequences=True)(input)
  attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
  attention = Flatten()(attention)
  attention = Activation('softmax')(attention)
  attention = RepeatVector(n_hidden)(attention)
  attention = Permute([2, 1])(attention)
  # apply the attention
  sent_representation = multiply([activations, attention])
  sent_representation = Lambda(lambda xin: K.sum(xin, axis=1)
                               )(sent_representation)
  regression_score = Dense(1,
                           name='regression_output',
                           kernel_regularizer=l2(l2_factor))(
    sent_representation)

  return regression_score
