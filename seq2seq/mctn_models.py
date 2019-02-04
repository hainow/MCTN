from __future__ import absolute_import

from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Input
from recurrentshop import LSTMCell
from recurrentshop import RecurrentSequential

from .cells import AttentionDecoderCell
from .cells import LSTMDecoderCell


def mctn_model(output_dim,
               output_length,
               batch_input_shape=None,
               batch_size=None,
               input_shape=None,
               input_length=None,
               input_dim=None,
               hidden_dim=None,
               depth=1,
               bidirectional=True,
               unroll=False,
               stateful=False,
               dropout=0.0,
               is_cycled=True
               ):
  """
  MCTN Model (by default with Cycle Consistency Loss) 
  """
  if isinstance(depth, int):
    depth = (depth, depth)
  if batch_input_shape:
    shape = batch_input_shape
  elif input_shape:
    shape = (batch_size,) + input_shape
  elif input_dim:
    if input_length:
      shape = (batch_size,) + (input_length,) + (input_dim,)
    else:
      shape = (batch_size,) + (None,) + (input_dim,)
  else:
    # TODO Proper error message
    raise TypeError
  if hidden_dim is None:
    hidden_dim = output_dim

  _input = Input(batch_shape=shape)
  _input._keras_history[0].supports_masking = True

  # encoder phase
  encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                return_sequences=True)
  encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))

  for _ in range(1, depth[0]):
    encoder.add(Dropout(dropout))
    encoder.add(LSTMCell(hidden_dim))

  if bidirectional:
    encoder = Bidirectional(encoder, merge_mode='sum')
    encoder.forward_layer.build(shape)
    encoder.backward_layer.build(shape)
    # patch
    encoder.layer = encoder.forward_layer

  encoded = encoder(_input)

  # decoder phase
  decoder = RecurrentSequential(decode=True, output_length=output_length,
                                unroll=unroll, stateful=stateful)
  decoder.add(
    Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
  if depth[1] == 1:
    decoder.add(
      AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
  else:
    decoder.add(
      AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    for _ in range(depth[1] - 2):
      decoder.add(Dropout(dropout))
      decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
    decoder.add(Dropout(dropout))
    decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))

  inputs = [_input]
  decoded = decoder(encoded)

  # cycle phase
  cycled_decoded = None
  if is_cycled:
    cycled_encoded = encoder(decoded)
    cycled_decoded = decoder(cycled_encoded)

  return inputs, encoded, decoded, cycled_decoded


def mctn_level2_model(input,
                      output_dim,
                      output_length,
                      batch_input_shape=None,
                      batch_size=None,
                      input_shape=None,
                      input_length=None,
                      input_dim=None,
                      hidden_dim=None,
                      depth=1,
                      bidirectional=True,
                      unroll=False,
                      stateful=False,
                      dropout=0.0):
  """ 
  Level 2 MCTN used for translation between the joint embedded of 
  2 modalities to the third one. Due to the lack of ground truth, no 
  cycle phase happens
  """
  if isinstance(depth, int):
    depth = (depth, depth)
  if batch_input_shape:
    shape = batch_input_shape
  elif input_shape:
    shape = (batch_size,) + input_shape
  elif input_dim:
    if input_length:
      shape = (batch_size,) + (input_length,) + (input_dim,)
    else:
      shape = (batch_size,) + (None,) + (input_dim,)
  else:
    # TODO Proper error message
    raise

  if hidden_dim is None:
    hidden_dim = output_dim

  encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                return_sequences=True)
  encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))

  for _ in range(1, depth[0]):
    encoder.add(Dropout(dropout))
    encoder.add(LSTMCell(hidden_dim))

  if bidirectional:
    encoder = Bidirectional(encoder, merge_mode='sum')
    encoder.forward_layer.build(shape)
    encoder.backward_layer.build(shape)
    # patch
    encoder.layer = encoder.forward_layer

  encoded = encoder(input)
  decoder = RecurrentSequential(decode=True, output_length=output_length,
                                unroll=unroll, stateful=stateful)
  decoder.add(
    Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
  if depth[1] == 1:
    decoder.add(
      AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
  else:
    decoder.add(
      AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    for _ in range(depth[1] - 2):
      decoder.add(Dropout(dropout))
      decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
    decoder.add(Dropout(dropout))
    decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))

  inputs = [input]
  decoded = decoder(encoded)

  return inputs, encoded, decoded


def paired_trimodal_model(output_dim,
                          output_length,
                          batch_input_shape=None,
                          batch_size=None,
                          input_shape=None,
                          input_length=None,
                          input_dim=None,
                          hidden_dim=None,
                          depth=1,
                          bidirectional=True,
                          unroll=False,
                          stateful=False,
                          dropout=0.0):
  """
  One modal translates into two other modalities, no cycle involved 
  The model has 1 encoder and 2 decoders 
  """
  if isinstance(depth, int):
    depth = (depth, depth)
  if batch_input_shape:
    shape = batch_input_shape
  elif input_shape:
    shape = (batch_size,) + input_shape
  elif input_dim:
    if input_length:
      shape = (batch_size,) + (input_length,) + (input_dim,)
    else:
      shape = (batch_size,) + (None,) + (input_dim,)
  else:
    # TODO Proper error message
    raise TypeError

  if hidden_dim is None:
    hidden_dim = output_dim

  _input = Input(batch_shape=shape)
  _input._keras_history[0].supports_masking = True

  # encoder phase
  encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                return_sequences=True)
  encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))

  # encoder phase
  encoder_2 = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  return_sequences=True)
  encoder_2.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], output_dim)))

  for _ in range(1, depth[0]):
    encoder.add(Dropout(dropout))
    encoder.add(LSTMCell(hidden_dim))

    encoder_2.add(Dropout(dropout))
    encoder_2.add(LSTMCell(hidden_dim))

  if bidirectional:
    encoder = Bidirectional(encoder, merge_mode='sum')
    encoder.forward_layer.build(shape)
    encoder.backward_layer.build(shape)
    # patch
    encoder.layer = encoder.forward_layer

    encoder_2 = Bidirectional(encoder_2, merge_mode='sum')
    encoder_2.forward_layer.build(shape)
    encoder_2.backward_layer.build(shape)
    # patch
    encoder_2.layer = encoder_2.forward_layer

  encoded_one = encoder(_input)

  # decoder phase
  decoder = RecurrentSequential(decode=True, output_length=output_length,
                                unroll=unroll, stateful=stateful)
  decoder.add(
    Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))

  decoder_2 = RecurrentSequential(decode=True, output_length=input_length,
                                  unroll=unroll, stateful=stateful)
  decoder_2.add(
    Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))

  if depth[1] == 1:
    decoder.add(
      AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
  else:
    decoder.add(
      AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    for _ in range(depth[1] - 2):
      decoder.add(Dropout(dropout))
      decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
    decoder.add(Dropout(dropout))
    decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))

  if depth[1] == 1:
    decoder_2.add(
      AttentionDecoderCell(output_dim=input_dim, hidden_dim=hidden_dim))
  else:
    decoder_2.add(
      AttentionDecoderCell(output_dim=input_dim, hidden_dim=hidden_dim))
    for _ in range(depth[1] - 2):
      decoder_2.add(Dropout(dropout))
      decoder_2.add(LSTMDecoderCell(output_dim=hidden_dim,
                                    hidden_dim=hidden_dim))
    decoder_2.add(Dropout(dropout))
    decoder_2.add(LSTMDecoderCell(output_dim=input_dim,
                                  hidden_dim=hidden_dim))

  inputs = [_input]
  decoded_one = decoder(encoded_one)

  encoded_two = encoder_2(decoded_one)
  decoded_two = decoder_2(encoded_two)

  return inputs, encoded_one, encoded_two, decoded_one, decoded_two


