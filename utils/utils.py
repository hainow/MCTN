import os
import pickle
import sys

import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def dump_obj(obj, name, output_dir='output'):
  """ Dump to pickle """
  save_dir = output_dir
  if not os.path.exists(save_dir): os.mkdir(save_dir)
  print("Saving into {}".format(save_dir))

  with open(os.path.join(save_dir, name + '.pkl'), 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, output_dir='output'):
  """ Load from pickle binary file """
  save_dir = output_dir
  with open(os.path.join(save_dir, name + '.pkl'), 'rb') as f:
    return pickle.load(f)


def convert_labels_or_predictions(predictions, num_classes=2):
  predictions = np.round(predictions)

  if num_classes == 5:
    converted_preds = []
    for p in predictions:
      if p <= -2:
        converted_preds.append(-2)
      elif p >= 2:
        converted_preds.append(2)
      else:
        converted_preds.append(p)
  elif num_classes == 7:
    converted_preds = []
    for p in predictions:
      if p <= -3:
        converted_preds.append(-3)
      elif p >= 3:
        converted_preds.append(3)
      else:
        converted_preds.append(p)
  elif num_classes == 2:
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = 0
  else:
    raise NotImplementedError

  return np.array(predictions, dtype=np.int)  # convert to int to use as index


def get_confusion_matrix(labels, predictions, is_binary):
  label_list = list(set(labels))

  confusion_top1 = [[0 for _ in label_list]
                    for _ in label_list]

  # populate confusion matrix
  for j in range(len(predictions)):
    confusion_top1[labels[j]][predictions[j]] += 1

  # dump to disk
  # save confusion matrice
  print("Dumping confusion matrices to disk...")
  desc = 'bin' if is_binary else '7_classes'
  dump_obj(confusion_top1, "conf_top1_" + desc)


# custom callback
class TestCallback(Callback):
  # https://github.com/keras-team/keras/issues/2548
  def __init__(self, test_data):
    Callback.__init__(self)
    self.test_data = test_data
    self.count = 0

  def on_epoch_end(self, epoch, logs={}):
    if epoch % 10 == 0 and epoch > 0:
      x, y = self.test_data
      score = self.model.evaluate(x, y, verbose=0)
      print('\t\tTesting MSE: {}\n'.format(score))


class GetJointRepresentationCallback(Callback):
  # https://github.com/keras-team/keras/issues/2548
  def __init__(self, test_data, train_data, is_bidirectional, infeat, outfeat,
               model_type):
    Callback.__init__(self)
    self.test_data = test_data
    self.train_data = train_data
    self.count = 0
    self.idx = 1 if is_bidirectional else 5
    self.infeat = infeat
    self.outfeat = outfeat
    self.description = model_type

  def on_epoch_end(self, epoch, logs={}):
    if epoch % 10 == 0 and epoch > 0:
      input_train, output_train = self.train_data
      input_test, output_test = self.test_data

      get_encoder_output = K.function(
        [self.model.layers[0].input, K.learning_phase()],
        [self.model.layers[self.idx].output])

      # test flag is 0
      train_encoder_output = get_encoder_output([input_train, 0])[0]
      test_encoder_output = get_encoder_output([input_test, 0])[0]
      print("Train encoder output: {}".format(train_encoder_output.shape))
      print("Test encoder output: {}".format(test_encoder_output.shape))

      # dumping to disk
      dump_obj(train_encoder_output, "{}_train_encoder_{}_{}_{}".
               format(self.description, self.infeat, self.outfeat, epoch))
      dump_obj(test_encoder_output, "{}_test_encoder_{}_{}_{}".
               format(self.description, self.infeat, self.outfeat, epoch))
      if epoch == 10:
        dump_obj(output_train, "y_train")
        dump_obj(output_test, "y_test")
      print("\tALL SAVED TO DISK")


def get_preds_statistics(predictions, y_test):
  """
  Get MAE, CORR and Accuracy 
  
  Args:
    predictions: 
    y_test: labels 

  """
  mae = np.mean(np.absolute(predictions - y_test))
  print("mae: {}".format(mae))
  corr = np.corrcoef(predictions, y_test)[0][1]
  print("corr: {}".format(corr))
  mult = round(
    sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
  print("mult_acc: {}".format(mult))
  f_score = round(f1_score(np.round(predictions),
                           np.round(y_test), average='weighted'),
                  5)
  print("mult f_score: {}".format(f_score))
  true_label = (y_test >= 0)
  predicted_label = (predictions >= 0)
  print("Confusion Matrix :")
  print(confusion_matrix(true_label, predicted_label))
  print("Classification Report :")
  print(classification_report(true_label, predicted_label, digits=5))
  print("Accuracy: {}".format(accuracy_score(true_label, predicted_label)))
  sys.stdout.flush()
