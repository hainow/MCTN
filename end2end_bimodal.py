import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from models.bimodals import E2E_MCTN_Model
from utils.args import parse_args
from utils.utils import get_preds_statistics
from utils.data_loader import load_and_preprocess_data

np.random.seed(123)
tf.set_random_seed(456)

# arguments
args, configs = parse_args()

# data load
is_cycled = configs['translation']['is_cycled']
feats_dict = \
  load_and_preprocess_data(max_seq_len=configs['general']['max_seq_len'],
                           train_split=configs['general']['train_split'],
                           is_cycled=is_cycled)

print("FORMING SEQ2SEQ MODEL...")
features = args.feature  # e.g. ['a', 't']
assert len(features) == 2, 'Wrong number of features'
end2end_model = E2E_MCTN_Model(configs, features, feats_dict)

print("PREP FOR TRAINING...")
filename = '_'.join(args.feature) + "_attention_seq2seq_" + \
           str("bi_directional" if configs['translation']['is_bidirectional']
               else '') + \
           "_bimodal.h5"

output_dir = configs['general']['output_dir']
weights_path = os.path.join(output_dir, filename)
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

callbacks = [
  EarlyStopping(monitor='val_loss', patience=args.train_patience, verbose=0),
  ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True,
                  verbose=1),
]

print("TRAINING NOW...")
end2end_model.train(weights_path=weights_path,
                    n_epochs=args.train_epoch,
                    val_split=args.val_split,
                    batch_size=args.batch_size,
                    callbacks=callbacks)

print("PREDICTING...")
predictions = end2end_model.predict()
predictions = predictions.reshape(-1, )
get_preds_statistics(predictions, feats_dict['test_labels'])
