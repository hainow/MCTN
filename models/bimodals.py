from keras import Model
from seq2seq.mctn_models import mctn_model

from optimizers import get_optimizers
from regression_model import create_regression_model


class BaseModel(object):
  def __init__(self, configs, features, feats_dict):
    self.configs = configs
    self.features = features
    self.feats_dict = feats_dict

    # to be updated by child classes
    self.model = None


class E2E_MCTN_Model(BaseModel):
  """End to end MCTN Bimodal, also a wrapper for data"""

  def __init__(self, configs, features, feats_dict):
    super(E2E_MCTN_Model, self).__init__(configs, features, feats_dict)

    # this model only has 2 features (modalities)
    self._in = self.features[0]
    self._out = self.features[1]

    self.input_train = self.feats_dict[self._in][0]
    self.input_test = self.feats_dict[self._in][1]
    self.output_train = self.feats_dict[self._out][0]
    self.output_test = self.feats_dict[self._out][1]

    # update self.model object
    self.is_cycled = configs['translation']['is_cycled']
    self._create_model()

  def _create_model(self):
    print("\n***SEQ2SEQ translates from {} to {}***".
          format(self.feats_dict[self._in][2],
                 self.feats_dict[self._out][2]))

    input_dims = self.input_train.shape
    output_dims = self.output_train.shape
    print("Input Dims = {}".format(input_dims))
    print("Output (Translation) Dims = {}".format(output_dims))
    print("Output (Regression) Dims = {}\n".format(self.feats_dict[
                                                     'train_labels'].shape))
    # --------------------Seq2Seq network params--------------------------------
    n_samples = input_dims[0]
    input_length = input_dims[1]
    input_dim = input_dims[2]
    output_length = output_dims[1]
    output_dim = output_dims[2]

    # --------------- MODEL TRANSLATION DEFINITION -----------------------------
    print("Creating TRANSLATION SEQ2SEQ model ...")
    inputs, encoded_seq, decoded_seq, cycled_decoded_seq = \
      mctn_model(output_dim=output_dim,
                 hidden_dim=self.configs['translation']['hidden_dim'],
                 output_length=output_length,
                 input_dim=input_dim,
                 input_length=input_length,
                 depth=self.configs['translation']['depth'],
                 bidirectional=self.configs['translation']['is_bidirectional'],
                 is_cycled=self.is_cycled)

    # ---------------- MODEL REGRESSION DEFINITION -----------------------------
    print("Creating REGRESSION model ...")
    regression_score = \
      create_regression_model(
        n_hidden=self.configs['regression']['reg_hidden_dim'],
        input=encoded_seq,
        l2_factor=self.configs['regression']['l2_factor'])

    # ------------------ E2E REGRESSION DEFINITION -----------------------------
    print("BUILDING A JOINT END-TO-END MODEL")
    if self.is_cycled:
      outputs = [decoded_seq, cycled_decoded_seq, regression_score]
      losses = [self.configs['translation']['loss_type'],
                self.configs['translation']['cycle_loss_type'],
                self.configs['regression']['loss_type']
                ]
      losses_weights = [self.configs['translation']['loss_weight'],
                        self.configs['translation']['cycle_loss_weight'],
                        self.configs['regression']['loss_weight']
                        ]
    else:
      outputs = [decoded_seq, regression_score]
      losses = [self.configs['translation']['loss_type'],
                self.configs['regression']['loss_type']
                ]
      losses_weights = [self.configs['translation']['loss_weight'],
                        self.configs['regression']['loss_weight']
                        ]

    end2end_model = Model(inputs=inputs,
                          outputs=outputs)
    print("Compiling model")
    optimizer = get_optimizers(opt=self.configs['general']['optim'],
                               init_lr=self.configs['general']['init_lr'])
    end2end_model.compile(loss=losses,
                          loss_weights=losses_weights,
                          optimizer=optimizer)
    print("Model summary:")
    print(end2end_model.summary())
    print("END2END MODEL CREATED!")

    self.model = end2end_model

  def train(self,
            weights_path=None,
            n_epochs=200,
            val_split=2.0 / 3,
            batch_size=32,
            is_verbose=False,
            callbacks=None
            ):
    try:
      self.model.load_weights(weights_path)
      print("\nWeights loaded from {}\n".format(weights_path))
    except:
      print("\nCannot load weight. Training from scratch\n")

    # train now
    if self.is_cycled:
      output_feeds = [self.output_train,
                      self.input_train,
                      self.feats_dict['train_labels']
                      ]
    else:
      output_feeds = [self.output_train,
                      self.feats_dict['train_labels']
                      ]
    self.model.fit(x=[self.input_train],
                   y=output_feeds,
                   epochs=n_epochs,
                   validation_split=val_split,
                   batch_size=batch_size,
                   verbose=is_verbose,
                   callbacks=callbacks)

  def predict(self):
    print('Predicting the stored test input')
    preds = self.model.predict(self.input_test)

    return preds[-1]

  def evaluate(self, is_verbose=True):
    if self.is_cycled:
      output_feeds = [self.output_test,
                      self.input_test,
                      self.feats_dict['test_labels']
                      ]
    else:
      output_feeds = [self.output_test,
                      self.feats_dict['test_labels']
                      ]
    print('Evaluating the stored test input')
    self.model.evaluate(x=[self.input_test],
                        y=output_feeds,
                        verbose=is_verbose)

