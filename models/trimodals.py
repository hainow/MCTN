from keras import Model
from seq2seq.mctn_models import mctn_level2_model
from seq2seq.mctn_models import mctn_model

from bimodals import BaseModel
from optimizers import get_optimizers
from regression_model import create_regression_model


class E2E_Hierarchical_MCTN_Model(BaseModel):
  """End to end MCTN Trimodal, also a wrapper for data """

  def __init__(self, configs, features, feats_dict):
    super(E2E_Hierarchical_MCTN_Model, self).__init__(configs,
                                                      features,
                                                      feats_dict)

    # this model must have 3 features (modalities)
    self._first = self.features[0]
    self._second = self.features[1]
    self._third = self.features[2]

    self.train_first = self.feats_dict[self._first][0]
    self.test_first = self.feats_dict[self._first][1]

    self.train_second = self.feats_dict[self._second][0]
    self.test_second = self.feats_dict[self._second][1]

    self.train_third = self.feats_dict[self._third][0]
    self.test_third = self.feats_dict[self._third][1]

    # update self.model object
    self.is_cycled = configs['translation1']['is_cycled']
    self._create_model()

  def _create_model(self):
    print("\n*** HIERARHICAL SEQ2SEQ translates from {} -> {} -> {} ***".
          format(self.feats_dict[self._first][2],
                 self.feats_dict[self._second][2],
                 self.feats_dict[self._third][2]))

    # --------------------Seq2Seq network params--------------------------------
    n_samples = self.train_first.shape[0]
    input_length = self.train_first.shape[1]
    input_dim = self.train_first.shape[2]
    output_length = self.train_second.shape[1]
    output_dim = self.train_second.shape[2]

    # --------------- MODEL TRANSLATION DEFINITION -----------------------------
    print("Creating LEVEL 1 TRANSLATION SEQ2SEQ model ...")
    inputs, encoded_seq, decoded_seq, cycled_decoded_seq = \
      mctn_model(output_dim=output_dim,
                 hidden_dim=self.configs['translation1']['hidden_dim'],
                 output_length=output_length,
                 input_dim=input_dim,
                 input_length=input_length,
                 depth=self.configs['translation1']['depth'],
                 bidirectional=self.configs['translation1']['is_bidirectional'],
                 is_cycled=self.is_cycled)
    # -------------

    input_length_2 = input_length  # == encoded_seq.shape[1]
    input_dim_2 = self.configs['translation1']['hidden_dim']
    output_length_2 = self.train_third.shape[1]
    output_dim_2 = self.train_third.shape[2]

    print("Creating LEVEL 2 TRANSLATION SEQ2SEQ model ...")
    inputs_2, encoded_seq_2, decoded_seq_2 = \
      mctn_level2_model(input=encoded_seq,
                        output_dim=output_dim_2,
                        hidden_dim=self.configs['translation2']['hidden_dim'],
                        output_length=output_length_2,
                        input_dim=input_dim_2,
                        input_length=input_length_2,
                        depth=self.configs['translation2']['depth'],
                        bidirectional=self.configs['translation2'][
                          'is_bidirectional'])

    # ---------------- MODEL REGRESSION DEFINITION -----------------------------
    print("Creating REGRESSION model ...")
    regression_score = \
      create_regression_model(
        n_hidden=self.configs['regression']['reg_hidden_dim'],
        input=encoded_seq_2,
        l2_factor=self.configs['regression']['l2_factor'])

    # ------------------ E2E REGRESSION DEFINITION -----------------------------
    print("BUILDING A JOINT END-TO-END MODEL")
    if self.is_cycled:
      outputs = [decoded_seq,
                 cycled_decoded_seq,
                 decoded_seq_2,
                 regression_score]
      losses = [self.configs['translation1']['loss_type'],
                self.configs['translation1']['cycle_loss_type'],
                self.configs['translation2']['loss_type'],
                self.configs['regression']['loss_type']
                ]
      losses_weights = [self.configs['translation1']['loss_weight'],
                        self.configs['translation1']['cycle_loss_weight'],
                        self.configs['translation2']['loss_weight'],
                        self.configs['regression']['loss_weight']
                        ]
    else:
      outputs = [decoded_seq,
                 decoded_seq_2,
                 regression_score]
      losses = [self.configs['translation1']['loss_type'],
                self.configs['translation2']['loss_type'],
                self.configs['regression']['loss_type']
                ]
      losses_weights = [self.configs['translation1']['loss_weight'],
                        self.configs['translation2']['loss_weight'],
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
      output_feeds = [self.train_second,
                      self.train_first,
                      self.train_third,
                      self.feats_dict['train_labels']
                      ]
    else:
      output_feeds = [self.train_second,
                      self.train_third,
                      self.feats_dict['train_labels']
                      ]
    self.model.fit(x=[self.train_first],
                   y=output_feeds,
                   epochs=n_epochs,
                   validation_split=val_split,
                   batch_size=batch_size,
                   verbose=is_verbose,
                   callbacks=callbacks
                   )

  def predict(self):
    print('Predicting the stored test input')
    preds = self.model.predict(self.test_first)

    return preds[-1]

  def evaluate(self, is_verbose=True):
    if self.is_cycled:
      output_feeds = [self.test_second,
                      self.test_first,
                      self.test_third,
                      self.feats_dict['test_labels']
                      ]
    else:
      output_feeds = [self.test_second,
                      self.test_third,
                      self.feats_dict['test_labels']
                      ]
    print('Evaluating the stored test input')
    self.model.evaluate(x=[self.test_first],
                        y=output_feeds,
                        verbose=is_verbose)
