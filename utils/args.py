import argparse
import os
import yaml


def parse_args():
  """
  Parsing input arguments from command line 
  """
  parser = argparse.ArgumentParser(description='(H)MCTN Experiments')
  parser.add_argument('--train_epoch', default=2, type=int,
                      help='default is 200 epochs')
  parser.add_argument('--train_patience', default=20, type=int)
  parser.add_argument('--batch_size', default=32, type=int)
  # parser.add_argument('-f', '--feature', default=['t', 'f'], nargs='+',
  parser.add_argument('-f', '--feature', default=['t', 'c', 'f'], nargs='+',
                      help='what features to use besides text. c: covarep; '
                           'f: facet. t: text')
  parser.add_argument('--train_split', default=2.0 / 3, type=float, help='')
  parser.add_argument('--val_split', default=0.1, type=float,
                      help='valitation split in percentage from training set')
  # parser.add_argument('--cfg', default='configs/mctn.yaml',
  parser.add_argument('--cfg', default='configs/hierarchical_mctn.yaml',
                      type=str,
                      help='YAML configuration file')

  args = parser.parse_args()

  # parsing configs from yaml
  assert os.path.exists(args.cfg)
  with open(args.cfg) as f:
    configs = yaml.load(f)

  return args, configs