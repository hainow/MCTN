from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam


def get_optimizers(opt, init_lr):
  """
  Return a keras optimizer object
  
  Args:
    opt (string): any keras optimizers as below
    init_lr (float): initial learning rate 

  Returns: an optimizer object 

  """
  if opt == 'adam':
    return Adam(lr=init_lr)
  if opt == 'rmsprop':
    return RMSprop(lr=init_lr)
  if opt == 'adagrad':
    return Adagrad(lr=init_lr)
  if opt == 'adadelta':
    return Adadelta(lr=init_lr)
  if opt == 'adamax':
    return Adamax(lr=init_lr)
  if opt == 'nadam':
    return Nadam(lr=init_lr)
