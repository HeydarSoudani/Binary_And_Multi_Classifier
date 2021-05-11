import numpy as np
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

import SGD
import SVRG

np.random.seed(1)

if __name__ == "__main__":
  
  gamma = 1e-4

  M = 6000 # train data size
  train_dataset_size = 60000
  test_dataset_size = 10000
  test_data_size = 1000

  feature_dim = 28*28
  class_num = 10

  """
  ### Loading & Preparing Data -------------------------------------
  """
  # Loading train & test data from MNIST dataset
  mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
  mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
  
  X_train_arr = mnist_trainset.data.numpy()
  X_tr = X_train_arr.reshape(X_train_arr.shape[0], -1)
  y_tr = mnist_trainset.targets.numpy()
  
  X_test_arr = mnist_testset.data.numpy()
  X_te = X_test_arr.reshape(X_test_arr.shape[0], -1)
  y_te = mnist_testset.targets.numpy()

  # Separating M data from whole dataset
  train_indices = np.random.choice(train_dataset_size, M)
  test_indices = np.random.choice(test_dataset_size, test_data_size)
  x_train_n = X_tr[train_indices]
  y_train = y_tr[train_indices]
  x_test_n = X_te[test_indices]
  y_test = y_te[test_indices]

  # Normalization on feature set
  x_min = 0
  x_max = 255
  x_train = (x_train_n-x_min)/(x_max-x_min)
  x_test = (x_test_n-x_min)/(x_max-x_min)

  # stack data and labels in one array
  train_data = np.column_stack((x_train, y_train))
  test_data = np.column_stack((x_test, y_test))
  
  # Creating randomized weights & biases
  w = np.random.rand(class_num, feature_dim)
  b = np.random.rand(class_num, 1)


  """
  ### Part1: SGD ---------------------------------------
  """
  t = 100
  landa = [0.01, 0.05]
  batch_size = 1

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

  for lr in landa:
    print("Run SGD with learning rate: {}.".format(lr))
    sgd = SGD.sgd(train_data, test_data, w, b, batch_size=batch_size, lr=lr, t=t, gamma=gamma)
    sgd.run()
    
    sgd.plot_loss(ax1, label='lr: '+str(lr))
    sgd.plot_eval_data_accuracy(ax2, label='lr: '+str(lr))
    sgd.plot_steps_variance(ax3, label='lr: '+str(lr))
  
  ax3.set_yscale('log')
  ax1.title.set_text('Loss')
  ax2.title.set_text('Accuracy')
  ax3.title.set_text('Variance')
  ax1.legend()
  ax2.legend()
  ax3.legend()
  plt.show()
  
  
  """
  ### Part2: mini-batch SGD ----------------------------
  """
  t = 100
  landa = [0.01, 0.04, 0.2]
  batch_size = 4

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

  for lr in landa:
    print("Run SGD with learning rate: {}.".format(lr))
    sgd = SGD.sgd(train_data, test_data, w, b, batch_size=batch_size, lr=lr, t=t, gamma=gamma)
    sgd.run()
    
    sgd.plot_loss(ax1, label='lr: '+str(lr))
    sgd.plot_eval_data_accuracy(ax2, label='lr: '+str(lr))
    sgd.plot_steps_variance(ax3, label='lr: '+str(lr))

  ax1.title.set_text('Loss')
  ax2.title.set_text('Accuracy')
  ax3.title.set_text('Variance')
  ax3.set_yscale('log')
  ax1.legend()
  ax2.legend()
  ax3.legend()
  plt.show()

  
  """
  ### Part4: SVRG --------------------------------------
  """
  m = M
  landa = [0.01, 0.05]
  s = 100
  option = '1'

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

  for lr in landa:
    print("Run SVRG with learning rate: {}.".format(lr))
    svrg = SVRG.svrg(train_data, test_data, w, b, m=m, lr=lr, s=s, option=option, gamma=gamma)
    svrg.run()

    svrg.plot_loss(ax1, label='lr: '+str(lr))
    svrg.plot_eval_data_accuracy(ax2, label='lr: '+str(lr))
    svrg.plot_steps_variance(ax3, label='lr: '+str(lr))

  ax1.title.set_text('Loss')
  ax2.title.set_text('Accuracy')
  ax3.title.set_text('Variance')
  ax3.set_yscale('log')
  ax1.legend()
  ax2.legend()
  ax3.legend()
  plt.show()


  """
  ### Part6: Change m parameter in SVRG --------------------------------------
  """
  landa = [0.01, 0.05]
  s = 100
  option = '1'

  # Decrease m
  m = int(M/2)
 
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

  for lr in landa:
    print("Run SVRG with learning rate: {}.".format(lr))
    svrg = SVRG.svrg(train_data, test_data, w, b, m=m, lr=lr, s=s, option=option, gamma=gamma)
    svrg.run()

    svrg.plot_loss(ax1, label='lr: '+str(lr))
    svrg.plot_eval_data_accuracy(ax2, label='lr: '+str(lr))
    svrg.plot_steps_variance(ax3, label='lr: '+str(lr))

  fig.suptitle('SVRG (m = M/2)')
  ax1.title.set_text('Loss')
  ax2.title.set_text('Accuracy')
  ax3.title.set_text('Variance')
  ax3.set_yscale('log')
  ax1.legend()
  ax2.legend()
  ax3.legend()
  plt.show()

  # Increase m
  m = int(M*2)
 
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

  for lr in landa:
    print("Run SVRG with learning rate: {}.".format(lr))
    svrg = SVRG.svrg(train_data, test_data, w, b, m=m, lr=lr, s=s, option=option, gamma=gamma)
    svrg.run()

    svrg.plot_loss(ax1, label='lr: '+str(lr))
    svrg.plot_eval_data_accuracy(ax2, label='lr: '+str(lr))
    svrg.plot_steps_variance(ax3, label='lr: '+str(lr))

  fig.suptitle('SVRG (m = 2*M)')
  ax1.title.set_text('Loss')
  ax2.title.set_text('Accuracy')
  ax3.title.set_text('Variance')
  ax3.set_yscale('log')
  ax1.legend()
  ax2.legend()
  ax3.legend()
  plt.show()
