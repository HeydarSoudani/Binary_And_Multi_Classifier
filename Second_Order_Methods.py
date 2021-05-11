from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from SOM_problem import loss
import Gradient_Descent
import Newton
import Natural_Gradient

np.random.seed(0)

if __name__ == "__main__":
  
  feature_dim = 2
  class_size = 100
  train_num = 150
  test_num = 50

  """
  ### Generating data -------------------------------------
  """
  # -- Class 0 -----
  mean_0 = np.array([-1, -1])
  cov_0 = np.identity(feature_dim)
  class_0 = np.random.multivariate_normal(mean_0, cov_0, class_size)
  label_0 = np.array([0 for i in range(class_size)])
  data_0 = np.column_stack((class_0, label_0))

  # -- Class 1 -----
  mean_1 = np.array([1, 1])
  cov_1 = np.identity(feature_dim)
  class_1 = np.random.multivariate_normal(mean_1, cov_1, class_size)
  label_1 = np.array([1 for i in range(class_size)])
  data_1 = np.column_stack((class_1, label_1))

  # -- Stack classes and shuffle data randomly -----
  data = np.vstack((data_0, data_1))
  data = np.take(data, np.random.permutation(data.shape[0]), axis=0, out=data)
  
  # -- Seprate train and test data -----
  data_train = data[0:train_num,:]
  data_test = data[train_num:,:]

  # -- Seprate train and test data -----
  x_train = data_train[:, 0:-1]
  y_train = data_train[:, -1]
  x_test = data_test[:, 0:-1]
  y_test = data_test[:, -1]

  # init w
  # w = np.array([0., 0.])
  w_init = np.random.rand(2)


  """
  ### Part2: Surface diagram of loss function --------------
  """
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  
  w1 = np.arange(-2, 2, 0.1)
  w2 = np.arange(-2, 2, 0.1)
  loss_arr = np.zeros((40, 40), dtype=float)

  for i in range(w1.shape[0]):
    for j in range(w2.shape[0]):
      w = np.array([w1[i], w2[j]])
      x = data[:, 0:2]
      y_act = data[:, -1]
      loss_arr[i, j] = loss(w, x, y_act)
  
  w1, w2 = np.meshgrid(w1, w2)
  surf = ax.plot_surface(w1, w2, loss_arr, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  plt.title('Loss function')
  ax.set_xlabel('w1')
  ax.set_ylabel('w2')
  ax.set_zlabel('loss')
  plt.show()


  """
  ### Part3: Gradient Descent -------------------------------
  """
  lr = 0.2
  iteration = 5

  gd = Gradient_Descent.gd(x_train, y_train, w_init=w_init, iteration=iteration, lr=lr)
  gd.run()
  y_predict = gd.predict(x_test)
  print('SGD accuracy is: {:.2f}%'.format(gd.accuracy(y_test, y_predict)))


  """
  ### Part4: Newton -----------------------------------------
  """
  alpha = 0.2
  beta = 0.9
  iteration = 5

  newton = Newton.newton(x_train, y_train, w_init=w_init, iteration=iteration, alpha=alpha, beta=beta)
  newton.run()
  y_predict = newton.predict(x_test)
  print('Newton accuracy is: {:.2f}%'.format(newton.accuracy(y_test, y_predict)))


  """
  ### Part6: Natural Gradient -------------------------------
  """
  lr = 0.2
  iteration = 5

  ng = Natural_Gradient.ng(x_train, y_train, w_init=w_init, iteration=iteration, lr=lr)
  ng.run()
  y_predict = ng.predict(x_test)
  print('Natural Gradient accuracy is: {:.2f}%'.format(ng.accuracy(y_test, y_predict)))


  """
  ### Part7: Surface diagram of loss function with change in parameter space -------------------------------
  """
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  
  w1 = np.arange(-2, 2, 0.1)
  w2 = np.arange(-2, 2, 0.1)
  loss_arr = np.zeros((40, 40), dtype=float)

  for i in range(w1.shape[0]):
    for j in range(w2.shape[0]):
      w = np.array([w1[i], w2[j]])
      x = data[:, 0:2]
      y_act = data[:, -1]
      loss_arr[i, j] = loss(w, x, y_act, coefficient=0.01)
  
  w1, w2 = np.meshgrid(w1, w2)
  surf = ax.plot_surface(w1, w2, loss_arr, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  plt.title('Loss function')
  ax.set_xlabel('w1')
  ax.set_ylabel('w2')
  ax.set_zlabel('loss')
  plt.show()


  """
  ### Part8: Gradient Descent (with change in parameter space) -------------------------------
  """
  lr = 0.2
  iteration = 5

  gd_ps = Gradient_Descent.gd(x_train, y_train, w_init=w_init, iteration=iteration, lr=lr, ps_coeff=0.01)
  gd_ps.run()
  y_predict = gd_ps.predict(x_test)
  print('SGD accuracy (Change in param-space) is: {:.2f}%'.format(gd.accuracy(y_test, y_predict)))

  # plot
  plt.title('Gradient Descent')
  plt.ylabel('Loss')
  plt.xlabel('Iteration')  
  gd.loss_changes_diagram()
  gd_ps.loss_changes_diagram()
  plt.xticks(np.arange(0, 6, step=1))
  plt.yticks(np.arange(0, 1.1, step=0.2))
  plt.legend()
  plt.show()


  """
  ### Part9: Newton (with change in parameter space) -------------------------------
  """
  alpha = 0.2
  beta = 0.99
  iteration = 5

  newton_ps = Newton.newton(x_train, y_train, w_init=w_init, iteration=iteration, alpha=alpha, beta=beta, ps_coeff=0.01)
  newton_ps.run()
  y_predict = newton_ps.predict(x_test)
  print('Newton accuracy (Change in param-space) is: {:.2f}%'.format(newton_ps.accuracy(y_test, y_predict)))

  # plot
  plt.title('Newton')
  plt.ylabel('Loss')
  plt.xlabel('Iteration')  
  newton.loss_changes_diagram()
  newton_ps.loss_changes_diagram()
  plt.xticks(np.arange(0, 6, step=1))
  plt.yticks(np.arange(0, 1.1, step=0.2))
  plt.legend()
  plt.show()


  """
  ### Part10: Natural Gradient (with change in parameter space) -------------------------------
  """
  lr = 0.2
  iteration = 5

  ng_ps = Natural_Gradient.ng(x_train, y_train, w_init=w_init, iteration=iteration, lr=lr, ps_coeff=0.01)
  ng_ps.run()
  y_predict = ng_ps.predict(x_test)
  print('Natural Gradient accuracy (Change in param-space) is: {:.2f}%'.format(ng_ps.accuracy(y_test, y_predict)))

  # plot
  plt.title('Natural Gradient')
  plt.ylabel('Loss')
  plt.xlabel('Iteration')  
  ng.loss_changes_diagram()
  ng_ps.loss_changes_diagram()
  plt.xticks(np.arange(0, 6, step=1))
  plt.yticks(np.arange(0, 1.1, step=0.2))
  plt.legend()
  plt.show()


  """
  ### Part12: Plot likelihood for Natural Gradient -------------------------------
  """
  ng_ps.plot_liklihood()
  