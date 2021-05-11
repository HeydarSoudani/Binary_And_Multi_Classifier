import numpy as np
import matplotlib.pyplot as plt

from functions import sigmoid
from SOM_problem import loss, grad_loss, point_likelihood

class ng():
  def __init__(self, x, y, w_init=np.array([0., 0.]), lr=0, iteration=10, ps_coeff=1):
    self.x = x
    self.y = y
    self.w = np.copy(w_init)

    self.lr = lr
    self.iteration = iteration
    self.ps_coeff = ps_coeff

    self.loss_list = list()
    self.likelihood_list = list()

  
  def fisher(self):
    # score_func = self.ps_coeff * self.x.T * (self.y-sigmoid(self.ps_coeff*np.matmul(self.x, self.w.T)))
    # return np.cov(score_func)
    M = self.y.shape[0]
    yhat = sigmoid(self.ps_coeff * self.x @ self.w)
    grad_ll = self.ps_coeff * ((self.y - yhat) * self.x.T).T
    return np.sum([np.outer(grad_ll[i], grad_ll[i]) for i in range(M)], axis=0) / M

  def run(self):
    # save loss and liklihood in lists
    data_num = self.x.shape[0]
    self.loss_list.append(loss(self.w, self.x, self.y, coefficient=self.ps_coeff))
    self.likelihood_list.append([point_likelihood(self.w, self.x[i], self.y[i], coefficient=self.ps_coeff) for i in range(data_num)])

    for _ in range(self.iteration):
      # Compute Fisher Information Matrix
      F = self.fisher()

      # Compute the natural gradient
      nat_grad = np.matmul(np.linalg.inv(F), grad_loss(self.w, self.x, self.y, coefficient=self.ps_coeff))

      # Update w
      self.w -= self.lr*nat_grad

      # save loss and liklihood in lists
      self.loss_list.append(loss(self.w, self.x, self.y, coefficient=self.ps_coeff))
      self.likelihood_list.append([point_likelihood(self.w, self.x[i], self.y[i], coefficient=self.ps_coeff) for i in range(data_num)])


  def predict(self, x_test):
    y_pred = sigmoid(self.ps_coeff*np.matmul(x_test, self.w.T))
    y_pred[np.where(y_pred >= 1/2)] = 1
    y_pred[np.where(y_pred < 1/2)] = 0
    return y_pred


  def accuracy(self, y_act, y_pred):
    return (len(np.where(y_act == y_pred)[0])/y_act.shape[0])*100

 
  def loss_changes_diagram(self):
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.plot([i for i in range(self.iteration+1)], self.loss_list, label='ps coeff: '+str(self.ps_coeff))


  def plot_liklihood(self):
    plt.style.use('seaborn-deep')

    for i in range(self.iteration):
      before = self.likelihood_list[i]
      after = self.likelihood_list[i+1]

      plt.subplot(2, 3, i+1)
      plt.title('Step '+str(i+1))
      plt.hist([before, after], label=['before', 'after'])
      plt.legend(loc='upper left')
      plt.ylabel('Counts')
      plt.xlabel('Likelihood')
      plt.xticks(np.arange(0, 1.1, step=0.1))
      plt.yticks(np.arange(0, 140, step=20))

    plt.show()
